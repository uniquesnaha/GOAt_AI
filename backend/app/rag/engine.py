"""
GOAt AI RAG serving engine.

Frozen retrieval:
- Qwen/Qwen3-Embedding-0.6B
- 256-dimensional normalized embeddings
- Qdrant HNSW
- BM25
- language-specific weighted RRF
- fused Top-20 parents

Only downstream evidence packing / answer extraction is improved.

No reranker.
No second LLM.
No retrieval-weight changes.
"""

from __future__ import annotations

import threading
import time
import unicodedata

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import bm25s
import pandas as pd
import pyarrow.parquet as pq
import torch

from bm25s.tokenization import Tokenizer
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.generation.streamers import BaseStreamer

from app.config import settings
from app.rag.evidence_quality import (
    evidence_pack_score,
    split_sentences,
    strongest_supporting_unit,
)


ROOT = Path(
    settings.data_root
)


# =============================================================================
# CORPUS PROFILE
# =============================================================================

CORPUS_PROFILE = (
    settings.corpus_profile
    .strip()
    .lower()
)


if CORPUS_PROFILE == "25k":

    CHUNK_ROOT = (
        ROOT
        / "chunks_25k"
        / "fixed_384_96"
    )

    ACTIVE_COLLECTIONS = {
        "ta": "hhgoa_fixed384_ta",
        "hi": "hhgoa_fixed384_hi",
    }


elif CORPUS_PROFILE == "350k":

    CHUNK_ROOT = (
        ROOT
        / "chunks_350k"
        / "fixed_384_96"
    )

    ACTIVE_COLLECTIONS = {
        "ta": "hhgoa_350k_fixed384_ta",
        "hi": "hhgoa_350k_fixed384_hi",
    }


else:

    raise RuntimeError(
        "GOAT_CORPUS_PROFILE must be "
        "'25k' or '350k', got "
        f"{CORPUS_PROFILE!r}"
    )


CHUNKS = {
    "ta":
        CHUNK_ROOT
        / "tamil.parquet",

    "hi":
        CHUNK_ROOT
        / "hindi.parquet",
}


# =============================================================================
# MODELS
# =============================================================================

EMBED_MODEL = (
    "Qwen/Qwen3-Embedding-0.6B"
)

GEN_MODEL = (
    "Qwen/Qwen3-0.6B"
)

EMBED_DIM = 256
QUERY_MAX_LENGTH = 256


# =============================================================================
# FROZEN RETRIEVAL CONFIG
# =============================================================================

CFG = {
    "ta": {
        "collection":
            ACTIVE_COLLECTIONS["ta"],

        "bm25_k1":
            0.90,

        "bm25_b":
            0.40,

        "rrf_k":
            2.0,

        "dense_weight":
            2.0,

        "sparse_weight":
            1.0,
    },

    "hi": {
        "collection":
            ACTIVE_COLLECTIONS["hi"],

        "bm25_k1":
            2.00,

        "bm25_b":
            0.75,

        "rrf_k":
            2.0,

        "dense_weight":
            1.5,

        "sparse_weight":
            1.0,
    },
}


DENSE_CHILD_K = 100
DENSE_PARENT_K = 50

BM25_CHUNK_K = 500
SPARSE_PARENT_K = 50

FUSION_DEPTH = 50
FINAL_PARENT_K = 20

HNSW_EF = 64


# =============================================================================
# CONTEXT / GENERATION
# =============================================================================

# 210 + 2 + 210 = 422.
#
# 430 therefore safely holds two full weak-evidence blocks.
# Strong direct evidence collapses to one block.
CONTEXT_CHAR_BUDGET = 430
MAX_CONTEXT_PARENTS = 2
PER_CHUNK_CHARS = 210

# Tiny extractor should not generate paragraphs.
MAX_NEW_TOKENS = 12


# =============================================================================
# TOKENIZATION
# =============================================================================

def word_splitter(text):

    text = str(text).casefold()

    tokens = []
    current = []

    for char in text:

        category = (
            unicodedata.category(char)
        )

        major = (
            category[0]
            if category
            else ""
        )

        if major in {
            "L",
            "M",
            "N",
        }:

            current.append(char)

        elif current:

            tokens.append(
                "".join(current)
            )

            current = []

    if current:

        tokens.append(
            "".join(current)
        )

    return tokens


# =============================================================================
# BM25
# =============================================================================

class BM25Engine:

    def __init__(
        self,
        language,
    ):

        cfg = CFG[
            language
        ]

        path = CHUNKS[
            language
        ]

        schema = (
            pq.read_schema(path)
            .names
        )

        text_col = None

        for candidate in [
            "text",
            "chunk_text",
            "content",
        ]:

            if candidate in schema:

                text_col = candidate
                break


        if text_col is None:

            raise RuntimeError(
                f"No text column in {path}"
            )


        read_columns = [
            "parent_id",
            text_col,
        ]

        has_chunk_id = (
            "chunk_id" in schema
        )

        if has_chunk_id:

            read_columns.append(
                "chunk_id"
            )


        df = pd.read_parquet(
            path,
            columns=read_columns,
        )


        self.parent_ids = (
            df[
                "parent_id"
            ]
            .astype(str)
            .tolist()
        )


        self.chunk_texts = (
            df[
                text_col
            ]
            .fillna("")
            .astype(str)
            .tolist()
        )


        if has_chunk_id:

            self.chunk_ids = (
                df[
                    "chunk_id"
                ]
                .astype(str)
                .tolist()
            )

        else:

            self.chunk_ids = [
                f"{language}_row_{i}"
                for i in range(
                    len(df)
                )
            ]


        self.tokenizer = (
            Tokenizer(
                stemmer=None,
                stopwords=[],
                splitter=
                    word_splitter,
            )
        )


        print(
            f"Building "
            f"{language.upper()} BM25 "
            f"over {len(self.chunk_texts):,} chunks..."
        )


        corpus_tokens = (
            self.tokenizer.tokenize(
                self.chunk_texts,
                return_as=
                    "tuple",
            )
        )


        self.retriever = (
            bm25s.BM25(
                k1=
                    cfg[
                        "bm25_k1"
                    ],

                b=
                    cfg[
                        "bm25_b"
                    ],

                method=
                    "lucene",
            )
        )


        self.retriever.index(
            corpus_tokens,
            show_progress=False,
        )


        self.chunk_count = (
            len(
                self.chunk_texts
            )
        )


    def search_with_evidence(
        self,
        query,
    ):

        start = (
            time.perf_counter()
        )


        tokens = (
            self.tokenizer.tokenize(
                [query],
                update_vocab=False,
                return_as="tuple",
            )
        )


        indices, scores = (
            self.retriever.retrieve(
                tokens,

                k=min(
                    BM25_CHUNK_K,
                    self.chunk_count,
                ),
            )
        )


        parents = []
        seen = set()
        evidence = {}


        for chunk_index, score in zip(
            indices[0],
            scores[0],
        ):

            score = float(score)

            if score <= 0:
                continue


            idx = int(
                chunk_index
            )

            parent_id = (
                self.parent_ids[
                    idx
                ]
            )


            if parent_id in seen:
                continue


            seen.add(
                parent_id
            )

            parents.append(
                parent_id
            )


            evidence[
                parent_id
            ] = {
                "parent_id":
                    parent_id,

                "chunk_id":
                    self.chunk_ids[
                        idx
                    ],

                "text":
                    self.chunk_texts[
                        idx
                    ],

                "score":
                    score,

                "lane":
                    "bm25",

                "rank":
                    len(parents),
            }


            if (
                len(parents)
                >=
                SPARSE_PARENT_K
            ):
                break


        elapsed_ms = (
            (
                time.perf_counter()
                -
                start
            )
            *
            1000
        )


        return (
            parents,
            elapsed_ms,
            evidence,
        )


    def search(
        self,
        query,
    ):

        (
            parents,
            elapsed_ms,
            _,
        ) = (
            self.search_with_evidence(
                query
            )
        )

        return (
            parents,
            elapsed_ms,
        )


# =============================================================================
# WEIGHTED RRF — FROZEN
# =============================================================================

def weighted_rrf(
    dense,
    sparse,
    language,
):

    cfg = CFG[
        language
    ]

    scores = {}


    for rank0, parent in enumerate(
        dense[
            :FUSION_DEPTH
        ]
    ):

        contribution = (
            1.0
            /
            (
                cfg[
                    "rrf_k"
                ]
                +
                (
                    rank0 + 1
                )
                /
                cfg[
                    "dense_weight"
                ]
                -
                1
            )
        )


        scores[
            parent
        ] = (
            scores.get(
                parent,
                0.0,
            )
            +
            contribution
        )


    for rank0, parent in enumerate(
        sparse[
            :FUSION_DEPTH
        ]
    ):

        contribution = (
            1.0
            /
            (
                cfg[
                    "rrf_k"
                ]
                +
                (
                    rank0 + 1
                )
                /
                cfg[
                    "sparse_weight"
                ]
                -
                1
            )
        )


        scores[
            parent
        ] = (
            scores.get(
                parent,
                0.0,
            )
            +
            contribution
        )


    return [
        parent

        for parent, _
        in sorted(
            scores.items(),

            key=lambda item:
                (
                    -item[1],
                    item[0],
                ),
        )[
            :FINAL_PARENT_K
        ]
    ]


# =============================================================================
# EVIDENCE WINDOW
# =============================================================================

def _crop_window(
    text: str,
    query: str,
    max_chars: int,
    language: str,
) -> str:

    text = " ".join(
        str(text).split()
    )


    if len(text) <= max_chars:
        return text


    windows = []


    # Head window.
    windows.append(
        text[
            :max_chars
        ]
    )


    # Tail window.
    windows.append(
        text[
            -max_chars:
        ]
    )


    query_terms = word_splitter(
        query
    )

    folded = text.casefold()

    positions = []

    for term in query_terms:

        if len(term) < 2:
            continue

        position = folded.find(
            term.casefold()
        )

        if position >= 0:
            positions.append(
                position
            )


    for position in positions:

        start = max(
            0,
            position
            -
            max_chars // 3,
        )

        end = min(
            len(text),
            start
            +
            max_chars,
        )

        windows.append(
            text[
                start:end
            ]
        )


    cleaned_windows = []

    for window in windows:

        window = (
            window.strip()
        )

        if not window:
            continue

        cleaned_windows.append(
            window
        )


    if not cleaned_windows:

        return text[
            :max_chars
        ]


    return max(
        cleaned_windows,

        key=lambda candidate:
            evidence_pack_score(
                query,
                candidate,
                language,
            ),
    ).strip()


def evidence_window(
    text: str,
    query: str,
    max_chars: int,
    language: str,
) -> str:

    text = " ".join(
        str(text).split()
    )

    if not text:
        return ""


    support = (
        strongest_supporting_unit(
            query,
            text,
            language,
        )
    )


    if (
        support.strong
        and
        support.unit
    ):

        if len(
            support.unit
        ) <= max_chars:

            return support.unit.strip()


        return _crop_window(
            support.unit,
            query,
            max_chars,
            language,
        )


    units = split_sentences(
        text
    )


    if not units:

        return _crop_window(
            text,
            query,
            max_chars,
            language,
        )


    best = max(
        units,

        key=lambda unit:
            evidence_pack_score(
                query,
                unit,
                language,
            ),
    )


    if len(best) <= max_chars:
        return best.strip()


    return _crop_window(
        best,
        query,
        max_chars,
        language,
    )


# =============================================================================
# CONTEXT STORE
# =============================================================================

class ContextStore:

    def __init__(
        self,
    ):

        self.parent_chunks = {}
        self.chunk_lookup = {}


        for language, path in (
            CHUNKS.items()
        ):

            schema = (
                pq.read_schema(path)
                .names
            )


            text_col = None

            for candidate in [
                "text",
                "chunk_text",
                "content",
            ]:

                if candidate in schema:

                    text_col = candidate
                    break


            if text_col is None:

                raise RuntimeError(
                    f"No text column in {path}"
                )


            has_chunk_id = (
                "chunk_id"
                in schema
            )


            read_columns = [
                "parent_id",
                text_col,
            ]


            if has_chunk_id:

                read_columns.append(
                    "chunk_id"
                )


            df = pd.read_parquet(
                path,
                columns=
                    read_columns,
            )


            parent_ids = (
                df[
                    "parent_id"
                ]
                .astype(str)
                .tolist()
            )


            texts = (
                df[
                    text_col
                ]
                .fillna("")
                .astype(str)
                .tolist()
            )


            if has_chunk_id:

                chunk_ids = (
                    df[
                        "chunk_id"
                    ]
                    .astype(str)
                    .tolist()
                )

            else:

                chunk_ids = [
                    f"{language}_row_{idx}"
                    for idx in range(
                        len(df)
                    )
                ]


            parent_mapping = {}
            chunk_mapping = {}


            for (
                parent_id,
                chunk_id,
                text,
            ) in zip(
                parent_ids,
                chunk_ids,
                texts,
            ):

                text = text.strip()

                if not text:
                    continue


                item = {
                    "parent_id":
                        parent_id,

                    "chunk_id":
                        chunk_id,

                    "text":
                        text,

                    "lane":
                        "sibling",

                    "score":
                        None,
                }


                parent_mapping.setdefault(
                    parent_id,
                    [],
                ).append(
                    item
                )


                chunk_mapping[
                    chunk_id
                ] = text


            self.parent_chunks[
                language
            ] = (
                parent_mapping
            )


            self.chunk_lookup[
                language
            ] = (
                chunk_mapping
            )


    def _resolve_candidate(
        self,
        language: str,
        candidate: dict | None,
    ) -> dict | None:

        if not candidate:
            return None


        result = dict(
            candidate
        )


        text = result.get(
            "text"
        )


        chunk_id = result.get(
            "chunk_id"
        )


        if (
            not text
            and
            chunk_id
        ):

            text = (
                self.chunk_lookup[
                    language
                ]
                .get(
                    str(
                        chunk_id
                    )
                )
            )


        if not text:
            return None


        result[
            "text"
        ] = str(text)


        return result


    def _all_parent_candidates(
        self,
        language: str,
        parent: str,
        evidence_by_parent: dict,
    ) -> list[dict]:

        candidates = []


        preferred = (
            evidence_by_parent.get(
                parent
            )
        )


        if preferred:

            resolved = (
                self._resolve_candidate(
                    language,
                    preferred,
                )
            )

            if resolved:
                candidates.append(
                    resolved
                )


            alternate = preferred.get(
                "alternate"
            )

            if alternate:

                resolved = (
                    self._resolve_candidate(
                        language,
                        alternate,
                    )
                )

                if resolved:
                    candidates.append(
                        resolved
                    )


        # Important quality improvement:
        #
        # Once a parent is already inside the frozen fused Top-20,
        # inspect its sibling chunks and choose the child that most
        # directly supports the question.
        #
        # This is NOT new retrieval and does not alter fused ranking.
        for sibling in (
            self.parent_chunks[
                language
            ]
            .get(
                parent,
                [],
            )
        ):

            candidates.append(
                sibling
            )


        unique = []
        seen = set()


        for candidate in candidates:

            key = (
                candidate.get(
                    "chunk_id"
                ),
                candidate.get(
                    "text"
                ),
            )

            if key in seen:
                continue


            seen.add(key)

            unique.append(
                candidate
            )


        return unique


    def build(
        self,
        language,
        query,
        parents,
        char_budget,
        max_parents,
        per_chunk_chars,
        evidence_by_parent=None,
    ):

        start = (
            time.perf_counter()
        )


        evidence_by_parent = (
            evidence_by_parent
            or {}
        )


        original_rank = {
            str(parent):
                rank

            for rank, parent
            in enumerate(
                parents
            )
        }


        parent_best = {}


        for parent in parents:

            parent = str(
                parent
            )


            candidates = (
                self._all_parent_candidates(
                    language,
                    parent,
                    evidence_by_parent,
                )
            )


            if not candidates:
                continue


            best = max(
                candidates,

                key=lambda candidate:
                    evidence_pack_score(
                        query,
                        candidate[
                            "text"
                        ],
                        language,
                    ),
            )


            parent_best[
                parent
            ] = (
                best
            )


        # ---------------------------------------------------------
        # Context packing ordering only.
        #
        # Fused parent ranking itself remains untouched.
        # ---------------------------------------------------------

        ordered_parents = sorted(
            parent_best.keys(),

            key=lambda parent: (
                -evidence_pack_score(
                    query,
                    parent_best[
                        parent
                    ][
                        "text"
                    ],
                    language,
                ),

                original_rank[
                    parent
                ],
            ),
        )


        blocks = []
        used_evidence = []
        support_signals = []

        used_chars = 0


        for parent in ordered_parents:

            if len(
                blocks
            ) >= max_parents:
                break


            selected = (
                parent_best[
                    parent
                ]
            )


            snippet = (
                evidence_window(
                    selected[
                        "text"
                    ],
                    query,
                    per_chunk_chars,
                    language,
                )
            )


            if not snippet:
                continue


            separator_cost = (
                2
                if blocks
                else 0
            )


            candidate_cost = (
                separator_cost
                +
                len(snippet)
            )


            if (
                used_chars
                +
                candidate_cost
                >
                char_budget
            ):
                continue


            blocks.append(
                snippet
            )


            used_chars += (
                candidate_cost
            )


            used_evidence.append({
                "parent_id":
                    parent,

                "chunk_id":
                    selected.get(
                        "chunk_id"
                    ),

                "lane":
                    selected.get(
                        "lane",
                        "sibling",
                    ),

                "score":
                    selected.get(
                        "score"
                    ),

                "text":
                    snippet,
            })


            support_signals.append(
                strongest_supporting_unit(
                    query,
                    snippet,
                    language,
                )
            )


        # ---------------------------------------------------------
        # If one packed source directly answers the query, do NOT
        # give the 0.6B model a second distractor source.
        # ---------------------------------------------------------

        strong_indices = [
            idx

            for idx, signal
            in enumerate(
                support_signals
            )

            if signal.strong
        ]


        if strong_indices:

            best_idx = max(
                strong_indices,

                key=lambda idx: (
                    support_signals[
                        idx
                    ].score,

                    support_signals[
                        idx
                    ].coverage,
                ),
            )


            signal = (
                support_signals[
                    best_idx
                ]
            )


            source = dict(
                used_evidence[
                    best_idx
                ]
            )


            source[
                "text"
            ] = (
                signal.unit
            )


            context = (
                signal.unit
            )


            used_evidence = [
                source
            ]


            context_parent_count = 1


        else:

            context = "\n\n".join(
                blocks
            )

            context_parent_count = (
                len(blocks)
            )


        elapsed_ms = (
            (
                time.perf_counter()
                -
                start
            )
            *
            1000
        )


        return (
            context,
            elapsed_ms,
            context_parent_count,
            used_evidence,
        )


# =============================================================================
# FIRST TOKEN STREAMER
# =============================================================================

class FirstTokenStreamer(
    BaseStreamer
):

    def __init__(
        self,
        tokenizer,
    ):

        self.tokenizer = tokenizer

        self.ignore_prompt = True

        self.first_token_at = None
        self.completed_at = None

        self.generated_ids = []

        self.done = (
            threading.Event()
        )


    def put(
        self,
        value,
    ):

        ids = (
            value
            .detach()
            .cpu()
            .view(-1)
            .tolist()
        )


        if self.ignore_prompt:

            self.ignore_prompt = False
            return


        if self.first_token_at is None:

            self.first_token_at = (
                time.perf_counter()
            )


        self.generated_ids.extend(
            ids
        )


    def end(
        self,
    ):

        self.completed_at = (
            time.perf_counter()
        )

        self.done.set()


# =============================================================================
# FULL ENGINE
# =============================================================================

class FullRAG:

    def __init__(
        self,
    ):

        self.qdrant = (
            QdrantClient(
                url=
                    settings.qdrant_url,

                prefer_grpc=
                    settings.qdrant_prefer_grpc,

                timeout=
                    30,

                check_compatibility=
                    False,
            )
        )


        print(
            "Loading Qwen embedder..."
        )


        self.embedder = (
            SentenceTransformer(
                EMBED_MODEL,

                device=
                    "cuda",

                model_kwargs={
                    "torch_dtype":
                        torch.float16,

                    "attn_implementation":
                        "sdpa",
                },
            )
        )


        self.embedder.max_seq_length = (
            QUERY_MAX_LENGTH
        )

        self.embedder.eval()


        print(
            "Loading Qwen generator..."
        )


        self.gen_tokenizer = (
            AutoTokenizer
            .from_pretrained(
                GEN_MODEL
            )
        )


        self.gen_tokenizer.truncation_side = (
            "right"
        )


        self.generator = (
            AutoModelForCausalLM
            .from_pretrained(
                GEN_MODEL,

                torch_dtype=
                    torch.float16,

                device_map=
                    "cuda",
            )
        )


        self.generator.eval()


        self.bm25 = {
            "ta":
                BM25Engine(
                    "ta"
                ),

            "hi":
                BM25Engine(
                    "hi"
                ),
        }


        self.contexts = (
            ContextStore()
        )


        self.executor = (
            ThreadPoolExecutor(
                max_workers=1
            )
        )


        print(
            "GPU:",
            torch.cuda
            .get_device_name(
                0
            )
        )


        print(
            "GPU allocated:",
            round(
                torch.cuda
                .memory_allocated()
                /
                1024**3,
                2,
            ),
            "GB",
        )


    # =========================================================================
    # RETRIEVAL
    # =========================================================================

    def retrieve(
        self,
        query,
        language,
    ):

        overall = (
            time.perf_counter()
        )


        sparse_future = (
            self.executor.submit(
                self.bm25[
                    language
                ].search_with_evidence,
                query,
            )
        )


        torch.cuda.synchronize()

        embed_start = (
            time.perf_counter()
        )


        vector = (
            self.embedder.encode(
                [query],

                prompt_name=
                    "query",

                truncate_dim=
                    EMBED_DIM,

                normalize_embeddings=
                    True,

                convert_to_numpy=
                    True,

                show_progress_bar=
                    False,
            )[0]
        )


        torch.cuda.synchronize()


        embed_ms = (
            (
                time.perf_counter()
                -
                embed_start
            )
            *
            1000
        )


        dense_start = (
            time.perf_counter()
        )


        response = (
            self.qdrant.query_points(
                collection_name=
                    CFG[
                        language
                    ][
                        "collection"
                    ],

                query=
                    vector.tolist(),

                limit=
                    DENSE_CHILD_K,

                with_payload=[
                    "parent_id",
                    "chunk_id",
                ],

                with_vectors=
                    False,

                search_params=
                    models.SearchParams(
                        hnsw_ef=
                            HNSW_EF,

                        exact=
                            False,

                        indexed_only=
                            True,
                    ),
            )
        )


        dense = []
        seen = set()
        dense_evidence = {}


        for point in response.points:

            payload = (
                point.payload
                or {}
            )


            parent = str(
                payload.get(
                    "parent_id",
                    "",
                )
            )


            if (
                not parent
                or
                parent in seen
            ):
                continue


            seen.add(
                parent
            )


            dense.append(
                parent
            )


            dense_evidence[
                parent
            ] = {
                "parent_id":
                    parent,

                "chunk_id":
                    str(
                        payload.get(
                            "chunk_id",
                            "",
                        )
                    ),

                "score":
                    float(
                        point.score
                    ),

                "lane":
                    "dense",

                "rank":
                    len(dense),
            }


            if (
                len(dense)
                >=
                DENSE_PARENT_K
            ):
                break


        dense_ms = (
            (
                time.perf_counter()
                -
                dense_start
            )
            *
            1000
        )


        (
            sparse,
            bm25_ms,
            sparse_evidence,
        ) = (
            sparse_future.result()
        )


        fused = weighted_rrf(
            dense,
            sparse,
            language,
        )


        cfg = CFG[
            language
        ]


        dense_rank = {
            parent:
                rank

            for rank, parent
            in enumerate(
                dense,
                start=1,
            )
        }


        sparse_rank = {
            parent:
                rank

            for rank, parent
            in enumerate(
                sparse,
                start=1,
            )
        }


        def _rrf_contribution(
            rank,
            weight,
        ):

            if rank is None:
                return 0.0


            return (
                1.0
                /
                (
                    cfg[
                        "rrf_k"
                    ]
                    +
                    rank
                    /
                    weight
                    -
                    1
                )
            )


        evidence_by_parent = {}


        for parent in fused:

            dense_info = (
                dense_evidence.get(
                    parent
                )
            )


            sparse_info = (
                sparse_evidence.get(
                    parent
                )
            )


            dense_contrib = (
                _rrf_contribution(
                    dense_rank.get(
                        parent
                    ),

                    cfg[
                        "dense_weight"
                    ],
                )
            )


            sparse_contrib = (
                _rrf_contribution(
                    sparse_rank.get(
                        parent
                    ),

                    cfg[
                        "sparse_weight"
                    ],
                )
            )


            if (
                dense_info
                and
                dense_contrib
                >=
                sparse_contrib
            ):

                selected = dict(
                    dense_info
                )


                if sparse_info:

                    selected[
                        "alternate"
                    ] = (
                        sparse_info
                    )


                evidence_by_parent[
                    parent
                ] = (
                    selected
                )


            elif sparse_info:

                selected = dict(
                    sparse_info
                )


                if dense_info:

                    selected[
                        "alternate"
                    ] = (
                        dense_info
                    )


                evidence_by_parent[
                    parent
                ] = (
                    selected
                )


            elif dense_info:

                evidence_by_parent[
                    parent
                ] = dict(
                    dense_info
                )


        retrieval_ms = (
            (
                time.perf_counter()
                -
                overall
            )
            *
            1000
        )


        return {
            "parents":
                fused,

            "dense_parents":
                dense,

            "sparse_parents":
                sparse,

            "evidence_by_parent":
                evidence_by_parent,

            "embed_ms":
                embed_ms,

            "dense_ms":
                dense_ms,

            "bm25_ms":
                bm25_ms,

            "retrieval_ms":
                retrieval_ms,
        }


    # =========================================================================
    # GENERATION
    # =========================================================================

    def generate(
        self,
        query,
        context,
        max_new_tokens,
        language="ta",
    ):

        stage_start = (
            time.perf_counter()
        )


        support = (
            strongest_supporting_unit(
                query,
                context,
                language,
            )
        )


        strong_evidence = (
            support.strong
        )


        # ---------------------------------------------------------
        # The 0.6B model gets a much smaller job.
        # ---------------------------------------------------------

        if strong_evidence:

            system_content = (
                "Copy ONLY the shortest answer span from Evidence. "
                "Evidence has been verified to directly answer Question. "
                "Return 1-5 words only. "
                "No explanation. "
                "Do not repeat Question. "
                "Do not choose a distractor from a list."
            )


        else:

            system_content = (
                "Copy ONLY a directly supported answer from Evidence. "
                "Return 1-5 words. "
                "If Evidence is about a different subject/country, "
                "contradicts Question, or does not directly answer it, "
                "output exactly NOT_FOUND. "
                "Do not guess."
            )


        messages = [
            {
                "role":
                    "system",

                "content":
                    system_content,
            },

            {
                "role":
                    "user",

                "content":
                    (
                        f"Question:\n"
                        f"{query}\n\n"
                        f"Evidence:\n"
                        f"{context}\n\n"
                        "Answer:"
                    ),
            },
        ]


        prompt = (
            self.gen_tokenizer
            .apply_chat_template(
                messages,

                tokenize=False,

                add_generation_prompt=
                    True,

                enable_thinking=
                    False,
            )
        )


        inputs = (
            self.gen_tokenizer(
                prompt,

                return_tensors=
                    "pt",

                truncation=
                    True,

                max_length=
                    1024,
            )
            .to(
                "cuda"
            )
        )


        prompt_tokens = int(
            inputs[
                "input_ids"
            ].shape[1]
        )


        prep_ms = (
            (
                time.perf_counter()
                -
                stage_start
            )
            *
            1000
        )


        streamer = (
            FirstTokenStreamer(
                self.gen_tokenizer
            )
        )


        kwargs = {
            **inputs,

            "streamer":
                streamer,

            "max_new_tokens":
                max_new_tokens,

            "do_sample":
                False,

            "use_cache":
                True,

            "pad_token_id":
                self.gen_tokenizer
                .eos_token_id,
        }


        torch.cuda.synchronize()


        model_start = (
            time.perf_counter()
        )


        worker = threading.Thread(
            target=
                self.generator.generate,

            kwargs=
                kwargs,

            daemon=
                True,
        )


        worker.start()


        streamer.done.wait(
            timeout=60
        )


        worker.join()


        if (
            streamer.first_token_at
            is None
        ):

            raise RuntimeError(
                "No generated token received"
            )


        raw_answer = (
            self.gen_tokenizer.decode(
                streamer.generated_ids,

                skip_special_tokens=
                    True,
            )
        ).strip()


        lines = [
            line.strip()

            for line
            in raw_answer.splitlines()

            if line.strip()
        ]


        answer = (
            lines[0]
            if lines
            else ""
        )


        first_token_ms = (
            (
                streamer.first_token_at
                -
                model_start
            )
            *
            1000
        )


        complete_ms = (
            (
                streamer.completed_at
                -
                model_start
            )
            *
            1000
        )


        return {
            "answer":
                answer,

            "raw_answer":
                raw_answer,

            "strong_evidence":
                strong_evidence,

            "support_score":
                support.score,

            "support_coverage":
                support.coverage,

            "support_unit":
                support.unit,

            "prompt_tokens":
                prompt_tokens,

            "prompt_context_chars":
                len(
                    context
                ),

            "prompt_prep_ms":
                prep_ms,

            "model_first_token_ms":
                first_token_ms,

            "generation_stage_ttft_ms":
                prep_ms
                +
                first_token_ms,

            "generation_complete_ms":
                prep_ms
                +
                complete_ms,

            "first_token_at":
                streamer.first_token_at,

            "completed_at":
                streamer.completed_at,

            "generated_tokens":
                len(
                    streamer.generated_ids
                ),

            # TELEMETRY ONLY.
            #
            # guardrails.py no longer automatically rejects a grounded
            # answer just because this is True.
            "possibly_truncated": (
                len(
                    streamer.generated_ids
                )
                >=
                max_new_tokens
            ),
        }


    # =========================================================================
    # WARMUP
    # =========================================================================

    def warmup(
        self,
    ) -> None:

        warmup_queries = [
            (
                "இந்தியாவின் தலைநகரம் எது?",
                "ta",
            ),
            (
                "भारत की राजधानी क्या है?",
                "hi",
            ),
        ]


        for query, language in (
            warmup_queries
        ):

            try:

                retrieval = (
                    self.retrieve(
                        query,
                        language,
                    )
                )


                (
                    context,
                    _,
                    _,
                    _,
                ) = (
                    self.contexts.build(
                        language,
                        query,

                        retrieval[
                            "parents"
                        ],

                        CONTEXT_CHAR_BUDGET,
                        MAX_CONTEXT_PARENTS,
                        PER_CHUNK_CHARS,

                        evidence_by_parent=
                            retrieval.get(
                                "evidence_by_parent",
                                {},
                            ),
                    )
                )


                if context:

                    self.generate(
                        query,
                        context,
                        8,

                        language=
                            language,
                    )


            except Exception as exc:

                print(
                    f"Warmup warning "
                    f"for {language}: "
                    f"{exc}"
                )