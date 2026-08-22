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
    AutoModelForSeq2SeqLM,
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
    settings.gen_model
)

GEN_BACKEND = (
    settings.gen_backend
    .strip()
    .lower()
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

# Keep the prompt small for TTFT, but preserve enough contiguous evidence
# for Tamil/Hindi answers. Two 240-char windows + separator = <= 482 chars.
CONTEXT_CHAR_BUDGET = 490
MAX_CONTEXT_PARENTS = 2
PER_CHUNK_CHARS = 240

# 12 tokens is too tight for some Tamil/Hindi answers even when the visible
# answer is only a few words. This does not materially change TTFT because
# TTFT is dominated by prompt prefill and the first decode step.
MAX_NEW_TOKENS = 24

# Preserve one adjacent sentence on either side of the best lexical sentence.
# This avoids throwing away the answer when a heuristic picks a nearby sentence.
EVIDENCE_NEIGHBOR_RADIUS = 1


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
    """Return a compact *contiguous* evidence window.

    Important: do not collapse a retrieved chunk to a single heuristic
    "strong" sentence. A tiny model is much more reliable when it sees the
    answer-bearing sentence together with its immediate local context.
    """

    text = " ".join(
        str(text).split()
    )

    if not text:
        return ""

    if len(text) <= max_chars:
        return text

    units = [
        unit.strip()
        for unit in split_sentences(text)
        if str(unit).strip()
    ]

    if not units:
        return _crop_window(
            text,
            query,
            max_chars,
            language,
        )

    scores = [
        evidence_pack_score(
            query,
            unit,
            language,
        )
        for unit in units
    ]

    best_idx = max(
        range(len(units)),
        key=lambda idx: scores[idx],
    )

    left = max(
        0,
        best_idx - EVIDENCE_NEIGHBOR_RADIUS,
    )
    right = min(
        len(units),
        best_idx + EVIDENCE_NEIGHBOR_RADIUS + 1,
    )

    window = " ".join(
        units[left:right]
    ).strip()

    if len(window) <= max_chars:
        return window

    # If the local sentence window is still too large, crop within that local
    # region rather than jumping to an unrelated head/tail sentence elsewhere.
    return _crop_window(
        window,
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

        start = time.perf_counter()
        evidence_by_parent = evidence_by_parent or {}

        blocks = []
        used_evidence = []
        used_chars = 0

        # IMPORTANT: preserve weighted-RRF parent order.
        # We may choose the best child *inside* an already-retrieved parent,
        # but we do not reorder the fused parents with a second heuristic ranker.
        for parent in parents:

            if len(blocks) >= max_parents:
                break

            parent = str(parent)

            candidates = self._all_parent_candidates(
                language,
                parent,
                evidence_by_parent,
            )

            if not candidates:
                continue

            # Child selection is local to this parent only. This helps recover
            # the answer-bearing sibling without changing frozen retrieval rank.
            selected = max(
                candidates,
                key=lambda candidate: evidence_pack_score(
                    query,
                    candidate["text"],
                    language,
                ),
            )

            snippet = evidence_window(
                selected["text"],
                query,
                per_chunk_chars,
                language,
            )

            if not snippet:
                continue

            separator_cost = 2 if blocks else 0
            candidate_cost = separator_cost + len(snippet)

            if used_chars + candidate_cost > char_budget:
                # Do not silently skip a high-ranked parent just because the
                # selected snippet is slightly too large. Fit it to remaining
                # budget when enough room is available.
                remaining = char_budget - used_chars - separator_cost

                if remaining < 80:
                    break

                snippet = evidence_window(
                    selected["text"],
                    query,
                    remaining,
                    language,
                )

                if not snippet:
                    continue

                candidate_cost = separator_cost + len(snippet)

            blocks.append(snippet)
            used_chars += candidate_cost

            used_evidence.append({
                "parent_id": parent,
                "chunk_id": selected.get("chunk_id"),
                "lane": selected.get("lane", "sibling"),
                "score": selected.get("score"),
                "text": snippet,
            })

        context = "\n\n".join(blocks)

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        return (
            context,
            elapsed_ms,
            len(blocks),
            used_evidence,
        )


# =============================================================================
# GROUNDED ANSWER VALIDATION
# =============================================================================

def _normalize_grounding_text(text: str) -> str:
    """Unicode/case/whitespace normalization for extractive validation."""

    text = unicodedata.normalize(
        "NFKC",
        str(text),
    ).casefold()

    return " ".join(
        text.split()
    )


def _answer_is_grounded(
    answer: str,
    context: str,
) -> bool:
    """True when the generated short answer is actually present in evidence."""

    answer_norm = _normalize_grounding_text(answer)
    context_norm = _normalize_grounding_text(context)

    if not answer_norm:
        return False

    if answer_norm == "not_found":
        return True

    return answer_norm in context_norm


def _clean_short_answer(answer: str) -> str:
    """Remove common tiny-model wrappers without rewriting the answer."""

    answer = str(answer).strip()

    if not answer:
        return ""

    first_line = next(
        (
            line.strip()
            for line in answer.splitlines()
            if line.strip()
        ),
        "",
    )

    for prefix in (
        "Answer:",
        "answer:",
        "उत्तर:",
        "பதில்:",
    ):
        if first_line.startswith(prefix):
            first_line = first_line[len(prefix):].strip()
            break

    # Avoid accidental explanations after a semicolon/pipe while preserving
    # punctuation that can legitimately occur inside names or quantities.
    for separator in (" | ", "; "):
        if separator in first_line:
            first_line = first_line.split(separator, 1)[0].strip()

    return first_line


def _build_generation_prompt(
    query: str,
    context: str,
) -> str:
    return (
        "Answer the question using only the evidence.\n"
        "Return only the shortest answer.\n"
        "Do not explain.\n"
        "If the answer is not present, output NOT_FOUND.\n\n"
        f"Evidence: {context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
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
        is_seq2seq: bool = False,
    ):

        self.tokenizer = tokenizer
        self.is_seq2seq = is_seq2seq

        self.ignore_initial_token = True

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


        if self.ignore_initial_token:

            self.ignore_initial_token = False
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
            f"Loading {GEN_BACKEND} generator ({GEN_MODEL})..."
        )


        self.gen_tokenizer = (
            AutoTokenizer
            .from_pretrained(
                GEN_MODEL
            )
        )


        # If an unexpected prompt ever exceeds the safety cap, preserve the
        # tail containing the question and answer cue instead of truncating it.
        self.gen_tokenizer.truncation_side = (
            "left"
        )


        if GEN_BACKEND == "seq2seq":

            self.generator = (
                AutoModelForSeq2SeqLM
                .from_pretrained(
                    GEN_MODEL,

                    torch_dtype=
                        torch.float16,

                    low_cpu_mem_usage=
                        True,
                )
                .to("cuda")
            )

        elif GEN_BACKEND == "causal":

            self.generator = (
                AutoModelForCausalLM
                .from_pretrained(
                    GEN_MODEL,

                    torch_dtype=
                        torch.float16,

                    device_map=
                        "cuda",

                    attn_implementation=
                        "sdpa",
                )
            )

        else:

            raise RuntimeError(
                f"Unsupported generator backend: {GEN_BACKEND}"
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
        max_new_tokens=None,
        language="ta",
    ):

        stage_start = time.perf_counter()

        if max_new_tokens is None:
            max_new_tokens = settings.gen_max_new_tokens

        if not str(context).strip():
            return {
                "answer": "NOT_FOUND",
                "raw_answer": "NOT_FOUND",
                "grounded": True,
                "strong_evidence": False,
                "support_score": 0.0,
                "support_coverage": 0.0,
                "support_unit": "",
                "prompt_tokens": 0,
                "prompt_context_chars": 0,
                "prompt_prep_ms": 0.0,
                "model_first_token_ms": 0.0,
                "generation_stage_ttft_ms": 0.0,
                "generation_complete_ms": 0.0,
                "first_token_at": None,
                "completed_at": None,
                "generated_tokens": 0,
                "possibly_truncated": False,
            }

        # Keep this signal for telemetry only.
        support = strongest_supporting_unit(
            query,
            context,
            language,
        )

        if GEN_BACKEND == "seq2seq":

            prompt = _build_generation_prompt(
                query,
                context,
            )

            inputs = self.gen_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=settings.gen_max_input_tokens,
            ).to("cuda")

        else:

            # Causal LM (e.g. Qwen fallback)
            system_content = (
                "Extract the answer from Evidence only. "
                "Copy the shortest exact span that answers Question. "
                "Output that span only. "
                "If absent, output exactly NOT_FOUND. "
                "Do not use outside knowledge."
            )

            messages = [
                {
                    "role": "system",
                    "content": system_content,
                },
                {
                    "role": "user",
                    "content": (
                        f"Evidence:\n{context}\n\n"
                        f"Question:\n{query}\n\n"
                        "Answer:"
                    ),
                },
            ]

            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }

            if "qwen3" in GEN_MODEL.casefold():
                template_kwargs["enable_thinking"] = False

            prompt = self.gen_tokenizer.apply_chat_template(
                messages,
                **template_kwargs,
            )

            inputs = self.gen_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to("cuda")

        prompt_tokens = int(
            inputs["input_ids"].shape[1]
        )

        prep_ms = (
            time.perf_counter() - stage_start
        ) * 1000

        streamer = FirstTokenStreamer(
            self.gen_tokenizer,
            is_seq2seq=(GEN_BACKEND == "seq2seq"),
        )

        kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "use_cache": True,
        }

        if self.gen_tokenizer.eos_token_id is not None:
            kwargs["pad_token_id"] = self.gen_tokenizer.eos_token_id

        if GEN_BACKEND == "seq2seq":
            kwargs["num_beams"] = 1

        torch.cuda.synchronize()
        model_start = time.perf_counter()

        self.generator.generate(
            **kwargs
        )

        if streamer.first_token_at is None:
            streamer.first_token_at = time.perf_counter()

        raw_answer = self.gen_tokenizer.decode(
            streamer.generated_ids,
            skip_special_tokens=True,
        ).strip()

        answer = _clean_short_answer(
            raw_answer
        )

        grounded = _answer_is_grounded(
            answer,
            context,
        )

        # Fail closed on hallucinations.
        if not grounded:
            answer = "NOT_FOUND"

        first_token_ms = (
            streamer.first_token_at - model_start
        ) * 1000

        complete_ms = (
            (streamer.completed_at or time.perf_counter()) - model_start
        ) * 1000

        generated_tokens = len(
            streamer.generated_ids
        )

        return {
            "answer": answer,
            "raw_answer": raw_answer,
            "grounded": grounded,
            "strong_evidence": bool(support.strong),
            "support_score": support.score,
            "support_coverage": support.coverage,
            "support_unit": support.unit,
            "prompt_tokens": prompt_tokens,
            "prompt_context_chars": len(context),
            "prompt_prep_ms": prep_ms,
            "model_first_token_ms": first_token_ms,
            "generation_stage_ttft_ms": prep_ms + first_token_ms,
            "generation_complete_ms": prep_ms + complete_ms,
            "first_token_at": streamer.first_token_at,
            "completed_at": streamer.completed_at,
            "generated_tokens": generated_tokens,
            "possibly_truncated": (
                generated_tokens >= max_new_tokens
            ),
        }


    # =========================================================================
    # WARMUP
    # =========================================================================

    def warmup(
        self,
    ) -> None:

        # 1. Direct generator warmups (compiles CUDA kernels)
        generator_warmup_samples = [
            (
                "இந்தியாவின் தலைநகரம் எது?",
                "இந்தியாவின் தலைநகரம் புதுதில்லி ஆகும்.",
                "ta",
            ),
            (
                "தாவரங்கள் ஒளிச்சேர்க்கைக்கு பயன்படுத்தும் வாயு எது?",
                "தாவரங்கள் ஒளிச்சேர்க்கைக்கு கார்பன் டை ஆக்சைடு வாயுவை பயன்படுத்துகின்றன.",
                "ta",
            ),
            (
                "भारत की राजधानी क्या है?",
                "भारत की राजधानी नई दिल्ली है।",
                "hi",
            ),
            (
                "सौरमंडल का सबसे बड़ा ग्रह कौन सा है?",
                "बृहस्पति सौरमंडल का सबसे बड़ा ग्रह है।",
                "hi",
            ),
        ]

        for query, ctx, lang in generator_warmup_samples:
            try:
                self.generate(
                    query,
                    ctx,
                    settings.gen_max_new_tokens,
                    language=lang,
                )
            except Exception as exc:
                print(f"Generator warmup warning ({lang}): {exc}")

        # 2. End-to-end RAG pipeline warmups (embedder + Qdrant + BM25 + Context + Generator)
        full_rag_queries = [
            (
                "இந்தியாவின் தலைநகரம் எது?",
                "ta",
            ),
            (
                "भारत की राजधानी क्या है?",
                "hi",
            ),
        ]

        for query, language in full_rag_queries:
            try:
                retrieval = self.retrieve(
                    query,
                    language,
                )

                (
                    context,
                    _,
                    _,
                    _,
                ) = self.contexts.build(
                    language,
                    query,
                    retrieval["parents"],
                    CONTEXT_CHAR_BUDGET,
                    MAX_CONTEXT_PARENTS,
                    PER_CHUNK_CHARS,
                    evidence_by_parent=retrieval.get(
                        "evidence_by_parent",
                        {},
                    ),
                )

                if context:
                    self.generate(
                        query,
                        context,
                        settings.gen_max_new_tokens,
                        language=language,
                    )

            except Exception as exc:
                print(f"Full RAG warmup warning for {language}: {exc}")