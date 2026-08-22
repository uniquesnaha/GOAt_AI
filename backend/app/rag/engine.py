"""
GOAt AI RAG serving engine.

Retrieval remains frozen:
    Qwen3-Embedding-0.6B
    256d normalized embeddings
    Qdrant HNSW
    BM25
    language-specific weighted RRF
    Top-20 parent fusion

This version changes ONLY downstream evidence packing / tiny-model
generation behavior.

Quality objectives:
- preserve <200ms steady-state TTFT as far as possible,
- no reranker,
- no second LLM call,
- no change to retrieval ranking,
- reduce false NOT_FOUND from Qwen3-0.6B,
- prevent numbered-list distractor copying,
- keep unsupported answers blocked.
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
    score_sentence,
    split_sentences,
    strongest_supporting_sentence,
)


ROOT = Path(
    settings.data_root
)


# =============================================================================
# ACTIVE CORPUS PROFILE
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
        "ta":
            "hhgoa_fixed384_ta",

        "hi":
            "hhgoa_fixed384_hi",
    }


elif CORPUS_PROFILE == "350k":

    CHUNK_ROOT = (
        ROOT
        / "chunks_350k"
        / "fixed_384_96"
    )

    ACTIVE_COLLECTIONS = {
        "ta":
            "hhgoa_350k_fixed384_ta",

        "hi":
            "hhgoa_350k_fixed384_hi",
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
            ACTIVE_COLLECTIONS[
                "ta"
            ],

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
            ACTIVE_COLLECTIONS[
                "hi"
            ],

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
# CONTEXT / GENERATION CONFIG
# =============================================================================

# Important correction:
#
# 220 + "\n\n" + 220 = 442
#
# The previous 440 budget could silently reject a full second block.
#
# 450 allows two 220-character blocks while staying very compact.
CONTEXT_CHAR_BUDGET = 450
MAX_CONTEXT_PARENTS = 2
PER_CHUNK_CHARS = 220

MAX_NEW_TOKENS = 24


# =============================================================================
# TEXT HELPERS
# =============================================================================

def word_splitter(
    text,
):

    text = str(
        text
    ).casefold()

    tokens = []
    current = []

    for char in text:

        category = (
            unicodedata.category(
                char
            )
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

            current.append(
                char
            )

        elif current:

            tokens.append(
                "".join(
                    current
                )
            )

            current = []

    if current:

        tokens.append(
            "".join(
                current
            )
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
            pq.read_schema(
                path
            )
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
            "chunk_id"
            in schema
        )

        if has_chunk_id:

            read_columns.append(
                "chunk_id"
            )

        df = pd.read_parquet(
            path,
            columns=
                read_columns,
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
                for i
                in range(
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
            f"{language.upper()} "
            f"BM25 over "
            f"{len(self.chunk_texts):,} chunks..."
        )

        corpus_tokens = (
            self.tokenizer
            .tokenize(
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
            self.tokenizer
            .tokenize(
                [query],
                update_vocab=False,
                return_as=
                    "tuple",
            )
        )

        indices, scores = (
            self.retriever
            .retrieve(
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

            score = float(
                score
            )

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
                    len(
                        parents
                    ),
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

def _query_centered_crop(
    text: str,
    query: str,
    max_chars: int,
) -> str:

    text = " ".join(
        str(
            text
        ).split()
    )

    if len(text) <= max_chars:
        return text

    query_terms = [
        term
        for term
        in word_splitter(
            query
        )
        if len(term) >= 2
    ]

    if not query_terms:
        return (
            text[
                :max_chars
            ]
            .rsplit(
                " ",
                1,
            )[0]
            .strip()
        )

    folded = (
        text.casefold()
    )

    positions = []

    for term in query_terms:

        pos = (
            folded.find(
                term.casefold()
            )
        )

        if pos >= 0:
            positions.append(
                pos
            )

    if not positions:

        result = text[
            :max_chars
        ]

        if (
            len(result)
            <
            len(text)
            and
            " " in result
        ):
            result = (
                result
                .rsplit(
                    " ",
                    1,
                )[0]
            )

        return result.strip()

    center = int(
        sum(
            positions
        )
        /
        len(
            positions
        )
    )

    start = max(
        0,
        center
        -
        max_chars
        //
        3,
    )

    end = min(
        len(text),
        start
        +
        max_chars,
    )

    if (
        end
        -
        start
        <
        max_chars
    ):

        start = max(
            0,
            end
            -
            max_chars,
        )

    if start > 0:

        first_space = (
            text.find(
                " ",
                start,
            )
        )

        if (
            first_space >= 0
            and
            first_space < end
        ):
            start = (
                first_space + 1
            )

    if end < len(text):

        last_space = (
            text.rfind(
                " ",
                start,
                end,
            )
        )

        if last_space > start:
            end = last_space

    return (
        text[
            start:end
        ]
        .strip()
    )


def evidence_window(
    text: str,
    query: str,
    max_chars: int,
    language: str = "ta",
) -> str:
    """
    Produce one compact answer-focused evidence window.

    Important fixes:
    - NEVER bypass sentence scoring merely because a chunk is short.
    - numbered lists are split into logical items.
    - strong direct evidence returns ONLY the direct sentence.
    - weak/list evidence may include one useful adjacent unit.
    """

    text = str(
        text
    ).strip()

    if not text:
        return ""

    sentences = split_sentences(
        text
    )

    if not sentences:
        return _query_centered_crop(
            text,
            query,
            max_chars,
        )

    # ---------------------------------------------------------
    # Strong direct support:
    # give the tiny model only the direct sentence.
    # ---------------------------------------------------------

    support = (
        strongest_supporting_sentence(
            query,
            text,
            language,
        )
    )

    if (
        support.strong
        and
        support.sentence
    ):

        if (
            len(
                support.sentence
            )
            <=
            max_chars
        ):
            return (
                support.sentence
                .strip()
            )

        return _query_centered_crop(
            support.sentence,
            query,
            max_chars,
        )

    # ---------------------------------------------------------
    # Otherwise choose the best sentence/list item.
    # ---------------------------------------------------------

    scored = [
        (
            score_sentence(
                query,
                sentence,
                language,
            ),
            idx,
            sentence,
        )
        for idx, sentence
        in enumerate(
            sentences
        )
    ]

    scored.sort(
        key=lambda item:
            (
                -item[0],
                item[1],
            )
    )

    (
        best_score,
        best_idx,
        best_sentence,
    ) = scored[0]

    if best_score <= 0:

        # No useful lexical alignment.
        # Keep the passage compact rather than flooding the model.
        return _query_centered_crop(
            text,
            query,
            max_chars,
        )

    if (
        len(best_sentence)
        >
        max_chars
    ):

        return _query_centered_crop(
            best_sentence,
            query,
            max_chars,
        )

    result_parts = [
        best_sentence
    ]

    remaining = (
        max_chars
        -
        len(
            best_sentence
        )
    )

    # ---------------------------------------------------------
    # Adjacent evidence is useful for cases such as:
    #
    # 3. Earth ...
    # 4. It has one natural moon.
    #
    # We prefer NEXT before PREVIOUS.
    # ---------------------------------------------------------

    neighbor_indices = []

    if (
        best_idx + 1
        <
        len(sentences)
    ):
        neighbor_indices.append(
            best_idx + 1
        )

    if best_idx > 0:
        neighbor_indices.append(
            best_idx - 1
        )

    for neighbor_idx in neighbor_indices:

        neighbor = (
            sentences[
                neighbor_idx
            ]
        )

        neighbor_cost = (
            1
            +
            len(
                neighbor
            )
        )

        if neighbor_cost > remaining:
            continue

        neighbor_score = (
            score_sentence(
                query,
                neighbor,
                language,
            )
        )

        # Add an adjacent unit only if:
        # - it has some relevance, OR
        # - we're dealing with list-style evidence where relation
        #   can be split across adjacent items.
        if (
            neighbor_score > 0
            or
            neighbor.lstrip()[
                :2
            ].rstrip(".")
            .isdigit()
        ):

            if (
                neighbor_idx
                <
                best_idx
            ):

                result_parts.insert(
                    0,
                    neighbor,
                )

            else:

                result_parts.append(
                    neighbor
                )

            break

    result = " ".join(
        result_parts
    )

    if len(result) > max_chars:

        result = _query_centered_crop(
            result,
            query,
            max_chars,
        )

    return result.strip()


# =============================================================================
# CONTEXT STORE
# =============================================================================

class ContextStore:

    def __init__(
        self,
    ):

        self.data = {}
        self.chunk_lookup = {}

        for language, path in (
            CHUNKS.items()
        ):

            schema = (
                pq.read_schema(
                    path
                )
                .names
            )

            text_col = None

            for candidate in [
                "text",
                "chunk_text",
                "content",
            ]:

                if candidate in schema:

                    text_col = (
                        candidate
                    )

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

            parents = (
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
                    None
                    for _
                    in range(
                        len(df)
                    )
                ]

            mapping = {}
            chunk_mapping = {}

            # Faster than iterrows() for ~500k chunks.
            for (
                parent,
                text,
                chunk_id,
            ) in zip(
                parents,
                texts,
                chunk_ids,
            ):

                text = (
                    text.strip()
                )

                if not text:
                    continue

                mapping.setdefault(
                    parent,
                    [],
                ).append(
                    text
                )

                if chunk_id:

                    chunk_mapping[
                        chunk_id
                    ] = text

            self.data[
                language
            ] = mapping

            self.chunk_lookup[
                language
            ] = (
                chunk_mapping
            )


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

        query_tokens = set(
            word_splitter(
                query
            )
        )

        original_rank = {
            str(
                parent
            ):
                rank

            for rank, parent
            in enumerate(
                parents
            )
        }


        def resolve_candidate_text(
            candidate,
        ):

            if not candidate:
                return None

            text = (
                candidate.get(
                    "text"
                )
            )

            if text:
                return str(
                    text
                )

            chunk_id = (
                candidate.get(
                    "chunk_id"
                )
            )

            if chunk_id:

                return (
                    self.chunk_lookup[
                        language
                    ]
                    .get(
                        str(
                            chunk_id
                        )
                    )
                )

            return None


        def evidence_texts(
            parent,
        ):

            evidence = (
                evidence_by_parent
                .get(
                    str(
                        parent
                    )
                )
            )

            if not evidence:
                return []

            texts = []

            for candidate in (
                evidence,
                evidence.get(
                    "alternate"
                ),
            ):

                text = (
                    resolve_candidate_text(
                        candidate
                    )
                )

                if text:
                    texts.append(
                        text
                    )

            return texts


        # ==============================================================
        # CONTEXT-PACK RERANK ONLY.
        #
        # The frozen fused parent ranking is NOT changed.
        #
        # We simply choose which already-retrieved parents deserve the
        # tiny two-slot generation context.
        # ==============================================================

        parents = sorted(
            (
                str(
                    parent
                )
                for parent
                in parents
            ),

            key=lambda parent: (
                -max(
                    (
                        evidence_pack_score(
                            query,
                            text,
                            language,
                        )

                        for text
                        in evidence_texts(
                            parent
                        )
                    ),

                    default=
                        0.0,
                ),

                original_rank[
                    parent
                ],
            ),
        )


        blocks = []
        used_chars = 0
        used_evidence = []
        support_signals = []


        for parent in parents:

            if (
                len(blocks)
                >=
                max_parents
            ):
                break


            selected_text = None
            selected_chunk_id = None
            selected_lane = "fallback"
            selected_score = None


            evidence = (
                evidence_by_parent
                .get(
                    str(
                        parent
                    )
                )
            )


            # ==========================================================
            # Preferred retrieved child
            # ==========================================================

            if evidence:

                selected_chunk_id = (
                    evidence.get(
                        "chunk_id"
                    )
                )

                selected_lane = (
                    evidence.get(
                        "lane",
                        "unknown",
                    )
                )

                selected_score = (
                    evidence.get(
                        "score"
                    )
                )

                selected_text = (
                    resolve_candidate_text(
                        evidence
                    )
                )


            # ==========================================================
            # Compare alternate dense/BM25 child.
            #
            # Use answer-focused pack score instead of raw overlap.
            # ==========================================================

            if evidence:

                alternate = (
                    evidence.get(
                        "alternate"
                    )
                )

                if alternate:

                    alt_text = (
                        resolve_candidate_text(
                            alternate
                        )
                    )

                    if alt_text:

                        selected_pack_score = (
                            evidence_pack_score(
                                query,
                                selected_text,
                                language,
                            )
                            if selected_text
                            else 0.0
                        )

                        alternate_pack_score = (
                            evidence_pack_score(
                                query,
                                alt_text,
                                language,
                            )
                        )

                        if (
                            alternate_pack_score
                            >
                            selected_pack_score
                        ):

                            selected_text = (
                                alt_text
                            )

                            selected_chunk_id = (
                                alternate.get(
                                    "chunk_id"
                                )
                            )

                            selected_lane = (
                                alternate.get(
                                    "lane",
                                    "alternate",
                                )
                            )

                            selected_score = (
                                alternate.get(
                                    "score"
                                )
                            )


            # ==========================================================
            # Parent-level compatibility fallback
            # ==========================================================

            if not selected_text:

                candidates = (
                    self.data[
                        language
                    ]
                    .get(
                        str(
                            parent
                        ),
                        [],
                    )
                )

                if not candidates:
                    continue

                selected_text = max(
                    candidates,

                    key=lambda candidate_text:
                        (
                            evidence_pack_score(
                                query,
                                candidate_text,
                                language,
                            ),

                            len(
                                query_tokens
                                &
                                set(
                                    word_splitter(
                                        candidate_text
                                    )
                                )
                            ),
                        ),
                )

                selected_lane = (
                    "fallback"
                )


            snippet = (
                evidence_window(
                    selected_text,
                    query,
                    per_chunk_chars,
                    language=
                        language,
                )
            )

            if not snippet:
                continue


            # ==========================================================
            # Correct character-budget accounting.
            #
            # Previous version effectively charged "+2" after inserting
            # the first block but did not include that separator in the
            # actual pre-check.
            # ==============================================================

            separator_cost = (
                2
                if blocks
                else 0
            )

            candidate_cost = (
                separator_cost
                +
                len(
                    snippet
                )
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


            source = {
                "parent_id":
                    str(
                        parent
                    ),

                "chunk_id":
                    selected_chunk_id,

                "lane":
                    selected_lane,

                "score":
                    selected_score,

                "text":
                    snippet,
            }

            used_evidence.append(
                source
            )

            support_signals.append(
                strongest_supporting_sentence(
                    query,
                    snippet,
                    language,
                )
            )


        # ==============================================================
        # STRONG SUPPORT COLLAPSE
        #
        # This is the most important tiny-model optimization.
        #
        # If ONE sentence strongly answers the query:
        #
        #     Question
        #       ↓
        #     exactly one strong sentence
        #       ↓
        #     0.6B extractor
        #
        # rather than feeding an unrelated second source.
        #
        # No LLM call. No retrieval changes.
        # ==============================================================

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

                    support_signals[
                        idx
                    ].matched_terms,

                    -idx,
                ),
            )

            signal = (
                support_signals[
                    best_idx
                ]
            )

            focused_source = dict(
                used_evidence[
                    best_idx
                ]
            )

            focused_source[
                "text"
            ] = (
                signal.sentence
            )

            context = (
                signal.sentence
            )

            used_evidence = [
                focused_source
            ]

            context_parent_count = 1

        else:

            context = (
                "\n\n".join(
                    blocks
                )
            )

            context_parent_count = (
                len(
                    blocks
                )
            )


        ms = (
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
            ms,
            context_parent_count,
            used_evidence,
        )


# =============================================================================
# FIRST-TOKEN STREAMER
# =============================================================================

class FirstTokenStreamer(
    BaseStreamer
):

    def __init__(
        self,
        tokenizer,
    ):

        self.tokenizer = (
            tokenizer
        )

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

        # generate() sends prompt through streamer first.
        if self.ignore_prompt:

            self.ignore_prompt = (
                False
            )

            return

        if (
            self.first_token_at
            is None
        ):

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

        # ---------------------------------------------------------
        # QDRANT
        # ---------------------------------------------------------

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


        # ---------------------------------------------------------
        # EMBEDDER
        # ---------------------------------------------------------

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


        # ---------------------------------------------------------
        # GENERATOR
        # ---------------------------------------------------------

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

                dtype=
                    torch.float16,

                device_map=
                    "cuda",
            )
        )

        self.generator.eval()


        # ---------------------------------------------------------
        # PRE-COMPUTE SENTINEL TOKENIZATION
        # ---------------------------------------------------------

        self.not_found_token_ids = (
            self.gen_tokenizer.encode(
                "NOT_FOUND",
                add_special_tokens=False,
            )
        )


        # ---------------------------------------------------------
        # BM25
        # ---------------------------------------------------------

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


        # ---------------------------------------------------------
        # CONTEXT
        # ---------------------------------------------------------

        self.contexts = (
            ContextStore()
        )


        # ---------------------------------------------------------
        # BM25 THREAD
        # ---------------------------------------------------------

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
    # RETRIEVE
    # =========================================================================

    def retrieve(
        self,
        query,
        language,
    ):

        overall = (
            time.perf_counter()
        )


        # ---------------------------------------------------------
        # Start sparse retrieval concurrently.
        # ---------------------------------------------------------

        sparse_future = (
            self.executor.submit(
                self.bm25[
                    language
                ]
                .search_with_evidence,

                query,
            )
        )


        # ---------------------------------------------------------
        # QUERY EMBEDDING
        # ---------------------------------------------------------

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


        # ---------------------------------------------------------
        # DENSE
        # ---------------------------------------------------------

        dense_start = (
            time.perf_counter()
        )

        response = (
            self.qdrant
            .query_points(
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


        for point in (
            response.points
        ):

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
                    len(
                        dense
                    ),
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
            sparse_future
            .result()
        )


        # ---------------------------------------------------------
        # FROZEN FUSION
        # ---------------------------------------------------------

        fused = weighted_rrf(
            dense,
            sparse,
            language,
        )


        # ---------------------------------------------------------
        # EVIDENCE CHILD PRESERVATION
        #
        # Parent ranking is unchanged.
        # ---------------------------------------------------------

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
    # GENERATE
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


        # ---------------------------------------------------------
        # DETERMINE WHETHER THE EVIDENCE IS ALREADY STRONG.
        #
        # ContextStore usually collapses strong evidence to one sentence.
        # We recompute here because it is extremely cheap and keeps the
        # generator self-contained.
        # ---------------------------------------------------------

        support = (
            strongest_supporting_sentence(
                query,
                context,
                language,
            )
        )

        strong_evidence = (
            support.strong
        )


        # ---------------------------------------------------------
        # TWO PROMPT MODES
        #
        # CRITICAL:
        #
        # Strong evidence:
        #     DO NOT expose NOT_FOUND as the easy escape route.
        #
        # Weak evidence:
        #     allow strict abstention.
        #
        # No second LLM call.
        # ---------------------------------------------------------

        if strong_evidence:

            system_content = (
                "You are a factual span extractor. "
                "The Evidence contains the answer to the Question. "
                "Return ONLY the shortest answer words or value that are "
                "directly supported by the Evidence. "
                "Keep the original language/script. "
                "Do not explain. "
                "Do not repeat the question. "
                "Do not automatically choose the first item from a list."
            )

        else:

            system_content = (
                "You are a grounded factual extractor. "
                "Use ONLY the Evidence. "
                "If one evidence sentence directly answers the Question, "
                "return ONLY the shortest supported answer words or value. "
                "Otherwise output exactly NOT_FOUND. "
                "Do not guess. "
                "Do not use outside knowledge. "
                "Do not automatically choose the first item from a list."
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
                        f"Answer:"
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
            ]
            .shape[
                1
            ]
        )


        if prompt_tokens >= 950:

            print(
                "⚠️ Prompt near truncation: "
                f"{prompt_tokens} tokens"
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


        worker = (
            threading.Thread(
                target=
                    self.generator
                    .generate,

                kwargs=
                    kwargs,

                daemon=
                    True,
            )
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
            self.gen_tokenizer
            .decode(
                streamer.generated_ids,

                skip_special_tokens=
                    True,
            )
        )


        lines = [
            line.strip()

            for line
            in raw_answer
            .strip()
            .splitlines()

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
                answer.strip(),

            "raw_answer":
                raw_answer.strip(),

            "strong_evidence":
                strong_evidence,

            "support_score":
                support.score,

            "support_coverage":
                support.coverage,

            "support_sentence":
                support.sentence,

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

                    # IMPORTANT FIX:
                    # Hindi warmup previously used default language="ta".
                    self.generate(
                        query,
                        context,
                        16,
                        language=
                            language,
                    )

            except Exception as exc:

                print(
                    f"Warmup warning "
                    f"for {language}: "
                    f"{exc}"
                )