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

import math
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
# RETRIEVAL V3
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

BM25_CHUNK_K = 350
SPARSE_PARENT_K = 50

# Candidate generation.
FUSION_DEPTH = 50
FUSION_CANDIDATE_K = 30

# Lightweight deterministic rerank.
RERANK_PARENT_K = 30

# Returned retrieval parents.
FINAL_PARENT_K = 20


# Keep HNSW unchanged initially.
HNSW_EF = 64

# Standard-ish RRF smoothing.
RRF_K_V3 = 20.0

# Query-view weights.
DENSE_FULL_WEIGHT = 1.00
DENSE_ANCHOR_WEIGHT = 0.70
SPARSE_FULL_WEIGHT = 1.00
SPARSE_ANCHOR_WEIGHT = 1.15

# Deterministic parent-rerank weights.
RR_WEIGHT = 0.28
ANCHOR_WEIGHT = 0.27
QUERY_COVERAGE_WEIGHT = 0.20
EVIDENCE_WEIGHT = 0.15
LANE_AGREEMENT_WEIGHT = 0.05
PHRASE_WEIGHT = 0.05
ANCHOR_MISS_PENALTY = 0.18

MAX_ANCHOR_TERMS = 3



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


QUESTION_TERMS = {
    "ta": {
        "எது",
        "என்ன",
        "யார்",
        "எங்கு",
        "எங்கே",
        "எந்த",
        "எத்தனை",
        "எவ்வளவு",
        "எப்போது",
        "எப்படி",
        "ஏன்",
        "ஒரு",
        "என்று",
    },

    "hi": {
        "क्या",
        "कौन",
        "कहाँ",
        "कहां",
        "किस",
        "कितने",
        "कितना",
        "कितनी",
        "कब",
        "कैसे",
        "क्यों",
        "एक",
        "है",
        "हैं",
    },
}


def _normalize_retrieval_text(
    text: str,
) -> str:
    text = unicodedata.normalize(
        "NFKC",
        str(text),
    )
    return " ".join(
        text.casefold().split()
    )


def _content_query_terms(
    query: str,
    language: str,
) -> list[str]:
    stop = QUESTION_TERMS.get(
        language,
        set(),
    )
    tokens = word_splitter(
        _normalize_retrieval_text(
            query
        )
    )
    result = []
    seen = set()

    for token in tokens:
        if len(token) < 2:
            continue
        if token in stop:
            continue
        if token in seen:
            continue

        seen.add(token)
        result.append(token)

    return result


# =============================================================================
# BM25 ENGINE (V3 MULTI-VIEW)
# =============================================================================

class BM25Engine:

    def __init__(
        self,
        language,
    ):
        self.language = language

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
                splitter=word_splitter,
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
                return_as="tuple",
            )
        )

        self.retriever = (
            bm25s.BM25(
                k1=cfg["bm25_k1"],
                b=cfg["bm25_b"],
                method="lucene",
            )
        )

        self.retriever.index(
            corpus_tokens,
            show_progress=False,
        )

        self.chunk_count = len(
            self.chunk_texts
        )

    def idf_proxy(
        self,
        token: str,
    ) -> float:
        token = str(token).casefold()
        vocab = getattr(
            self.retriever,
            "vocab_dict",
            {},
        )
        token_id = vocab.get(token)
        if token_id is None:
            return 0.0

        scores = getattr(
            self.retriever,
            "scores",
            None,
        )
        if not scores:
            return 0.0

        indptr = scores.get("indptr")
        num_docs = int(
            scores.get(
                "num_docs",
                self.chunk_count,
            )
        )
        if (
            indptr is None
            or token_id + 1 >= len(indptr)
        ):
            return 0.0

        df = int(
            indptr[token_id + 1]
            - indptr[token_id]
        )
        if df <= 0:
            return 0.0

        return math.log(
            1.0
            + (num_docs + 1)
            / (df + 1)
        )

    def select_anchor_terms(
        self,
        query: str,
        language: str,
        max_terms: int = MAX_ANCHOR_TERMS,
    ) -> list[str]:
        terms = _content_query_terms(
            query,
            language,
        )
        if not terms:
            return []

        scored = []
        for position, term in enumerate(terms):
            idf = self.idf_proxy(term)
            if idf <= 0:
                continue
            scored.append(
                (
                    idf,
                    position,
                    term,
                )
            )

        if not scored:
            return terms[:max_terms]

        # Highest-IDF terms are most informative.
        selected = sorted(
            scored,
            key=lambda item: (
                -item[0],
                item[1],
            ),
        )[:max_terms]

        # Restore natural query order.
        selected.sort(
            key=lambda item: item[1]
        )

        return [
            term
            for _, _, term in selected
        ]

    def build_query_views(
        self,
        query: str,
        language: str,
    ) -> list[dict]:
        query = _normalize_retrieval_text(query)
        anchors = self.select_anchor_terms(
            query,
            language,
        )
        views = [
            {
                "name": "full",
                "text": query,
                "anchors": anchors,
            }
        ]

        if anchors:
            anchor_query = " ".join(anchors)
            # Adaptive fast-path: Only trigger 4-lane multi-view if anchor query
            # contains at least one strong rare/entity term (IDF >= 2.0).
            has_strong_entity = any(
                self.idf_proxy(term) >= 2.0
                for term in anchors
            )
            if (
                anchor_query
                and anchor_query != query
                and has_strong_entity
            ):
                views.append(
                    {
                        "name": "anchor",
                        "text": anchor_query,
                        "anchors": anchors,
                    }
                )

        return views


    def search_with_evidence(
        self,
        query,
        views=None,
    ):
        start = time.perf_counter()
        if views is None:
            views = self.build_query_views(
                query,
                self.language,
            )

        texts = [view["text"] for view in views]
        tokens = self.tokenizer.tokenize(
            texts,
            update_vocab=False,
            return_as="tuple",
        )

        indices, scores = self.retriever.retrieve(
            tokens,
            k=min(
                BM25_CHUNK_K,
                self.chunk_count,
            ),
        )

        rankings = {}
        evidence = {}

        for view_index, view in enumerate(views):
            name = view["name"]
            parents = []
            seen = set()

            for chunk_index, raw_score in zip(
                indices[view_index],
                scores[view_index],
            ):
                raw_score = float(raw_score)
                if raw_score <= 0:
                    continue

                idx = int(chunk_index)
                parent_id = self.parent_ids[idx]

                if parent_id in seen:
                    continue

                seen.add(parent_id)
                parents.append(parent_id)

                candidate = {
                    "parent_id": parent_id,
                    "chunk_id": self.chunk_ids[idx],
                    "text": self.chunk_texts[idx],
                    "score": raw_score,
                    "lane": f"bm25_{name}",
                    "rank": len(parents),
                }

                current = evidence.get(parent_id)
                if (
                    current is None
                    or raw_score > float(current.get("score", 0.0))
                ):
                    if current is not None:
                        candidate["alternate"] = current
                    evidence[parent_id] = candidate

                if len(parents) >= SPARSE_PARENT_K:
                    break

            rankings[f"sparse_{name}"] = parents

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        return {
            "rankings": rankings,
            "evidence": evidence,
            "views": views,
            "elapsed_ms": elapsed_ms,
        }

    def alignment_features(
        self,
        query: str,
        text: str,
        language: str,
        anchors: list[str],
    ):
        query_terms = _content_query_terms(
            query,
            language,
        )
        evidence_terms = word_splitter(
            _normalize_retrieval_text(text)
        )

        if not query_terms:
            return 0.0, 0.0

        total_weight = 0.0
        matched_weight = 0.0

        for term in query_terms:
            idf = self.idf_proxy(term)
            weight = max(
                1.0,
                min(idf, 8.0),
            )
            total_weight += weight
            if any(
                _token_matches(
                    term,
                    candidate,
                    language,
                )
                for candidate in evidence_terms
            ):
                matched_weight += weight

        query_coverage = (
            matched_weight / total_weight
            if total_weight
            else 0.0
        )

        if not anchors:
            anchor_coverage = 0.0
        else:
            matched = 0
            for anchor in anchors:
                if any(
                    _token_matches(
                        anchor,
                        token,
                        language,
                    )
                    for token in evidence_terms
                ):
                    matched += 1
            anchor_coverage = (
                matched / len(anchors)
            )

        return (
            query_coverage,
            anchor_coverage,
        )

    def search(
        self,
        query,
    ):
        res = self.search_with_evidence(query)
        # Compatibility fallback for first sparse view
        sparse_parents = next(
            iter(res["rankings"].values()),
            [],
        )
        return (
            sparse_parents,
            res["elapsed_ms"],
        )



def _token_matches(
    query_token: str,
    evidence_token: str,
    language: str,
) -> bool:
    if query_token == evidence_token:
        return True
    if language != "ta":
        return False
    a = query_token
    b = evidence_token
    shortest = min(len(a), len(b))
    if shortest < 5:
        return False
    common = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        common += 1
    return common >= 4 and (common / shortest) >= 0.70


# =============================================================================
# WEIGHTED RRF MULTI-LANE (V3)
# =============================================================================

def weighted_rrf_multi(
    rankings: dict[str, list[str]],
    weights: dict[str, float],
    k: float = RRF_K_V3,
):
    scores = {}
    lane_membership = {}
    for lane, parents in rankings.items():
        weight = float(weights.get(lane, 1.0))
        for rank, parent in enumerate(parents[:FUSION_DEPTH], start=1):
            contribution = weight / (k + rank)
            scores[parent] = scores.get(parent, 0.0) + contribution
            lane_membership.setdefault(parent, set()).add(lane)

    ordered = sorted(
        scores,
        key=lambda parent: (-scores[parent], parent),
    )
    return (
        ordered[:FUSION_CANDIDATE_K],
        scores,
        lane_membership,
    )



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

    def best_candidate(
        self,
        language,
        query,
        parent,
        evidence_by_parent,
    ):
        candidates = self._all_parent_candidates(
            language,
            str(parent),
            evidence_by_parent,
        )
        if not candidates:
            return None

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
            PER_CHUNK_CHARS,
            language,
        )

        if not snippet:
            return None

        return {
            "candidate": selected,
            "snippet": snippet,
            "evidence_score": evidence_pack_score(
                query,
                snippet,
                language,
            ),
        }

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

        # Parents arrive already query-aware reranked by Retrieval V3.
        # Preserve this final reranked order while selecting the best
        # child/sibling inside each parent.
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
    return _clean_generated_answer(answer)


def _clean_generated_answer(
    answer: str,
) -> str:
    answer = (
        str(answer)
        .strip()
    )

    prefixes = (
        "Answer:",
        "answer:",
        "பதில்:",
        "উত্তর:",
    )

    for prefix in prefixes:
        if answer.startswith(prefix):
            answer = (
                answer[len(prefix):]
                .strip()
            )

    # Keep NOT_FOUND exact.
    if answer.upper() == "NOT_FOUND":
        return "NOT_FOUND"

    return answer


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

class Seq2SeqFirstTokenStreamer(
    BaseStreamer
):

    def __init__(
        self,
        tokenizer,
    ):
        self.tokenizer = tokenizer

        # mT0 first emits decoder start token.
        self.ignore_initial = True

        self.first_token_at = None
        self.completed_at = None

        self.generated_ids = []
        self.done = threading.Event()

    def put(
        self,
        value,
    ):
        ids = (
            value
            .detach()
            .cpu()
            .reshape(-1)
            .tolist()
        )

        if self.ignore_initial:
            self.ignore_initial = False
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


FirstTokenStreamer = Seq2SeqFirstTokenStreamer


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

    def _rerank_parents(
        self,
        *,
        query,
        language,
        parents,
        evidence_by_parent,
        rrf_scores,
        lane_membership,
        anchors,
    ):
        records = []
        for parent in parents[:RERANK_PARENT_K]:
            best = self.contexts.best_candidate(
                language,
                query,
                parent,
                evidence_by_parent,
            )
            if not best:
                continue

            snippet = best["snippet"]
            query_coverage, anchor_coverage = self.bm25[language].alignment_features(
                query,
                snippet,
                language,
                anchors,
            )

            lanes = lane_membership.get(parent, set())
            dense_hit = any(lane.startswith("dense") for lane in lanes)
            sparse_hit = any(lane.startswith("sparse") for lane in lanes)
            agreement = float(dense_hit and sparse_hit)

            normalized_query = _normalize_retrieval_text(query)
            normalized_snippet = _normalize_retrieval_text(snippet)
            anchor_phrase = " ".join(anchors) if anchors else ""
            phrase_hit = float(
                bool(anchor_phrase and anchor_phrase in normalized_snippet)
            )

            records.append({
                "parent": parent,
                "rrf_raw": float(rrf_scores.get(parent, 0.0)),
                "evidence_raw": float(best["evidence_score"]),
                "query_coverage": query_coverage,
                "anchor_coverage": anchor_coverage,
                "agreement": agreement,
                "phrase_hit": phrase_hit,
                "snippet": snippet,
            })

        if not records:
            return [], {}

        def minmax(key):
            values = [row[key] for row in records]
            lo = min(values)
            hi = max(values)
            if abs(hi - lo) < 1e-9:
                return {row["parent"]: 0.5 for row in records}
            return {row["parent"]: (row[key] - lo) / (hi - lo) for row in records}

        rrf_norm = minmax("rrf_raw")
        evidence_norm = minmax("evidence_raw")

        metadata = {}
        for row in records:
            parent = row["parent"]
            score = (
                RR_WEIGHT * rrf_norm[parent]
                + ANCHOR_WEIGHT * row["anchor_coverage"]
                + QUERY_COVERAGE_WEIGHT * row["query_coverage"]
                + EVIDENCE_WEIGHT * evidence_norm[parent]
                + LANE_AGREEMENT_WEIGHT * row["agreement"]
                + PHRASE_WEIGHT * row["phrase_hit"]
            )
            if anchors and row["anchor_coverage"] == 0.0:
                score -= ANCHOR_MISS_PENALTY

            row["final_score"] = score
            metadata[parent] = row

        records.sort(
            key=lambda row: (-row["final_score"], row["parent"])
        )

        reranked = [row["parent"] for row in records[:FINAL_PARENT_K]]
        return reranked, metadata

    # =========================================================================
    # RETRIEVAL (V3 MULTI-VIEW ANCHOR-AWARE)
    # =========================================================================

    def retrieve(
        self,
        query,
        language="ta",
    ):
        overall = time.perf_counter()

        views = self.bm25[language].build_query_views(query, language)

        sparse_future = self.executor.submit(
            self.bm25[language].search_with_evidence,
            query,
            views,
        )

        torch.cuda.synchronize()
        embed_start = time.perf_counter()

        query_texts = [view["text"] for view in views]
        vectors = self.embedder.encode(
            query_texts,
            prompt_name="query",
            truncate_dim=EMBED_DIM,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        torch.cuda.synchronize()
        embed_ms = (time.perf_counter() - embed_start) * 1000

        dense_start = time.perf_counter()
        dense_rankings = {}
        dense_evidence = {}

        def _execute_qdrant_view(v_item):
            v, vec = v_item
            res = self.qdrant.query_points(
                collection_name=CFG[language]["collection"],
                query=vec.tolist(),
                limit=DENSE_CHILD_K,
                with_payload=["parent_id", "chunk_id"],
                with_vectors=False,
                search_params=models.SearchParams(
                    hnsw_ef=HNSW_EF,
                    exact=False,
                    indexed_only=True,
                ),
            )
            return v, res

        items = list(zip(views, vectors))
        if len(items) > 1:
            with ThreadPoolExecutor(max_workers=len(items)) as q_executor:
                qdrant_results = list(q_executor.map(_execute_qdrant_view, items))
        else:
            qdrant_results = [_execute_qdrant_view(items[0])]

        for view, response in qdrant_results:
            parents = []
            seen = set()
            view_name = view["name"]

            for point in response.points:
                payload = point.payload or {}
                parent = str(payload.get("parent_id", ""))
                if not parent or parent in seen:
                    continue

                seen.add(parent)
                parents.append(parent)

                candidate = {
                    "parent_id": parent,
                    "chunk_id": str(payload.get("chunk_id", "")),
                    "score": float(point.score),
                    "lane": f"dense_{view_name}",
                    "rank": len(parents),
                }

                current = dense_evidence.get(parent)
                if current is None:
                    dense_evidence[parent] = candidate
                else:
                    candidate["alternate"] = current
                    dense_evidence[parent] = candidate

                if len(parents) >= DENSE_PARENT_K:
                    break

            dense_rankings[f"dense_{view_name}"] = parents

        dense_ms = (time.perf_counter() - dense_start) * 1000


        sparse_result = sparse_future.result()

        rankings = {}
        rankings.update(dense_rankings)
        rankings.update(sparse_result["rankings"])

        cfg = CFG[language]
        weights = {
            "dense_full": cfg["dense_weight"] * DENSE_FULL_WEIGHT,
            "dense_anchor": cfg["dense_weight"] * DENSE_ANCHOR_WEIGHT,
            "sparse_full": cfg["sparse_weight"] * SPARSE_FULL_WEIGHT,
            "sparse_anchor": cfg["sparse_weight"] * SPARSE_ANCHOR_WEIGHT,
        }

        fused_candidates, rrf_scores, lane_membership = weighted_rrf_multi(
            rankings,
            weights,
        )

        evidence_candidates = {}
        for source in (dense_evidence, sparse_result["evidence"]):
            for parent, candidate in source.items():
                if parent not in evidence_candidates:
                    evidence_candidates[parent] = candidate
                else:
                    current = evidence_candidates[parent]
                    merged = dict(candidate)
                    merged["alternate"] = current
                    evidence_candidates[parent] = merged

        anchors = views[0].get("anchors", [])

        rerank_start = time.perf_counter()
        reranked, rerank_metadata = self._rerank_parents(
            query=query,
            language=language,
            parents=fused_candidates,
            evidence_by_parent=evidence_candidates,
            rrf_scores=rrf_scores,
            lane_membership=lane_membership,
            anchors=anchors,
        )
        rerank_ms = (time.perf_counter() - rerank_start) * 1000

        retrieval_ms = (time.perf_counter() - overall) * 1000

        return {
            "parents": reranked,
            "pre_rerank_parents": fused_candidates,
            "rankings": rankings,
            "evidence_by_parent": evidence_candidates,
            "rerank_metadata": rerank_metadata,
            "anchors": anchors,
            "embed_ms": embed_ms,
            "dense_ms": dense_ms,
            "bm25_ms": sparse_result["elapsed_ms"],
            "rerank_ms": rerank_ms,
            "retrieval_ms": retrieval_ms,
        }



    # =========================================================================
    # GENERATION
    # =========================================================================

    @torch.inference_mode()
    def generate(
        self,
        query,
        context,
        max_new_tokens=None,
        language="ta",
    ):

        stage_start = time.perf_counter()

        max_new_tokens = (
            max_new_tokens
            or settings.gen_max_new_tokens
        )

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

        streamer = Seq2SeqFirstTokenStreamer(
            self.gen_tokenizer
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

        outputs = self.generator.generate(
            **kwargs
        )

        torch.cuda.synchronize()
        model_complete = time.perf_counter()

        first_token_at = (
            streamer.first_token_at
            or model_start
        )

        raw_answer = self.gen_tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        ).strip()

        answer = _clean_generated_answer(
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
            (first_token_at - model_start) * 1000
            if first_token_at
            else None
        )

        complete_ms = (
            model_complete - model_start
        ) * 1000

        generated_tokens = int(
            outputs[0].shape[0]
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
            "generation_stage_ttft_ms": (
                (prep_ms + first_token_ms)
                if first_token_ms is not None
                else None
            ),
            "generation_complete_ms": (
                prep_ms + complete_ms
            ),
            "first_token_at": first_token_at,
            "completed_at": model_complete,
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