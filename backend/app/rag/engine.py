"""
RAG serving engine.

This is a verbatim extraction of the retrieval + context + generation logic
from the golden reference script
`scripts/benchmark_full_rag_t4_latency_winner.py` (kept untouched at that
path forever). Only two things were changed relative to that script:

  1. `ROOT` is read from `app.config.settings.data_root` instead of the
     Colab-only hardcoded `/content/HH-goa-task2`.
  2. The Qdrant client points at `app.config.settings.qdrant_url` instead of
     the hardcoded `http://127.0.0.1:6333`.

Retrieval ranking remains identical to the golden script. Context selection
may prefer the more query-relevant child already present in the frozen fused
top-20, and generation keeps the question ahead of evidence so tokenizer
truncation cannot silently remove it. Run the retrieval-only parity gate after
changes to confirm dense, sparse, and fused rankings are unchanged.
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
    AutoTokenizer,
    AutoModelForCausalLM,
)
from transformers.generation.streamers import BaseStreamer

from app.config import settings
from app.rag.evidence_quality import evidence_relevance_score

ROOT = Path(settings.data_root)


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
        /
        "tamil.parquet",

    "hi":
        CHUNK_ROOT
        /
        "hindi.parquet",
}


# =============================================================================
# MODELS
# =============================================================================

EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
GEN_MODEL = "Qwen/Qwen3-0.6B"

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
# FROZEN CONTEXT & GENERATION CONFIG (LATENCY WINNER)
# =============================================================================

CONTEXT_CHAR_BUDGET = 450
MAX_CONTEXT_PARENTS = 2

# Two 220-character evidence blocks fit inside the 450-char budget:
# 220 + "\n\n" + 220 = 442 chars. Snapped to word boundaries to prevent
# broken letter artifacts. Provides 2-source coverage for complex questions.
PER_CHUNK_CHARS = 220

# 40 tokens gives headroom for longer answer entities like
# "प्रशांत महासागर", "1947 ஆகஸ்ட் 15", "கார்பன் டை ஆக்சைடு"
MAX_NEW_TOKENS = 40



# =============================================================================
# TEXT HELPERS
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

        if major in {"L", "M", "N"}:

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
            df["parent_id"]
            .astype(str)
            .tolist()
        )

        self.chunk_texts = (
            df[text_col]
            .fillna("")
            .astype(str)
            .tolist()
        )

        if has_chunk_id:

            self.chunk_ids = (
                df["chunk_id"]
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

        texts = self.chunk_texts

        self.tokenizer = Tokenizer(
            stemmer=None,
            stopwords=[],
            splitter=
                word_splitter,
        )

        print(
            f"Building "
            f"{language.upper()} BM25..."
        )

        corpus_tokens = (
            self.tokenizer.tokenize(
                texts,
                return_as="tuple",
            )
        )

        self.retriever = bm25s.BM25(
            k1=
                cfg["bm25_k1"],

            b=
                cfg["bm25_b"],

            method=
                "lucene",
        )

        self.retriever.index(
            corpus_tokens,
            show_progress=False,
        )

        self.chunk_count = (
            len(texts)
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

            idx = int(chunk_index)

            parent_id = (
                self.parent_ids[idx]
            )

            if parent_id in seen:
                continue

            seen.add(parent_id)
            parents.append(parent_id)

            evidence[
                parent_id
            ] = {
                "parent_id":
                    parent_id,

                "chunk_id":
                    self.chunk_ids[idx],

                "text":
                    self.chunk_texts[idx],

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
        """Compatibility wrapper — returns (parents, elapsed_ms)."""

        (
            parents,
            elapsed_ms,
            _,
        ) = self.search_with_evidence(
            query
        )

        return parents, elapsed_ms


# =============================================================================
# RRF
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
        dense[:FUSION_DEPTH]
    ):

        contribution = (
            1.0
            /
            (
                cfg["rrf_k"]
                +
                (
                    rank0 + 1
                )
                /
                cfg["dense_weight"]
                -
                1
            )
        )

        scores[parent] = (
            scores.get(
                parent,
                0.0,
            )
            +
            contribution
        )

    for rank0, parent in enumerate(
        sparse[:FUSION_DEPTH]
    ):

        contribution = (
            1.0
            /
            (
                cfg["rrf_k"]
                +
                (
                    rank0 + 1
                )
                /
                cfg["sparse_weight"]
                -
                1
            )
        )

        scores[parent] = (
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
            key=lambda x:
                (-x[1], x[0]),
        )[:FINAL_PARENT_K]
    ]


# =============================================================================
# CONTEXTTT
# =============================================================================


def evidence_window(
    text: str,
    query: str,
    max_chars: int,
) -> str:
    """
    Return the ~max_chars character region of `text` that covers the most
    query terms, snapped to complete word boundaries.
    Interrogatives and superlatives are downweighted so the window centres
    on content/entity terms rather than question-type words.
    Does NOT alter retrieval ranking.
    """

    text = " ".join(str(text).split())

    if len(text) <= max_chars:
        return text

    # Interrogatives/superlatives carry no positional signal — they appear
    # in both query and all passages.  Downweight them so the best window
    # centres on noun/entity terms instead.
    _FILLER_TERMS: set[str] = {
        # Tamil interrogatives & superlatives
        "எது", "என்ன", "எந்த", "எத்தனை", "யார்", "எங்கே",
        "உள்ளது", "ஆகும்", "மிக", "மிகவும்", "ஒரே",
        # Hindi interrogatives & superlatives
        "क्या", "कौन", "कौनसा", "कहाँ", "कैसे",
        "है", "हैं", "सबसे", "सबसेबड़ा", "सबसेछोटा",
        "सबसेलंबा", "सबसेतेज", "सबसेलम्बी",
    }

    query_terms = [
        term
        for term in word_splitter(query)
        if len(term) >= 2
    ]

    # Assign each query term a weight: filler/interrogative = 0.2, else 1.0
    term_weights = {
        term: (0.2 if term.casefold() in _FILLER_TERMS else 1.0)
        for term in query_terms
    }

    def _snap_to_words(raw_start: int, raw_end: int) -> str:
        # Snap start forward to word boundary if slicing mid-word
        if raw_start > 0:
            space_after = text.find(" ", raw_start)
            start = (space_after + 1) if (space_after != -1 and space_after < raw_end) else raw_start
        else:
            start = 0

        # Snap end backward to word boundary if slicing mid-word
        if raw_end < len(text):
            space_before = text.rfind(" ", start, raw_end)
            end = space_before if (space_before > start) else raw_end
        else:
            end = len(text)

        return text[start:end].strip()

    if not query_terms:
        return _snap_to_words(0, max_chars)

    folded = text.casefold()

    positions = []

    for term in query_terms:
        pos = folded.find(term.casefold())
        if pos >= 0:
            positions.append(pos)

    if not positions:
        return _snap_to_words(0, max_chars)

    best_window = None
    best_score = -1.0

    for position in positions:
        raw_start = max(
            0,
            position - max_chars // 3,
        )

        raw_end = min(
            len(text),
            raw_start + max_chars,
        )

        window = _snap_to_words(raw_start, raw_end)

        window_terms = set(
            word_splitter(window)
        )

        # Weighted coverage: entity terms count fully, fillers count 0.2
        score = sum(
            weight
            for term, weight in term_weights.items()
            if term.casefold() in {t.casefold() for t in window_terms}
        )

        if score > best_score:
            best_score = score
            best_window = window

    if not best_window:
        return _snap_to_words(0, max_chars)

    return best_window.strip()



class ContextStore:

    def __init__(self):

        self.data = {}
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

            has_chunk_id = (
                "chunk_id" in schema
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
                columns=read_columns,
            )

            mapping = {}
            chunk_mapping = {}

            for row_index, row in (
                df.iterrows()
            ):

                parent = str(
                    row["parent_id"]
                )

                text = str(
                    row[text_col]
                ).strip()

                if not text:
                    continue

                mapping.setdefault(
                    parent,
                    [],
                ).append(text)

                if has_chunk_id:

                    chunk_id = str(
                        row["chunk_id"]
                    )

                    chunk_mapping[
                        chunk_id
                    ] = text

            self.data[
                language
            ] = mapping

            self.chunk_lookup[
                language
            ] = chunk_mapping


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
            evidence_by_parent or {}
        )

        query_tokens = set(
            word_splitter(query)
        )

        # --------------------------------------------------------------
        # Rerank only the frozen fused candidates for context packing.
        # Exact query coverage is a low-cost tie-breaker that helps BM25
        # evidence containing the named entity/fact reach the two-slot
        # prompt. Original fused order remains the deterministic fallback.
        # --------------------------------------------------------------

        original_rank = {
            str(parent): rank
            for rank, parent in enumerate(
                parents
            )
        }

        def evidence_texts(parent):
            evidence = evidence_by_parent.get(
                str(parent)
            )

            if not evidence:
                return []

            texts = []

            for candidate in (
                evidence,
                evidence.get("alternate"),
            ):
                if not candidate:
                    continue

                text = candidate.get("text")
                chunk_id = candidate.get(
                    "chunk_id"
                )

                if not text and chunk_id:
                    text = self.chunk_lookup[
                        language
                    ].get(str(chunk_id))

                if text:
                    texts.append(str(text))

            return texts

        parents = sorted(
            (
                str(parent)
                for parent in parents
            ),
            key=lambda parent: (
                -max(
                    (
                        evidence_relevance_score(
                            query,
                            text,
                            language,
                        )
                        for text in evidence_texts(
                            parent
                        )
                    ),
                    default=0.0,
                ),
                original_rank[parent],
            ),
        )

        blocks = []
        used = 0
        used_evidence = []

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

            # ----------------------------------------------------------
            # Prefer the actual child that caused retrieval.
            # ----------------------------------------------------------

            evidence = evidence_by_parent.get(
                str(parent)
            )

            if evidence:

                selected_chunk_id = (
                    evidence.get("chunk_id")
                )

                selected_lane = (
                    evidence.get("lane", "unknown")
                )

                selected_score = (
                    evidence.get("score")
                )

                # BM25 already carries the actual child text.
                if evidence.get("text"):
                    selected_text = evidence["text"]

                # Dense result: resolve chunk_id -> text.
                elif selected_chunk_id:
                    selected_text = (
                        self.chunk_lookup[
                            language
                        ].get(
                            str(selected_chunk_id)
                        )
                    )

            # ----------------------------------------------------------
            # Compare the preferred and alternate retrieved children.
            # The higher query-coverage child is more useful to the small
            # generator. This changes neither parent fusion nor retrieval.
            # ----------------------------------------------------------

            if evidence:

                alternate = evidence.get(
                    "alternate"
                )

                if alternate:

                    alt_text = (
                        alternate.get("text")
                    )

                    alt_chunk_id = (
                        alternate.get("chunk_id")
                    )

                    # Dense alternate resolves via chunk_lookup.
                    if not alt_text and alt_chunk_id:
                        alt_text = (
                            self.chunk_lookup[
                                language
                            ].get(
                                str(alt_chunk_id)
                            )
                        )

                    if (
                        alt_text
                        and (
                            not selected_text
                            or evidence_relevance_score(
                                query,
                                alt_text,
                                language,
                            )
                            > evidence_relevance_score(
                                query,
                                selected_text,
                                language,
                            )
                        )
                    ):
                        selected_text = alt_text
                        selected_chunk_id = alt_chunk_id
                        selected_lane = (
                            alternate.get(
                                "lane",
                                "alternate",
                            )
                        )
                        selected_score = (
                            alternate.get("score")
                        )

            # ----------------------------------------------------------
            # Compatibility fallback: lexical selection inside parent.
            # ----------------------------------------------------------

            if not selected_text:

                candidates = (
                    self.data[
                        language
                    ].get(
                        str(parent),
                        [],
                    )
                )

                if not candidates:
                    continue

                selected_text = max(
                    candidates,

                    key=lambda t: len(
                        query_tokens
                        &
                        set(word_splitter(t))
                    ),
                )

                selected_lane = "fallback"

            # ----------------------------------------------------------
            # Select a query-centred evidence window instead of blindly
            # truncating the beginning of the chunk.
            # ----------------------------------------------------------

            snippet = evidence_window(
                selected_text,
                query,
                per_chunk_chars,
            )

            # Source numbers remain in used_evidence/UI metadata. Do not put
            # them in the generator prompt: the small model can mistake the
            # label itself for the requested shortest answer span.
            block = snippet

            if (
                used
                +
                len(block)
                >
                char_budget
            ):
                # Use continue (not break) so a shorter later snippet
                # can still fill the remaining budget.
                continue

            blocks.append(block)

            used += (
                len(block) + 2
            )

            used_evidence.append({
                "parent_id":
                    str(parent),

                "chunk_id":
                    selected_chunk_id,

                "lane":
                    selected_lane,

                "score":
                    selected_score,

                "text":
                    snippet,
            })

        context = (
            "\n\n".join(blocks)
        )

        ms = (
            time.perf_counter()
            -
            start
        ) * 1000

        return (
            context,
            ms,
            len(blocks),
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

        # generate() sends the prompt
        # through streamer first.
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


    def end(self):

        self.completed_at = (
            time.perf_counter()
        )

        self.done.set()


# =============================================================================
# FULL ENGINE
# =============================================================================

class FullRAG:

    def __init__(self):

        # ---------------------------------------------------------
        # QDRANT
        # ---------------------------------------------------------

        self.qdrant = QdrantClient(
            url=
                settings.qdrant_url,

            prefer_grpc=
                settings.qdrant_prefer_grpc,

            timeout=
                30,

            check_compatibility=
                False,
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

        # generate() budgets evidence so the final prompt fits. Right-side
        # truncation remains an additional safety net for unexpected template
        # overhead; the explicit budget normally prevents it from activating.
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
        # BM25
        # ---------------------------------------------------------

        self.bm25 = {
            "ta":
                BM25Engine("ta"),

            "hi":
                BM25Engine("hi"),
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
            torch.cuda.get_device_name(0)
        )

        print(
            "GPU allocated:",
            round(
                torch.cuda.memory_allocated()
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

        sparse_future = (
            self.executor.submit(
                self.bm25[
                    language
                ].search_with_evidence,
                query,
            )
        )


        # EMBEDDING

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
            time.perf_counter()
            -
            embed_start
        ) * 1000


        # DENSE

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

            seen.add(parent)
            dense.append(parent)

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
            time.perf_counter()
            -
            dense_start
        ) * 1000


        (
            sparse,
            bm25_ms,
            sparse_evidence,
        ) = sparse_future.result()


        fused = weighted_rrf(
            dense,
            sparse,
            language,
        )


        # -----------------------------------------------------------------
        # EVIDENCE SELECTION
        #
        # For each parent in the already-ranked fused list, pick the child
        # (dense or BM25) whose RRF contribution was larger. This does not
        # affect the fused ranking — it only determines which actual child
        # text is forwarded to ContextStore.
        # -----------------------------------------------------------------

        cfg = CFG[language]

        dense_rank = {
            parent: rank
            for rank, parent
            in enumerate(dense, start=1)
        }

        sparse_rank = {
            parent: rank
            for rank, parent
            in enumerate(sparse, start=1)
        }

        def _rrf_contribution(rank, weight):
            if rank is None:
                return 0.0
            return (
                1.0
                /
                (
                    cfg["rrf_k"]
                    +
                    rank / weight
                    -
                    1
                )
            )

        evidence_by_parent = {}

        for parent in fused:

            dense_info = (
                dense_evidence.get(parent)
            )

            sparse_info = (
                sparse_evidence.get(parent)
            )

            dense_contrib = _rrf_contribution(
                dense_rank.get(parent),
                cfg["dense_weight"],
            )

            sparse_contrib = _rrf_contribution(
                sparse_rank.get(parent),
                cfg["sparse_weight"],
            )

            if (
                dense_info
                and
                dense_contrib >= sparse_contrib
            ):
                # Dense wins — keep sparse as alternate in case
                # chunk_lookup can't resolve the dense chunk_id.
                selected = dict(dense_info)
                if sparse_info:
                    selected["alternate"] = sparse_info
                evidence_by_parent[parent] = selected

            elif sparse_info:
                # Sparse wins — keep dense as alternate.
                selected = dict(sparse_info)
                if dense_info:
                    selected["alternate"] = dense_info
                evidence_by_parent[parent] = selected

            elif dense_info:
                # Dense only, no sparse match.
                evidence_by_parent[parent] = dict(
                    dense_info
                )


        retrieval_ms = (
            time.perf_counter()
            -
            overall
        ) * 1000


        return {
            "parents":
                fused,

            # Exposed for the retrieval parity gate (dense + sparse + fused).
            # Ignored by all production callers which read ["parents"] only.
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
    ):

        stage_start = (
            time.perf_counter()
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a fact extractor. "
                    "Find the specific name, number, or value in C that directly answers Q. "
                    "Output ONLY that extracted word or phrase — never repeat words from Q. "
                    "If C has no relevant information at all, output NOT_FOUND."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Q:\n{query}\n\n"
                    f"C:\n{context}"
                ),
            },
        ]

        # apply_chat_template adds <|im_start|>assistant\n — we then
        # immediately prefix "Answer:" so the model completes from that
        # cue rather than generating conversationally.
        prompt = (
            self.gen_tokenizer
            .apply_chat_template(
                messages,

                tokenize=False,

                add_generation_prompt=True,

                enable_thinking=False,
            )
            + "Answer:"
        )


        inputs = (
            self.gen_tokenizer(
                prompt,

                return_tensors=
                    "pt",

                truncation=
                    True,

                max_length=
                    850,
            )
            .to("cuda")
        )


        prompt_tokens = int(
            inputs["input_ids"].shape[1]
        )

        if prompt_tokens >= 800:
            print(
                f"⚠️ Prompt near truncation: "
                f"{prompt_tokens} tokens"
            )



        prep_ms = (
            time.perf_counter()
            -
            stage_start
        ) * 1000


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


        answer = (
            self.gen_tokenizer.decode(
                streamer.generated_ids,

                skip_special_tokens=
                    True,
            )
        )


        first_token_ms = (
            streamer.first_token_at
            -
            model_start
        ) * 1000


        complete_ms = (
            streamer.completed_at
            -
            model_start
        ) * 1000


        return {
            "answer":
                answer.strip(),

            "prompt_tokens":
                prompt_tokens,

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

            # Absolute timestamps for clean complete-latency calculation.
            # Callers compute full_rag_complete_ms as:
            #   (completed_at - overall_start) * 1000
            "first_token_at":
                streamer.first_token_at,

            "completed_at":
                streamer.completed_at,

            # Telemetry for truncation-rate benchmark.
            # Correction 3: token-limit hit only — no EOS ID dependency.
            "generated_tokens":
                len(streamer.generated_ids),

            "possibly_truncated": (
                len(streamer.generated_ids)
                >=
                max_new_tokens
            ),
        }
