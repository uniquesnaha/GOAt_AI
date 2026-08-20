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

Every constant, weight, prompt, and function body below is otherwise
identical to the golden script. `backend/parity/test_parity.py` mechanically
checks that this file and the golden script produce identical retrieval and
generation output for the same queries — do not edit the RAG logic here
without also re-running that gate.
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


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(settings.data_root)

CHUNKS = {
    "ta":
        ROOT / "chunks_25k"
        / "fixed_384_96"
        / "tamil.parquet",

    "hi":
        ROOT / "chunks_25k"
        / "fixed_384_96"
        / "hindi.parquet",
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
            "hhgoa_fixed384_ta",

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
            "hhgoa_fixed384_hi",

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

CONTEXT_CHAR_BUDGET = 350
MAX_CONTEXT_PARENTS = 2
PER_CHUNK_CHARS = 175
MAX_NEW_TOKENS = 16


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

        df = pd.read_parquet(
            path,
            columns=[
                "parent_id",
                text_col,
            ],
        )

        self.parent_ids = (
            df["parent_id"]
            .astype(str)
            .tolist()
        )

        texts = (
            df[text_col]
            .fillna("")
            .astype(str)
            .tolist()
        )

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


    def search(
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

        for idx, score in zip(
            indices[0],
            scores[0],
        ):

            if float(score) <= 0:
                continue

            parent = (
                self.parent_ids[
                    int(idx)
                ]
            )

            if parent in seen:
                continue

            seen.add(parent)
            parents.append(parent)

            if (
                len(parents)
                >=
                SPARSE_PARENT_K
            ):
                break

        ms = (
            time.perf_counter()
            -
            start
        ) * 1000

        return parents, ms


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
# CONTEXT
# =============================================================================

class ContextStore:

    def __init__(self):

        self.data = {}

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

            df = pd.read_parquet(
                path,
                columns=[
                    "parent_id",
                    text_col,
                ],
            )

            mapping = {}

            for parent, text in zip(
                df["parent_id"],
                df[text_col],
            ):

                parent = str(parent)
                text = str(text).strip()

                if text:

                    mapping.setdefault(
                        parent,
                        [],
                    ).append(text)

            self.data[
                language
            ] = mapping


    def build(
        self,
        language,
        query,
        parents,
        char_budget,
        max_parents,
        per_chunk_chars,
    ):

        start = (
            time.perf_counter()
        )

        query_tokens = set(
            word_splitter(query)
        )

        blocks = []

        used = 0

        for parent in parents:

            if (
                len(blocks)
                >=
                max_parents
            ):
                break

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

            best = max(
                candidates,

                key=lambda text:
                    len(
                        query_tokens
                        &
                        set(
                            word_splitter(text)
                        )
                    ),
            )

            best = (
                " ".join(
                    best.split()
                )
                [:per_chunk_chars]
            )

            block = (
                f"[{len(blocks)+1}] "
                f"{best}"
            )

            if (
                used
                +
                len(block)
                >
                char_budget
            ):
                break

            blocks.append(block)

            used += (
                len(block) + 2
            )

        context = (
            "\n\n".join(
                blocks
            )
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
                ].search,
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
                    "parent_id"
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

        for point in response.points:

            parent = str(
                (
                    point.payload
                    or {}
                ).get(
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


        sparse, bm25_ms = (
            sparse_future.result()
        )


        fused = weighted_rrf(
            dense,
            sparse,
            language,
        )


        retrieval_ms = (
            time.perf_counter()
            -
            overall
        ) * 1000


        return {
            "parents":
                fused,

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
                    "Answer only from C. "
                    "Reply briefly in the language of Q. "
                    "If unsupported, reply NOT_FOUND."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"C:\n{context}\n"
                    f"Q:\n{query}"
                ),
            },
        ]


        prompt = (
            self.gen_tokenizer
            .apply_chat_template(
                messages,

                tokenize=False,

                add_generation_prompt=True,

                enable_thinking=False,
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
                    512,
            )
            .to("cuda")
        )

        prompt_tokens = int(
            inputs["input_ids"].shape[1]
        )

        if prompt_tokens >= 500:
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

            "first_token_at":
                streamer.first_token_at,
        }
