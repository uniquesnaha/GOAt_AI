
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import threading
import time
import unicodedata

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import bm25s
import numpy as np
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


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path("/content/HH-goa-task2")

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

QUERY_FILES = {
    "ta":
        ROOT / "data" / "eval"
        / "tamil_validation_queries.parquet",

    "hi":
        ROOT / "data" / "eval"
        / "hindi_validation_queries.parquet",
}

EVAL_FILE = (
    ROOT
    / "final_retrieval_tuning"
    / "candidate_ceiling_per_query.parquet"
)

OUTPUT = (
    ROOT
    / "full_rag_t4_results"
)

OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)


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


def parse_list(value):

    if value is None:
        return []

    if isinstance(value, str):

        value = value.strip()

        if value.startswith("["):

            try:
                value = json.loads(value)

            except Exception:
                value = [value]

        else:
            value = [value]

    if hasattr(value, "tolist"):
        value = value.tolist()

    if not isinstance(
        value,
        (list, tuple, set),
    ):
        value = [value]

    return [
        str(x).strip()
        for x in value
        if x is not None
        and str(x).strip()
    ]


# =============================================================================
# EVALUATION
# =============================================================================

def load_queries(language):

    path = QUERY_FILES[
        language
    ]

    df = pd.read_parquet(path)

    id_col = (
        "query_id"
        if "query_id" in df.columns
        else "id"
    )

    query_col = None

    for candidate in [
        "query",
        "translated_query",
        "target_query",
        "Query",
    ]:

        if candidate in df.columns:

            query_col = candidate
            break

    if query_col is None:

        raise RuntimeError(
            f"No query column: {path}"
        )

    return {
        str(qid): str(query)

        for qid, query
        in zip(
            df[id_col],
            df[query_col],
        )
    }


def load_eval():

    df = pd.read_parquet(
        EVAL_FILE
    )

    if "analysis_type" in df.columns:

        df = df[
            df["analysis_type"]
            ==
            "candidate_depth_100"
        ].copy()

    df["query_id"] = (
        df["query_id"]
        .astype(str)
    )

    return df


def deterministic_sample(
    df,
    language,
    n,
):

    subset = (
        df[
            df["language"]
            ==
            language
        ]
        .copy()
    )

    scored = []

    for idx, row in (
        subset.iterrows()
    ):

        key = (
            f"hhgoa-full-rag:"
            f"{language}:"
            f"{row['query_id']}"
        )

        digest = hashlib.sha256(
            key.encode()
        ).hexdigest()

        scored.append(
            (digest, idx)
        )

    scored.sort()

    chosen = [
        idx
        for _, idx
        in scored[:n]
    ]

    return (
        subset.loc[chosen]
        .reset_index(drop=True)
    )


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
        # QDRANT — HTTP localhost.
        # gRPC upload had issues in this Colab,
        # so keep this first integrated test simple.
        # ---------------------------------------------------------

        self.qdrant = QdrantClient(
            url=
                "http://127.0.0.1:6333",

            prefer_grpc=
                False,

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


# =============================================================================
# METRICS
# =============================================================================

def recall_at_20(
    ranking,
    relevant,
):

    relevant = set(relevant)

    if not relevant:
        return 0.0

    return (
        len(
            set(
                ranking[:20]
            )
            &
            relevant
        )
        /
        len(relevant)
    )


def hit_at_20(
    ranking,
    relevant,
):

    return float(
        bool(
            set(
                ranking[:20]
            )
            &
            set(relevant)
        )
    )


def percentiles(
    values,
):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "p50":
            np.percentile(
                values,
                50,
            ),

        "p70":
            np.percentile(
                values,
                70,
            ),

        "p90":
            np.percentile(
                values,
                90,
            ),

        "p95":
            np.percentile(
                values,
                95,
            ),

        "p100":
            np.max(values),

        "mean":
            np.mean(values),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--per-language",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--context-char-budget",
        type=int,
        default=CONTEXT_CHAR_BUDGET,
    )

    parser.add_argument(
        "--max-context-parents",
        type=int,
        default=MAX_CONTEXT_PARENTS,
    )

    parser.add_argument(
        "--per-chunk-chars",
        type=int,
        default=PER_CHUNK_CHARS,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
    )

    args = parser.parse_args()


    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA unavailable"
        )


    eval_df = load_eval()

    lookups = {
        "ta":
            load_queries("ta"),

        "hi":
            load_queries("hi"),
    }


    frames = []

    for language in [
        "ta",
        "hi",
    ]:

        selected = (
            deterministic_sample(
                eval_df,
                language,
                args.per_language,
            )
        )

        selected[
            "query"
        ] = [
            lookups[
                language
            ][
                str(qid)
            ]

            for qid
            in selected[
                "query_id"
            ]
        ]

        frames.append(
            selected
        )


    benchmark = (
        pd.concat(
            frames,
            ignore_index=True,
        )
    )


    engine = FullRAG()


    # -------------------------------------------------------------------------
    # WARMUP
    # -------------------------------------------------------------------------

    print()
    print(
        f"WARMUP — {args.warmup}"
    )


    for i in range(
        args.warmup
    ):

        row = (
            benchmark.iloc[
                i
                %
                len(benchmark)
            ]
        )


        overall = (
            time.perf_counter()
        )


        retrieval = (
            engine.retrieve(
                row["query"],
                row["language"],
            )
        )


        (
            context,
            context_ms,
            context_parents,
        ) = (
            engine.contexts.build(
                row["language"],
                row["query"],
                retrieval["parents"],
                args.context_char_budget,
                args.max_context_parents,
                args.per_chunk_chars,
            )
        )


        generation = (
            engine.generate(
                row["query"],
                context,
                args.max_new_tokens,
            )
        )


        full_ttft = (
            generation[
                "first_token_at"
            ]
            -
            overall
        ) * 1000


        print(
            f"{i+1}: "
            f"ret={retrieval['retrieval_ms']:.1f} "
            f"ctx={context_ms:.1f} "
            f"gen={generation['generation_stage_ttft_ms']:.1f} "
            f"FULL={full_ttft:.1f} ms"
        )


    # -------------------------------------------------------------------------
    # MEASURE
    # -------------------------------------------------------------------------

    results = []


    print()
    print(
        "MEASURED RUN"
    )


    for index, row in (
        benchmark.iterrows()
    ):

        overall_start = (
            time.perf_counter()
        )


        retrieval = (
            engine.retrieve(
                row["query"],
                row["language"],
            )
        )


        (
            context,
            context_ms,
            context_parents,
        ) = (
            engine.contexts.build(
                row["language"],
                row["query"],
                retrieval["parents"],
                args.context_char_budget,
                args.max_context_parents,
                args.per_chunk_chars,
            )
        )


        generation = (
            engine.generate(
                row["query"],
                context,
                args.max_new_tokens,
            )
        )


        full_ttft = (
            generation[
                "first_token_at"
            ]
            -
            overall_start
        ) * 1000


        relevant = parse_list(
            row[
                "relevant_parent_ids"
            ]
        )


        results.append({
            "language":
                row["language"],

            "query_id":
                row["query_id"],

            "query":
                row["query"],

            "recall_20":
                recall_at_20(
                    retrieval[
                        "parents"
                    ],
                    relevant,
                ),

            "hit_20":
                hit_at_20(
                    retrieval[
                        "parents"
                    ],
                    relevant,
                ),

            "embed_ms":
                retrieval[
                    "embed_ms"
                ],

            "dense_ms":
                retrieval[
                    "dense_ms"
                ],

            "bm25_ms":
                retrieval[
                    "bm25_ms"
                ],

            "retrieval_ms":
                retrieval[
                    "retrieval_ms"
                ],

            "context_ms":
                context_ms,

            "context_chars":
                len(context),

            "context_parents":
                context_parents,

            "prompt_tokens":
                generation[
                    "prompt_tokens"
                ],

            "prompt_prep_ms":
                generation[
                    "prompt_prep_ms"
                ],

            "model_first_token_ms":
                generation[
                    "model_first_token_ms"
                ],

            "generation_ttft_ms":
                generation[
                    "generation_stage_ttft_ms"
                ],

            "full_rag_ttft_ms":
                full_ttft,

            "answer":
                generation[
                    "answer"
                ],
        })


        print(
            f"{index+1:02d}/"
            f"{len(benchmark)} "
            f"FULL={full_ttft:.1f} ms"
        )


    df = pd.DataFrame(
        results
    )


    print()
    print("=" * 100)
    print("QUALITY")
    print("=" * 100)

    print(
        "Recall@20:",
        round(
            df[
                "recall_20"
            ].mean(),
            4,
        )
    )

    print(
        "Hit@20   :",
        round(
            df[
                "hit_20"
            ].mean(),
            4,
        )
    )


    print()
    print("=" * 100)
    print("LATENCY — MS")
    print("=" * 100)


    stages = [
        "embed_ms",
        "dense_ms",
        "bm25_ms",
        "retrieval_ms",
        "context_ms",
        "prompt_prep_ms",
        "model_first_token_ms",
        "generation_ttft_ms",
        "full_rag_ttft_ms",
    ]


    for stage in stages:

        s = percentiles(
            df[stage]
        )

        print(
            f"{stage:24s}"
            f"P50={s['p50']:7.2f} "
            f"P70={s['p70']:7.2f} "
            f"P90={s['p90']:7.2f} "
            f"P95={s['p95']:7.2f} "
            f"P100={s['p100']:7.2f}"
        )


    under200 = (
        df[
            "full_rag_ttft_ms"
        ]
        <
        200
    ).mean() * 100


    print()
    print(
        f"FULL RAG <200 ms: "
        f"{under200:.1f}%"
    )


    print()
    print("SAMPLE ANSWERS")
    print("-" * 100)


    for _, row in (
        df.head(5)
        .iterrows()
    ):

        print()
        print(
            "Q:",
            row["query"]
        )

        print(
            "A:",
            row["answer"]
        )

        print(
            "TTFT:",
            round(
                row[
                    "full_rag_ttft_ms"
                ],
                2,
            ),
            "ms",
        )


    df.to_csv(
        OUTPUT
        /
        "full_rag_t4_per_query.csv",

        index=False,

        encoding=
            "utf-8-sig",
    )


if __name__ == "__main__":
    main()
