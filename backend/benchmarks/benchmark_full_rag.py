"""
Offline latency + quality benchmark.

Verbatim extraction of the evaluation harness from
`scripts/benchmark_full_rag_t4_latency_winner.py` (kept untouched at that
path forever) — same percentile math, same recall@20/hit@20 definitions,
same warmup/measured-run structure, same CLI flags/defaults. It now imports
`FullRAG` from `app.rag.engine` instead of defining it inline, and resolves
`ROOT`/`QUERY_FILES`/`EVAL_FILE`/`OUTPUT` from `app.config.settings` instead
of the hardcoded Colab path. Nothing about *how* latency or quality is
measured has changed.

Run from `backend/`:
    python -m benchmarks.benchmark_full_rag --per-language 50
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd
import torch

from app.config import settings
from app.rag.engine import FullRAG


# =============================================================================
# PATHS
# =============================================================================

ROOT = settings.data_root

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
# TEXT HELPERS
# =============================================================================

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
        default=350,
    )

    parser.add_argument(
        "--max-context-parents",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--per-chunk-chars",
        type=int,
        default=175,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
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
