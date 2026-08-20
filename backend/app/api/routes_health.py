from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.config import settings

router = APIRouter()


@router.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@router.get("/readyz")
def readyz(request: Request) -> JSONResponse:
    # Non-2xx on "not ready" is deliberate: docker-compose's `curl -f`
    # healthcheck (see deploy/docker-compose.yml) only fails on HTTP error
    # status, not on response body content — a 200 here would always read
    # as "healthy" even while the engine is still loading or Qdrant is down.
    engine = getattr(request.app.state, "engine", None)
    load_error = getattr(request.app.state, "engine_load_error", None)

    if engine is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "reason": load_error or "models/data not loaded yet — see DATA_SETUP.md",
            },
        )

    try:
        engine.qdrant.get_collections()
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": f"Qdrant unreachable: {exc}"},
        )

    return JSONResponse(status_code=200, content={"status": "ready"})


@router.get("/api/metrics")
def metrics() -> dict:
    csv_path = settings.data_root / "full_rag_t4_results" / "full_rag_t4_per_query.csv"

    if not csv_path.exists():
        return {
            "available": False,
            "reason": (
                "No benchmark results yet. Run: "
                "python -m benchmarks.benchmark_full_rag --per-language 50"
            ),
        }

    from benchmarks.benchmark_full_rag import percentiles

    df = pd.read_csv(csv_path)

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

    report = {}
    for stage in stages:
        if stage in df.columns:
            report[stage] = {k: float(v) for k, v in percentiles(df[stage]).items()}

    under_200 = float((df["full_rag_ttft_ms"] < 200).mean() * 100) if "full_rag_ttft_ms" in df.columns else None

    return {
        "available": True,
        "num_queries": len(df),
        "recall_20": float(df["recall_20"].mean()) if "recall_20" in df.columns else None,
        "hit_20": float(df["hit_20"].mean()) if "hit_20" in df.columns else None,
        "full_rag_under_200ms_pct": under_200,
        "stage_latencies": report,
    }
