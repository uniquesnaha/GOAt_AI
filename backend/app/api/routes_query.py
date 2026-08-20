"""
POST /api/query — the orchestration harness.

Structured stages, each isolated and timed: input guardrail -> retrieve
(single whole-call retry on a transient error, flagged) -> context build ->
grounding guardrail -> generate -> output guardrail -> structured response.

Two latency numbers are reported separately on purpose (see
`backend/benchmarks/benchmark_full_rag.py` and DATA_SETUP.md /
gcloud README for the offline P50/P70/P100 measurement this mirrors):

  - `full_rag_ttft_ms`: the same "core RAG" window the offline benchmark
    measures — from just before `engine.retrieve()` to the model's first
    generated token.
  - `request_overhead_ms`: everything else in the HTTP request (guardrails,
    FastAPI/pydantic overhead) — never folded into the RAG number.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, Request

from app.api.schemas import ErrorResponse, GuardrailInfo, QueryRequest, QueryResponse, StageLatencies
from app.guardrails import guardrails
from app.rag.engine import CONTEXT_CHAR_BUDGET, MAX_CONTEXT_PARENTS, MAX_NEW_TOKENS, PER_CHUNK_CHARS

router = APIRouter()


@router.post(
    "/api/query",
    response_model=QueryResponse,
    responses={503: {"model": ErrorResponse}},
)
def query(payload: QueryRequest, request: Request) -> QueryResponse:
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="RAG engine is not ready yet (models/data still loading).",
        )

    wall_start = time.perf_counter()

    input_check = guardrails.check_input(payload.query, payload.language)
    if not input_check.allowed:
        return QueryResponse(
            answer="",
            language=payload.language,
            sources=[],
            stage_latencies=StageLatencies(
                request_overhead_ms=(time.perf_counter() - wall_start) * 1000
            ),
            guardrail=GuardrailInfo(
                blocked=True,
                stage=input_check.stage,
                code=input_check.code,
                reason=input_check.reason,
            ),
        )

    retried = False
    rag_start = time.perf_counter()

    try:
        retrieval = engine.retrieve(payload.query, payload.language)
    except Exception:
        retried = True
        rag_start = time.perf_counter()
        try:
            retrieval = engine.retrieve(payload.query, payload.language)
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Retrieval failed after retry: {exc}",
            ) from exc

    context, context_ms, context_parents = engine.contexts.build(
        payload.language,
        payload.query,
        retrieval["parents"],
        CONTEXT_CHAR_BUDGET,
        MAX_CONTEXT_PARENTS,
        PER_CHUNK_CHARS,
    )

    grounding_check = guardrails.check_grounding(retrieval["parents"], context)
    if not grounding_check.allowed:
        stage_latencies = StageLatencies(
            embed_ms=retrieval["embed_ms"],
            dense_ms=retrieval["dense_ms"],
            bm25_ms=retrieval["bm25_ms"],
            retrieval_ms=retrieval["retrieval_ms"],
            context_ms=context_ms,
            request_overhead_ms=(time.perf_counter() - wall_start) * 1000,
        )
        return QueryResponse(
            answer=guardrails.not_found_response_text(payload.language),
            language=payload.language,
            sources=[],
            stage_latencies=stage_latencies,
            guardrail=GuardrailInfo(
                blocked=True,
                stage=grounding_check.stage,
                code=grounding_check.code,
                reason=grounding_check.reason,
            ),
            retried=retried,
        )

    generation = engine.generate(payload.query, context, MAX_NEW_TOKENS)

    answer, output_check = guardrails.apply_output_guardrail(
        generation["answer"], payload.language
    )

    full_rag_ttft_ms = (generation["first_token_at"] - rag_start) * 1000
    wall_total_ms = (time.perf_counter() - wall_start) * 1000
    request_overhead_ms = max(wall_total_ms - full_rag_ttft_ms, 0.0)

    stage_latencies = StageLatencies(
        embed_ms=retrieval["embed_ms"],
        dense_ms=retrieval["dense_ms"],
        bm25_ms=retrieval["bm25_ms"],
        retrieval_ms=retrieval["retrieval_ms"],
        context_ms=context_ms,
        prompt_prep_ms=generation["prompt_prep_ms"],
        model_first_token_ms=generation["model_first_token_ms"],
        generation_ttft_ms=generation["generation_stage_ttft_ms"],
        full_rag_ttft_ms=full_rag_ttft_ms,
        request_overhead_ms=request_overhead_ms,
    )

    sources = retrieval["parents"][:context_parents]

    return QueryResponse(
        answer=answer,
        language=payload.language,
        sources=sources,
        stage_latencies=stage_latencies,
        guardrail=GuardrailInfo(
            blocked=not output_check.allowed,
            stage=output_check.stage,
            code=output_check.code,
            reason=output_check.reason,
        ),
        retried=retried,
    )
