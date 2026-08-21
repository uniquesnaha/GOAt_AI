"""
POST /api/query — GOAt AI orchestration harness.

Pipeline:

input guardrail
    ↓
frozen hybrid retrieval
    ↓
evidence-aware context packing
    ↓
grounding guardrail
    ↓
Qwen generation
    ↓
output guardrail
    ↓
structured grounded response

Important:
- Retrieval ranking is NOT modified here.
- Top-20 weighted RRF remains frozen.
- evidence_by_parent only preserves which child produced the retrieval hit.
- ContextStore decides which evidence snippets are actually passed to Qwen.
"""

from __future__ import annotations

import time

from fastapi import (
    APIRouter,
    HTTPException,
    Request,
)

from app.api.schemas import (
    ErrorResponse,
    GuardrailInfo,
    QueryRequest,
    QueryResponse,
    SourceEvidence,
    StageLatencies,
)

from app.guardrails import guardrails

from app.rag.engine import (
    CONTEXT_CHAR_BUDGET,
    MAX_CONTEXT_PARENTS,
    MAX_NEW_TOKENS,
    PER_CHUNK_CHARS,
)


router = APIRouter()


@router.post(
    "/api/query",
    response_model=QueryResponse,
    responses={
        503: {
            "model": ErrorResponse
        }
    },
)
def query(
    payload: QueryRequest,
    request: Request,
) -> QueryResponse:

    # =====================================================================
    # ENGINE READINESS
    # =====================================================================

    engine = getattr(
        request.app.state,
        "engine",
        None,
    )

    if engine is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "RAG engine is not ready yet "
                "(models/data still loading)."
            ),
        )


    wall_start = (
        time.perf_counter()
    )


    # =====================================================================
    # 1. INPUT GUARDRAIL
    # =====================================================================

    input_check = (
        guardrails.check_input(
            payload.query,
            payload.language,
        )
    )


    if not input_check.allowed:

        wall_ms = (
            (
                time.perf_counter()
                -
                wall_start
            )
            *
            1000
        )

        return QueryResponse(
            answer="",

            language=
                payload.language,

            sources=[],

            stage_latencies=
                StageLatencies(
                    request_overhead_ms=
                        wall_ms,
                ),

            guardrail=
                GuardrailInfo(
                    blocked=True,

                    stage=
                        input_check.stage,

                    code=
                        input_check.code,

                    reason=
                        input_check.reason,
                ),

            retried=False,
        )


    # =====================================================================
    # 2. RETRIEVAL
    # =====================================================================

    retried = False

    rag_start = (
        time.perf_counter()
    )


    try:

        retrieval = (
            engine.retrieve(
                payload.query,
                payload.language,
            )
        )

    except Exception:

        # --------------------------------------------------------------
        # Single retry.
        #
        # IMPORTANT:
        # This retries the SAME frozen retrieval operation.
        # It does NOT invoke any fallback retrieval algorithm.
        # --------------------------------------------------------------

        retried = True

        rag_start = (
            time.perf_counter()
        )

        try:

            retrieval = (
                engine.retrieve(
                    payload.query,
                    payload.language,
                )
            )

        except Exception as exc:

            raise HTTPException(
                status_code=503,

                detail=(
                    "Retrieval failed after retry: "
                    f"{exc}"
                ),
            ) from exc


    # =====================================================================
    # 3. EVIDENCE-AWARE CONTEXT PACKING
    # =====================================================================

    (
        context,
        context_ms,
        context_parents,
        used_evidence,
    ) = (
        engine.contexts.build(
            payload.language,

            payload.query,

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


    # =====================================================================
    # Convert actual packed evidence into API source objects.
    #
    # IMPORTANT:
    # These are the snippets that really reached Qwen.
    #
    # Do NOT return retrieval["parents"][:N] anymore because that only
    # proves ranking, not what evidence was actually inside the prompt.
    # =====================================================================

    sources = []

    for evidence in used_evidence:

        sources.append(
            SourceEvidence(
                parent_id=
                    str(
                        evidence[
                            "parent_id"
                        ]
                    ),

                chunk_id=(
                    str(
                        evidence[
                            "chunk_id"
                        ]
                    )
                    if evidence.get(
                        "chunk_id"
                    )
                    else None
                ),

                lane=
                    evidence.get(
                        "lane"
                    ),

                score=(
                    float(
                        evidence[
                            "score"
                        ]
                    )
                    if evidence.get(
                        "score"
                    )
                    is not None
                    else None
                ),

                text=
                    evidence[
                        "text"
                    ],
            )
        )


    # =====================================================================
    # 4. STRUCTURAL GROUNDING GUARDRAIL
    # =====================================================================

    grounding_check = (
        guardrails.check_grounding(
            retrieval[
                "parents"
            ],
            context,
        )
    )


    if not grounding_check.allowed:

        elapsed_ms = (
            (
                time.perf_counter()
                -
                wall_start
            )
            *
            1000
        )


        stage_latencies = (
            StageLatencies(
                embed_ms=
                    retrieval[
                        "embed_ms"
                    ],

                dense_ms=
                    retrieval[
                        "dense_ms"
                    ],

                bm25_ms=
                    retrieval[
                        "bm25_ms"
                    ],

                retrieval_ms=
                    retrieval[
                        "retrieval_ms"
                    ],

                context_ms=
                    context_ms,

                request_overhead_ms=
                    elapsed_ms,
            )
        )


        return QueryResponse(
            answer=
                guardrails
                .not_found_response_text(
                    payload.language
                ),

            language=
                payload.language,

            # There normally won't be sources if context is empty,
            # but return whatever was actually packed.
            sources=
                sources,

            stage_latencies=
                stage_latencies,

            guardrail=
                GuardrailInfo(
                    blocked=True,

                    stage=
                        grounding_check.stage,

                    code=
                        grounding_check.code,

                    reason=
                        grounding_check.reason,
                ),

            retried=
                retried,
        )


    # =====================================================================
    # 5. GENERATION
    #
    # MAX_NEW_TOKENS is now frozen at 24.
    #
    # Increasing 16 -> 24 affects completion time but should have almost
    # no effect on first-token latency.
    # =====================================================================

    generation = (
        engine.generate(
            payload.query,
            context,
            MAX_NEW_TOKENS,
        )
    )


    # =====================================================================
    # 6. OUTPUT GUARDRAIL
    #
    # Raw NOT_FOUND from Qwen becomes the localized user-facing message.
    # =====================================================================

    (
        answer,
        output_check,
    ) = (
        guardrails
        .apply_output_guardrail(
            generation[
                "answer"
            ],

            payload.language,
        )
    )


    # =====================================================================
    # 7. LATENCY CALCULATION
    # =====================================================================

    # ---------------------------------------------------------------------
    # Core RAG TTFT
    #
    # Starts immediately before retrieval.
    # Ends when Qwen produces its first generated token.
    #
    # This preserves the exact latency boundary used by the benchmark.
    # ---------------------------------------------------------------------

    full_rag_ttft_ms = (
        (
            generation[
                "first_token_at"
            ]
            -
            rag_start
        )
        *
        1000
    )


    # ---------------------------------------------------------------------
    # Full RAG COMPLETE
    #
    # Starts at the same point as TTFT but ends when generation finishes.
    # ---------------------------------------------------------------------

    completed_at = (
        generation.get(
            "completed_at"
        )
    )


    if completed_at is not None:

        full_rag_complete_ms = (
            (
                completed_at
                -
                rag_start
            )
            *
            1000
        )

    else:

        # Compatibility fallback if engine.py has not yet been patched
        # to expose completed_at.
        full_rag_complete_ms = (
            retrieval[
                "retrieval_ms"
            ]
            +
            context_ms
            +
            generation.get(
                "generation_complete_ms",
                0.0,
            )
        )


    # =====================================================================
    # Request overhead
    #
    # Use completed RAG time rather than TTFT here.
    #
    # Previously:
    #
    # wall_total - TTFT
    #
    # incorrectly counted tokens generated AFTER first token as HTTP
    # "overhead".
    # =====================================================================

    wall_total_ms = (
        (
            time.perf_counter()
            -
            wall_start
        )
        *
        1000
    )


    request_overhead_ms = max(
        wall_total_ms
        -
        full_rag_complete_ms,
        0.0,
    )


    # =====================================================================
    # 8. STRUCTURED LATENCY RESPONSE
    # =====================================================================

    stage_latencies = (
        StageLatencies(
            embed_ms=
                retrieval[
                    "embed_ms"
                ],

            dense_ms=
                retrieval[
                    "dense_ms"
                ],

            bm25_ms=
                retrieval[
                    "bm25_ms"
                ],

            retrieval_ms=
                retrieval[
                    "retrieval_ms"
                ],

            context_ms=
                context_ms,

            prompt_prep_ms=
                generation[
                    "prompt_prep_ms"
                ],

            model_first_token_ms=
                generation[
                    "model_first_token_ms"
                ],

            generation_ttft_ms=
                generation[
                    "generation_stage_ttft_ms"
                ],

            generation_complete_ms=
                generation.get(
                    "generation_complete_ms"
                ),

            full_rag_ttft_ms=
                full_rag_ttft_ms,

            full_rag_complete_ms=
                full_rag_complete_ms,

            request_overhead_ms=
                request_overhead_ms,
        )
    )


    # =====================================================================
    # 9. RESPONSE
    # =====================================================================

    return QueryResponse(
        answer=
            answer,

        language=
            payload.language,

        # Actual packed evidence.
        sources=
            sources,

        stage_latencies=
            stage_latencies,

        guardrail=
            GuardrailInfo(
                blocked=
                    not output_check.allowed,

                stage=
                    output_check.stage,

                code=
                    output_check.code,

                reason=
                    output_check.reason,
            ),

        retried=
            retried,
    )