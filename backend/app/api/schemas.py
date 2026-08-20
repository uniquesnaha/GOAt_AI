from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


Language = Literal["ta", "hi"]


class QueryRequest(BaseModel):
    query: str
    language: Language


class StageLatencies(BaseModel):
    embed_ms: Optional[float] = None
    dense_ms: Optional[float] = None
    bm25_ms: Optional[float] = None
    retrieval_ms: Optional[float] = None
    context_ms: Optional[float] = None
    prompt_prep_ms: Optional[float] = None
    model_first_token_ms: Optional[float] = None
    generation_ttft_ms: Optional[float] = None
    full_rag_ttft_ms: Optional[float] = None
    request_overhead_ms: Optional[float] = None


class GuardrailInfo(BaseModel):
    blocked: bool
    stage: Optional[str] = None
    code: Optional[str] = None
    reason: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    language: Language
    sources: list[str] = Field(default_factory=list)
    stage_latencies: StageLatencies
    guardrail: GuardrailInfo
    retried: bool = False


class ErrorResponse(BaseModel):
    code: str
    message: str
