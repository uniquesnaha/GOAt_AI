"""
Guardrails around the RAG engine — never inside it.

Every function here wraps `app.rag.engine.FullRAG`'s inputs/outputs; none of
them touch retrieval, fusion, context selection, or generation. Kept
deliberately minimal and language-agnostic: the corpus and queries are
Tamil/Hindi, so naive English keyword "off-topic" blocking would just reject
legitimate questions. The strongest grounding signal this system has is
structural, not a classifier: no retrieved context -> don't answer; model
says NOT_FOUND -> say so gracefully.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

SUPPORTED_LANGUAGES = {"ta", "hi"}

MAX_QUERY_CHARS = 500
NOT_FOUND_SENTINEL = "NOT_FOUND"

# Minimal, non-exhaustive safety net for clearly unsafe requests (self-harm
# instructions, weapon/explosive construction, csam). This is intentionally
# small and conservative — it is not a substitute for a real content
# moderation model, just a last-resort net that doesn't depend on English
# being the query language for the *unsafe* terms themselves (many of these
# concepts get typed in Latin script/English even in Tamil/Hindi chats).
_UNSAFE_PATTERNS = [
    re.compile(r"\bhow\s+to\s+make\s+a?\s*(bomb|explosive)\b", re.IGNORECASE),
    re.compile(r"\bkill\s+myself\b", re.IGNORECASE),
    re.compile(r"\bsuicide\s+method\b", re.IGNORECASE),
    re.compile(r"\bchild\s+sexual\b", re.IGNORECASE),
]

NOT_FOUND_MESSAGE = {
    "ta": "இந்தக் கேள்விக்கான தகவல் எங்கள் தரவுத்தளத்தில் இல்லை.",
    "hi": "इस प्रश्न से जुड़ी जानकारी हमारे डेटा में उपलब्ध नहीं है।",
}

DEFAULT_NOT_FOUND_MESSAGE = "No grounded information was found for this query."


@dataclass
class GuardrailResult:
    allowed: bool
    stage: str
    code: Optional[str] = None
    reason: Optional[str] = None


def check_input(query: str, language: str) -> GuardrailResult:
    stage = "input"

    if not query or not query.strip():
        return GuardrailResult(False, stage, "empty_query", "Query is empty.")

    if len(query) > MAX_QUERY_CHARS:
        return GuardrailResult(
            False, stage, "query_too_long",
            f"Query exceeds {MAX_QUERY_CHARS} characters.",
        )

    if language not in SUPPORTED_LANGUAGES:
        return GuardrailResult(
            False, stage, "unsupported_language",
            f"Language must be one of {sorted(SUPPORTED_LANGUAGES)}.",
        )

    for pattern in _UNSAFE_PATTERNS:
        if pattern.search(query):
            return GuardrailResult(False, stage, "unsafe_content", "Query matched an unsafe-content pattern.")

    return GuardrailResult(True, stage)


def check_grounding(parents: list, context: str) -> GuardrailResult:
    stage = "grounding"

    if not parents or not context.strip():
        return GuardrailResult(
            False, stage, "no_grounded_context",
            "Retrieval returned no usable context for this query.",
        )

    return GuardrailResult(True, stage)


def apply_output_guardrail(answer: str, language: str) -> tuple[str, GuardrailResult]:
    """Rewrites the model's NOT_FOUND sentinel into a graceful message.

    Does not alter *how* the model decided to say NOT_FOUND (that's the
    untouched system prompt in engine.py) — only how it's surfaced.
    """
    stage = "output"

    stripped = answer.strip()
    if stripped == NOT_FOUND_SENTINEL or stripped.startswith(NOT_FOUND_SENTINEL):
        message = NOT_FOUND_MESSAGE.get(language, DEFAULT_NOT_FOUND_MESSAGE)
        return message, GuardrailResult(False, stage, "ungrounded_answer", "Model reported NOT_FOUND.")

    return answer, GuardrailResult(True, stage)


def not_found_response_text(language: str) -> str:
    return NOT_FOUND_MESSAGE.get(language, DEFAULT_NOT_FOUND_MESSAGE)
