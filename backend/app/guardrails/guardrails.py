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
import unicodedata
from dataclasses import dataclass
from typing import Optional

SUPPORTED_LANGUAGES = {"ta", "hi"}

MAX_QUERY_CHARS = 500
NOT_FOUND_SENTINEL = "NOT_FOUND"

_CITATION_RE = re.compile(r"\[\s*\d+\s*\]")
_CITATION_ONLY_RE = re.compile(
    r"^(?:\s*\[\s*\d+\s*\]\s*)+$"
)
_TARGET_SCRIPT = {
    "ta": re.compile(r"[\u0B80-\u0BFF]"),
    "hi": re.compile(r"[\u0900-\u097F]"),
}
_NUMERIC_ANSWER_RE = re.compile(
    r"^[\s\d.,:+\-−°%/]+"
    r"(?:c|f|°c|°f|km|kg|m|cm|mm)?$",
    re.IGNORECASE,
)

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


def _terms(text: str) -> list[str]:
    terms = []
    current = []

    for char in str(text).casefold():
        category = unicodedata.category(char)
        if category and category[0] in {"L", "M", "N"}:
            current.append(char)
        elif current:
            terms.append("".join(current))
            current = []

    if current:
        terms.append("".join(current))

    return terms


_MCQ_PREFIX_RE = re.compile(
    r"^[A-Da-d][.)]\s*",
)

_ANSWER_PREFIX_RE = re.compile(
    r"^(?:A|Answer|Ans|उत्तर|பதில்)\s*[:\-–]\s*",
    re.IGNORECASE,
)


def _clean_answer(answer: str) -> str:
    cleaned = str(answer).strip()
    cleaned = cleaned.replace("**", "")
    cleaned = _CITATION_RE.sub("", cleaned)
    # Strip MCQ option prefixes (A. B. C. D.) — the 0.6B model generates these
    # when evidence contains numbered lists that look like answer choices.
    cleaned = _MCQ_PREFIX_RE.sub("", cleaned)
    # Strip A: / Answer: prefixes
    cleaned = _ANSWER_PREFIX_RE.sub("", cleaned)
    # Sanitize Unicode replacement character instead of rejecting the whole answer
    cleaned = cleaned.replace("\ufffd", "")
    cleaned = " ".join(cleaned.split())
    return cleaned.strip(" -–—:;,.|")




def _supported_by_context(answer: str, context: str) -> bool:
    """Require at least one meaningful answer term or number to occur in evidence."""

    answer_terms = [
        term
        for term in _terms(answer)
        if len(term) >= 2
    ]

    # Numeric grounding: if answer contains digits that appear in context, it is grounded
    answer_digits = set(re.findall(r"\d+", answer))
    context_digits = set(re.findall(r"\d+", context))
    if answer_digits and (answer_digits & context_digits):
        return True

    if not answer_terms:
        return bool(
            re.search(r"\d", answer)
            and re.search(r"\d", context)
        )

    context_terms = set(
        _terms(context)
    )

    return any(
        answer_term == context_term
        or (
            min(
                len(answer_term),
                len(context_term),
            ) >= 3
            and (
                answer_term in context_term
                or context_term in answer_term
            )
        )
        for answer_term in answer_terms
        for context_term in context_terms
    )


def _localized_rejection(
    language: str,
    code: str,
    reason: str,
) -> tuple[str, GuardrailResult]:
    message = NOT_FOUND_MESSAGE.get(
        language,
        DEFAULT_NOT_FOUND_MESSAGE,
    )
    return message, GuardrailResult(
        False,
        "output",
        code,
        reason,
    )


def apply_output_guardrail(
    answer: str,
    language: str,
    *,
    query: str = "",
    context: str = "",
    possibly_truncated: bool = False,
) -> tuple[str, GuardrailResult]:
    """Clean harmless formatting and reject malformed/ungrounded output.

    This is deterministic and adds no model call, preserving first-token
    latency. Rejected output is surfaced as the localized NOT_FOUND message.
    """
    stage = "output"

    stripped = answer.strip()
    if stripped == NOT_FOUND_SENTINEL or stripped.startswith(NOT_FOUND_SENTINEL):
        return _localized_rejection(
            language,
            "ungrounded_answer",
            "Model reported NOT_FOUND.",
        )

    if not stripped:
        return _localized_rejection(
            language,
            "empty_answer",
            "Model returned an empty answer.",
        )

    if _CITATION_ONLY_RE.fullmatch(stripped):
        return _localized_rejection(
            language,
            "citation_only",
            "Model returned a citation without an answer.",
        )

    cleaned = _clean_answer(stripped)

    if not cleaned:
        return _localized_rejection(
            language,
            "empty_answer",
            "Model returned an empty answer after cleaning.",
        )

    answer_terms = _terms(cleaned)
    for left, middle, right in zip(
        answer_terms,
        answer_terms[1:],
        answer_terms[2:],
    ):
        if left == middle == right:
            return _localized_rejection(
                language,
                "repeated_answer",
                "Model output contains excessive repetition.",
            )

    query_terms = set(_terms(query))
    if answer_terms and set(answer_terms).issubset(query_terms):
        return _localized_rejection(
            language,
            "question_echo",
            "Model repeated the question instead of answering it.",
        )

    target_script = _TARGET_SCRIPT.get(language)
    # Permissive script check: allow target script, Latin alphabet (for entities/units), or numeric answers
    has_target_script = bool(target_script and target_script.search(cleaned))
    has_latin_or_numbers = bool(re.search(r"[a-zA-Z0-9]", cleaned))
    if not (has_target_script or has_latin_or_numbers or _NUMERIC_ANSWER_RE.fullmatch(cleaned)):
        return _localized_rejection(
            language,
            "wrong_language",
            "Model answered in an unsupported script.",
        )

    if context and not _supported_by_context(
        cleaned,
        context,
    ):
        return _localized_rejection(
            language,
            "unsupported_answer",
            "Generated answer text is not supported by the packed evidence.",
        )

    return cleaned, GuardrailResult(True, stage)


def not_found_response_text(language: str) -> str:
    return NOT_FOUND_MESSAGE.get(language, DEFAULT_NOT_FOUND_MESSAGE)
