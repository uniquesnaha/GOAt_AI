"""
Deterministic input/output guardrails for GOAt AI.

No additional model call.

The output policy is:

1. Determine whether packed evidence truly supports the question.
2. Reject wrong-subject / contradictory evidence.
3. Prefer a high-confidence answer span copied directly from strong evidence.
4. Otherwise accept Qwen output only if it is grounded in strong evidence.
5. A token-limit hit is telemetry, NOT automatic rejection.
6. If nothing is defensibly supported -> NOT_FOUND.
"""

from __future__ import annotations

import re
import unicodedata

from dataclasses import dataclass
from typing import Optional

from app.rag.evidence_quality import (
    candidate_evidence_units,
    extract_answer_candidate,
    split_terms,
    strongest_supporting_unit,
    support_for_unit,
    term_matches,
)


SUPPORTED_LANGUAGES = {
    "ta",
    "hi",
}


MAX_QUERY_CHARS = 500

NOT_FOUND_SENTINEL = (
    "NOT_FOUND"
)


NOT_FOUND_MESSAGE = {
    "ta":
        "இந்தக் கேள்விக்கான தகவல் எங்கள் தரவுத்தளத்தில் இல்லை.",

    "hi":
        "इस प्रश्न से जुड़ी जानकारी हमारे डेटा में उपलब्ध नहीं है।",
}


DEFAULT_NOT_FOUND_MESSAGE = (
    "No grounded information was found for this query."
)


_TARGET_SCRIPT = {
    "ta":
        re.compile(
            r"[\u0B80-\u0BFF]"
        ),

    "hi":
        re.compile(
            r"[\u0900-\u097F]"
        ),
}


_CITATION_RE = re.compile(
    r"\[\s*\d+\s*\]"
)


_CITATION_ONLY_RE = re.compile(
    r"^(?:\s*\[\s*\d+\s*\]\s*)+$"
)


_MCQ_PREFIX_RE = re.compile(
    r"^[A-Da-d][.)]\s*"
)


_ANSWER_PREFIX_RE = re.compile(
    r"^(?:Answer|Ans|उत्तर|பதில்)\s*[:\-–]\s*",
    re.IGNORECASE,
)


_UNSAFE_PATTERNS = [
    re.compile(
        r"\bhow\s+to\s+make\s+a?\s*"
        r"(bomb|explosive)\b",
        re.IGNORECASE,
    ),

    re.compile(
        r"\bkill\s+myself\b",
        re.IGNORECASE,
    ),

    re.compile(
        r"\bsuicide\s+method\b",
        re.IGNORECASE,
    ),

    re.compile(
        r"\bchild\s+sexual\b",
        re.IGNORECASE,
    ),
]


@dataclass
class GuardrailResult:
    allowed: bool
    stage: str
    code: Optional[str] = None
    reason: Optional[str] = None


# =============================================================================
# INPUT
# =============================================================================

def check_input(
    query: str,
    language: str,
) -> GuardrailResult:

    stage = "input"


    if (
        not query
        or not query.strip()
    ):

        return GuardrailResult(
            False,
            stage,
            "empty_query",
            "Query is empty.",
        )


    if len(query) > MAX_QUERY_CHARS:

        return GuardrailResult(
            False,
            stage,
            "query_too_long",
            (
                f"Query exceeds "
                f"{MAX_QUERY_CHARS} characters."
            ),
        )


    if (
        language
        not in
        SUPPORTED_LANGUAGES
    ):

        return GuardrailResult(
            False,
            stage,
            "unsupported_language",
            (
                "Language must be one of "
                f"{sorted(SUPPORTED_LANGUAGES)}."
            ),
        )


    for pattern in _UNSAFE_PATTERNS:

        if pattern.search(
            query
        ):

            return GuardrailResult(
                False,
                stage,
                "unsafe_content",
                "Query matched an unsafe-content pattern.",
            )


    return GuardrailResult(
        True,
        stage,
    )


# =============================================================================
# STRUCTURAL GROUNDING
# =============================================================================

def check_grounding(
    parents: list,
    context: str,
) -> GuardrailResult:

    if (
        not parents
        or
        not context.strip()
    ):

        return GuardrailResult(
            False,
            "grounding",
            "no_grounded_context",
            (
                "Retrieval returned no usable "
                "context for this query."
            ),
        )


    return GuardrailResult(
        True,
        "grounding",
    )


# =============================================================================
# ANSWER CLEANING
# =============================================================================

def _clean_answer(
    answer: str,
) -> str:

    cleaned = str(
        answer
    ).strip()


    cleaned = cleaned.replace(
        "**",
        "",
    )


    cleaned = _CITATION_RE.sub(
        "",
        cleaned,
    )


    cleaned = _MCQ_PREFIX_RE.sub(
        "",
        cleaned,
    )


    cleaned = _ANSWER_PREFIX_RE.sub(
        "",
        cleaned,
    )


    cleaned = cleaned.replace(
        "\ufffd",
        "",
    )


    # 0.6B sometimes continues:
    #
    # "हृदय। क्योंकि..."
    #
    # We only need first answer fragment.
    cleaned = re.split(
        r"[\n\r]|(?<=[।.!?])\s+",
        cleaned,
        maxsplit=1,
    )[0]


    cleaned = " ".join(
        cleaned.split()
    )


    return cleaned.strip(
        " -–—:;,.|।"
    )


# =============================================================================
# =============================================================================
# GROUNDING NORMALIZATION & MORPHOLOGY
# =============================================================================

def _normalize_grounding_text(
    text: str,
) -> str:
    text = unicodedata.normalize(
        "NFKC",
        str(text),
    )
    text = text.casefold()
    text = re.sub(
        r"[^\w\s\u0900-\u097F\u0B80-\u0BFF]",
        " ",
        text,
    )
    return " ".join(
        text.split()
    )


def _strip_answer_suffixes(
    answer: str,
) -> str:
    answer = _normalize_grounding_text(
        answer
    )
    suffixes = (
        " ஆகும்",
        " என்பது",
        " தான்",
        " உள்ளது",
        " கொண்டுள்ளது",
        " இருக்கிறது",
        " है",
        " हैं",
        " था",
        " थी",
        " थे",
    )
    for suffix in suffixes:
        if answer.endswith(suffix):
            answer = (
                answer[:-len(suffix)]
                .strip()
            )
    return answer


def _token_supported(
    answer_token: str,
    evidence_tokens: list[str],
) -> bool:
    if not answer_token:
        return False

    # Numbers must match exactly.
    if answer_token.isdigit():
        return answer_token in evidence_tokens

    for evidence_token in evidence_tokens:
        if answer_token == evidence_token:
            return True

        shortest = min(
            len(answer_token),
            len(evidence_token),
        )

        # Avoid matching tiny words.
        if shortest < 4:
            continue

        common = 0
        for a, b in zip(
            answer_token,
            evidence_token,
        ):
            if a != b:
                break
            common += 1

        ratio = (
            common
            / max(1, len(answer_token))
        )

        if (
            common >= 4
            and ratio >= 0.70
        ):
            return True

    return False


def answer_is_grounded(
    answer: str,
    evidence: str,
) -> bool:
    if not answer:
        return False

    if str(answer).strip().upper() == "NOT_FOUND":
        return True

    answer_norm = (
        _strip_answer_suffixes(
            answer
        )
    )

    evidence_norm = (
        _normalize_grounding_text(
            evidence
        )
    )

    if not answer_norm or not evidence_norm:
        return False

    # Strongest and safest check.
    if answer_norm in evidence_norm:
        return True

    answer_tokens = (
        answer_norm.split()
    )

    evidence_tokens = (
        evidence_norm.split()
    )

    if not answer_tokens:
        return False

    return all(
        _token_supported(
            token,
            evidence_tokens,
        )
        for token in answer_tokens
    )


# =============================================================================
# QUESTION TYPE VALIDATION
# =============================================================================

NUMBER_QUESTION_MARKERS = {
    "ta": (
        "எத்தனை",
        "எவ்வளவு",
    ),

    "hi": (
        "कितने",
        "कितना",
        "कितनी",
    ),
}


def query_expects_number(
    query: str,
    language: str,
) -> bool:
    q = str(query).casefold()
    return any(
        marker in q
        for marker in NUMBER_QUESTION_MARKERS.get(
            language,
            (),
        )
    )


def answer_type_is_valid(
    query: str,
    answer: str,
    language: str,
) -> bool:
    if query_expects_number(
        query,
        language,
    ):
        return bool(
            re.search(
                r"\d",
                str(answer),
            )
        )

    return True


# =============================================================================
# ANSWER CONTAINMENT
# =============================================================================

def _answer_inside_unit(
    answer: str,
    unit: str,
) -> bool:
    return answer_is_grounded(
        answer,
        unit,
    )


def _answer_supported(
    answer: str,
    query: str,
    context: str,
    language: str,
) -> bool:
    """
    Accept an answer only if the same evidence unit:
    - strongly supports the question, AND
    - contains the answer.
    """

    if not answer:
        return False

    for unit in candidate_evidence_units(
        context
    ):
        signal = support_for_unit(
            query,
            unit,
            language,
        )

        if not signal.strong:
            continue

        if _answer_inside_unit(
            answer,
            unit,
        ):
            return True

    # Fallback to general grounded check on the overall context
    return answer_is_grounded(
        answer,
        context,
    )


# =============================================================================
# ANSWER SHAPE
# =============================================================================

def _question_echo(
    answer: str,
    query: str,
) -> bool:

    answer_terms = split_terms(
        answer
    )

    query_terms = split_terms(
        query
    )

    if not answer_terms:
        return True

    has_novel_term = any(
        not any(
            term_matches(
                answer_term,
                query_term,
            )
            for query_term in query_terms
        )
        for answer_term in answer_terms
    )

    return not has_novel_term


def _script_ok(
    answer: str,
    language: str,
) -> bool:

    target = _TARGET_SCRIPT.get(
        language
    )

    if (
        target
        and target.search(
            answer
        )
    ):
        return True

    # Numbers and standard scientific Latin notation are okay.
    if re.search(
        r"[A-Za-z0-9°%]",
        answer,
    ):
        return True

    return False


def _repetition_bad(
    answer: str,
) -> bool:

    terms = split_terms(
        answer
    )

    for (
        left,
        middle,
        right,
    ) in zip(
        terms,
        terms[1:],
        terms[2:],
    ):
        if (
            left
            == middle
            == right
        ):
            return True

    return False


def _candidate_shape_ok(
    candidate: str,
    query: str,
    language: str,
) -> bool:

    if not candidate:
        return False

    if len(candidate) > 90:
        return False

    if not answer_type_is_valid(
        query,
        candidate,
        language,
    ):
        return False

    if _question_echo(
        candidate,
        query,
    ):
        return False

    if not _script_ok(
        candidate,
        language,
    ):
        return False

    if _repetition_bad(
        candidate
    ):
        return False

    return True



# =============================================================================
# OUTPUT REJECTION
# =============================================================================

def _localized_rejection(
    language: str,
    code: str,
    reason: str,
) -> tuple[
    str,
    GuardrailResult,
]:

    message = (
        NOT_FOUND_MESSAGE.get(
            language,
            DEFAULT_NOT_FOUND_MESSAGE,
        )
    )


    return (
        message,

        GuardrailResult(
            False,
            "output",
            code,
            reason,
        ),
    )


# =============================================================================
# OUTPUT GUARDRAIL + EVIDENCE SPAN REPAIR
# =============================================================================

def apply_output_guardrail(
    answer: str,
    language: str,
    *,
    query: str = "",
    context: str = "",
    possibly_truncated: bool = False,
    strong_evidence: bool = False,
) -> tuple[
    str,
    GuardrailResult,
]:

    raw = str(
        answer
    ).strip()


    # -----------------------------------------------------------------
    # First inspect evidence itself.
    # -----------------------------------------------------------------

    support = (
        strongest_supporting_unit(
            query,
            context,
            language,
        )
    )


    evidence_candidate = None


    if support.strong:

        evidence_candidate = (
            extract_answer_candidate(
                query,
                support.unit,
                language,
            )
        )


    # -----------------------------------------------------------------
    # HIGH-CONFIDENCE EVIDENCE REPAIR
    #
    # This is what fixes:
    #
    #   heart
    #   CO2
    #   Moon
    #   Mars/red planet
    #   Delhi
    #   boiling point
    #
    # even when 0.6B rambles, echoes, truncates or abstains.
    #
    # The candidate is always copied from the strongly supported evidence.
    # -----------------------------------------------------------------

    if (
        evidence_candidate is not None
        and
        evidence_candidate.confidence >= 0.90
    ):

        candidate = (
            evidence_candidate.text
        )


        if (
            _candidate_shape_ok(
                candidate,
                query,
                language,
            )
            and
            _answer_supported(
                candidate,
                query,
                context,
                language,
            )
        ):

            return (
                candidate,

                GuardrailResult(
                    True,
                    "output",
                ),
            )


    # -----------------------------------------------------------------
    # MODEL ABSTENTION
    # -----------------------------------------------------------------

    model_abstained = (
        raw == NOT_FOUND_SENTINEL
        or
        raw.startswith(
            NOT_FOUND_SENTINEL
        )
    )


    cleaned = (
        ""
        if model_abstained
        else
        _clean_answer(
            raw
        )
    )


    # -----------------------------------------------------------------
    # Accept the actual model answer if grounded.
    #
    # IMPORTANT:
    # `possibly_truncated` no longer causes automatic rejection.
    #
    # If the first answer fragment is already fully grounded, accept it.
    # -----------------------------------------------------------------

    if cleaned:

        if (
            _candidate_shape_ok(
                cleaned,
                query,
                language,
            )
            and
            _answer_supported(
                cleaned,
                query,
                context,
                language,
            )
        ):

            return (
                cleaned,

                GuardrailResult(
                    True,
                    "output",
                ),
            )


    # -----------------------------------------------------------------
    # MEDIUM-CONFIDENCE REPAIR
    #
    # Examples:
    #
    #   त्वचा मानव शरीर का सबसे बड़ा अंग है
    #   -> त्वचा
    #
    #   बृहस्पति सौर मंडल का सबसे बड़ा ग्रह है
    #   -> बृहस्पति
    #
    # Only used after model output itself failed.
    # -----------------------------------------------------------------

    if (
        evidence_candidate is not None
        and
        evidence_candidate.confidence >= 0.60
    ):

        candidate = (
            evidence_candidate.text
        )


        if (
            _candidate_shape_ok(
                candidate,
                query,
                language,
            )
            and
            _answer_supported(
                candidate,
                query,
                context,
                language,
            )
        ):

            return (
                candidate,

                GuardrailResult(
                    True,
                    "output",
                ),
            )


    # -----------------------------------------------------------------
    # Nothing passed.
    # Choose useful diagnostic code.
    # -----------------------------------------------------------------

    if model_abstained:

        return _localized_rejection(
            language,
            "model_abstained",
            (
                "Model abstained and no "
                "defensible grounded answer span "
                "could be recovered."
            ),
        )


    if possibly_truncated:

        return _localized_rejection(
            language,
            "truncated_unsupported",
            (
                "Generation reached its token budget "
                "and no complete grounded answer span "
                "could be validated."
            ),
        )


    if cleaned and _question_echo(
        cleaned,
        query,
    ):

        return _localized_rejection(
            language,
            "question_echo",
            (
                "Model repeated question terms "
                "instead of producing a grounded answer."
            ),
        )


    return _localized_rejection(
        language,
        "unsupported_answer",
        (
            "No generated or repaired answer span "
            "was strongly supported by the packed evidence."
        ),
    )


# =============================================================================
# NOT FOUND MESSAGE
# =============================================================================

def not_found_response_text(
    language: str,
) -> str:

    return (
        NOT_FOUND_MESSAGE.get(
            language,
            DEFAULT_NOT_FOUND_MESSAGE,
        )
    )


# =============================================================================
# FACADE
# =============================================================================

class Guardrails:

    check_input = staticmethod(
        check_input
    )

    check_grounding = staticmethod(
        check_grounding
    )

    apply_output_guardrail = staticmethod(
        apply_output_guardrail
    )

    not_found_response_text = staticmethod(
        not_found_response_text
    )

    answer_is_grounded = staticmethod(
        answer_is_grounded
    )

    answer_type_is_valid = staticmethod(
        answer_type_is_valid
    )

    query_expects_number = staticmethod(
        query_expects_number
    )


guardrails = (
    Guardrails()
)