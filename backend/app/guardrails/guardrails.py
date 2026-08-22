"""
Deterministic guardrails around GOAt AI RAG.

Objectives:
- no extra LLM call,
- preserve latency,
- allow correct grounded short answers,
- reject distractor-list hallucinations,
- reject answers whose requested unit/modifier is unsupported,
- preserve strict NOT_FOUND behavior when evidence is genuinely weak.
"""

from __future__ import annotations

import re
import unicodedata

from dataclasses import dataclass
from typing import Optional

from app.rag.evidence_quality import (
    hard_constraints_satisfied,
    informative_query_terms,
    is_list_item,
    query_overlap_stats,
    split_sentences,
    split_terms,
    term_matches,
)


# =============================================================================
# BASIC CONFIG
# =============================================================================

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


# =============================================================================
# REGEX
# =============================================================================

_CITATION_RE = re.compile(
    r"\[\s*\d+\s*\]"
)

_CITATION_ONLY_RE = re.compile(
    r"^(?:\s*\[\s*\d+\s*\]\s*)+$"
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


_NUMERIC_ANSWER_RE = re.compile(
    r"^[\s\d.,:+\-−°%/]+"
    r"(?:f|c|°c|°f|km|kg|m|cm|mm)?$",
    re.IGNORECASE,
)


_MCQ_PREFIX_RE = re.compile(
    r"^[A-Da-d][.)]\s*"
)


_ANSWER_PREFIX_RE = re.compile(
    r"^(?:Answer|Ans|उत्तर|பதில்)\s*[:\-–]\s*",
    re.IGNORECASE,
)


# =============================================================================
# MINIMAL SAFETY NET
# =============================================================================

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


# =============================================================================
# RESULT TYPE
# =============================================================================

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

    stage = (
        "input"
    )


    if (
        not query
        or
        not query.strip()
    ):

        return GuardrailResult(
            False,
            stage,
            "empty_query",
            "Query is empty.",
        )


    if (
        len(query)
        >
        MAX_QUERY_CHARS
    ):

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


    for pattern in (
        _UNSAFE_PATTERNS
    ):

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

    stage = (
        "grounding"
    )


    if (
        not parents
        or
        not context.strip()
    ):

        return GuardrailResult(
            False,
            stage,
            "no_grounded_context",
            (
                "Retrieval returned no usable "
                "context for this query."
            ),
        )


    return GuardrailResult(
        True,
        stage,
    )


# =============================================================================
# CLEANING
# =============================================================================

def _terms(
    text: str,
) -> list[str]:

    terms = []
    current = []

    for char in str(
        text
    ).casefold():

        category = (
            unicodedata.category(
                char
            )
        )

        if (
            category
            and
            category[0]
            in {
                "L",
                "M",
                "N",
            }
        ):

            current.append(
                char
            )

        elif current:

            terms.append(
                "".join(
                    current
                )
            )

            current = []

    if current:

        terms.append(
            "".join(
                current
            )
        )

    return terms


def _clean_answer(
    answer: str,
) -> str:

    cleaned = str(
        answer
    ).strip()

    cleaned = (
        cleaned.replace(
            "**",
            "",
        )
    )

    cleaned = (
        _CITATION_RE.sub(
            "",
            cleaned,
        )
    )

    cleaned = (
        _MCQ_PREFIX_RE.sub(
            "",
            cleaned,
        )
    )

    cleaned = (
        _ANSWER_PREFIX_RE.sub(
            "",
            cleaned,
        )
    )

    cleaned = (
        cleaned.replace(
            "\ufffd",
            "",
        )
    )

    cleaned = " ".join(
        cleaned.split()
    )

    return cleaned.strip(
        " -–—:;,.|"
    )


# =============================================================================
# ANSWER → SENTENCE MATCHING
# =============================================================================

def _answer_is_inside_sentence(
    answer: str,
    sentence: str,
) -> bool:
    """
    Require substantially stronger containment than the old
    "ANY answer term appears" rule.

    Multi-word answer:
        all meaningful answer terms must appear morphologically.

    Numeric answer:
        all answer numbers must be present.
    """

    answer_cf = (
        answer.casefold()
    )

    sentence_cf = (
        sentence.casefold()
    )


    # Best case: exact phrase.
    if (
        answer_cf
        and
        answer_cf
        in
        sentence_cf
    ):
        return True


    answer_digits = set(
        re.findall(
            r"\d+",
            answer_cf,
        )
    )

    sentence_digits = set(
        re.findall(
            r"\d+",
            sentence_cf,
        )
    )


    if answer_digits:

        if not answer_digits.issubset(
            sentence_digits
        ):
            return False


    answer_terms = [
        term
        for term
        in split_terms(
            answer
        )
        if len(term) >= 2
    ]


    # Pure numeric answer.
    if (
        answer_digits
        and
        not answer_terms
    ):
        return True


    if not answer_terms:
        return False


    sentence_terms = (
        split_terms(
            sentence
        )
    )


    # Require ALL meaningful answer terms, not merely one.
    for answer_term in (
        answer_terms
    ):

        if not any(
            term_matches(
                answer_term,
                sentence_term,
            )

            for sentence_term
            in sentence_terms
        ):

            return False


    return True


# =============================================================================
# CONTEXT SUPPORT VALIDATION
# =============================================================================

def _supported_by_context(
    answer: str,
    context: str,
    query: str = "",
    language: str = "ta",
) -> bool:
    """
    Grounding validation at sentence/list-item level.

    A generated answer is accepted only when one evidence unit:
    1. actually contains the answer,
    2. substantially aligns with the question,
    3. satisfies explicit question constraints,
    4. is not merely an unrelated numbered-list distractor.
    """

    sentences = (
        split_sentences(
            context
        )
    )

    if not sentences:
        return False


    query_terms = (
        informative_query_terms(
            query,
            language,
        )
    )


    for sentence in sentences:


        # ---------------------------------------------------------
        # The answer itself must actually occur here.
        # ---------------------------------------------------------

        if not _answer_is_inside_sentence(
            answer,
            sentence,
        ):
            continue


        # ---------------------------------------------------------
        # Explicit unit / national / only / superlative constraints.
        # ---------------------------------------------------------

        if not hard_constraints_satisfied(
            query,
            sentence,
            language,
        ):
            continue


        (
            matched,
            total,
            coverage,
        ) = (
            query_overlap_stats(
                query,
                sentence,
                language,
            )
        )


        # ---------------------------------------------------------
        # No informative query terms:
        # answer containment is enough.
        # ---------------------------------------------------------

        if total == 0:
            return True


        # ---------------------------------------------------------
        # Very short query:
        # require every meaningful concept.
        # ---------------------------------------------------------

        if total <= 2:

            query_supported = (
                matched
                ==
                total
            )


        else:

            # Multiple terms:
            # require at least two informative matches and
            # meaningful overall coverage.
            query_supported = (
                matched >= 2
                and
                coverage >= 0.40
            )


        if not query_supported:
            continue


        # ---------------------------------------------------------
        # Numbered-list distractor protection.
        #
        # Example:
        #
        # 1. Mercury ...
        # 2. Venus ...
        # 3. Earth ...
        #
        # "Mercury" cannot pass just because it appears somewhere
        # in the retrieved list.
        # ---------------------------------------------------------

        if (
            is_list_item(
                sentence
            )
            and
            coverage < 0.67
        ):
            continue


        return True


    return False


# =============================================================================
# REJECTION
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
# OUTPUT
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
    """
    Deterministic output validation.

    Adds effectively negligible latency.
    """

    stage = (
        "output"
    )


    stripped = str(
        answer
    ).strip()


    # ---------------------------------------------------------
    # MODEL ABSTENTION
    # ---------------------------------------------------------

    if (
        stripped
        ==
        NOT_FOUND_SENTINEL
        or
        stripped.startswith(
            NOT_FOUND_SENTINEL
        )
    ):

        code = (
            "model_abstained_on_strong_evidence"
            if strong_evidence
            else "model_abstained"
        )

        reason = (
            "Model reported NOT_FOUND despite strong packed evidence."
            if strong_evidence
            else "Model reported NOT_FOUND."
        )

        return _localized_rejection(
            language,
            code,
            reason,
        )


    # ---------------------------------------------------------
    # EMPTY
    # ---------------------------------------------------------

    if not stripped:

        return _localized_rejection(
            language,
            "empty_answer",
            "Model returned an empty answer.",
        )


    # ---------------------------------------------------------
    # CITATION ONLY
    # ---------------------------------------------------------

    if (
        _CITATION_ONLY_RE.fullmatch(
            stripped
        )
    ):

        return _localized_rejection(
            language,
            "citation_only",
            "Model returned a citation without an answer.",
        )


    # ---------------------------------------------------------
    # TOKEN LIMIT
    # ---------------------------------------------------------

    if possibly_truncated:

        return _localized_rejection(
            language,
            "truncated_answer",
            "Model output hit the token limit before completing.",
        )


    cleaned = (
        _clean_answer(
            stripped
        )
    )


    if not cleaned:

        return _localized_rejection(
            language,
            "empty_answer",
            "Model returned an empty answer after cleaning.",
        )


    # ---------------------------------------------------------
    # LONG CONTEXT ECHO
    # ---------------------------------------------------------

    if (
        context
        and
        len(cleaned)
        >= 50
        and
        cleaned
        in
        context[
            :160
        ]
    ):

        return _localized_rejection(
            language,
            "context_echo",
            "Model copied a long context fragment instead of extracting an answer.",
        )


    # ---------------------------------------------------------
    # REPETITION
    # ---------------------------------------------------------

    answer_terms = (
        _terms(
            cleaned
        )
    )


    for (
        left,
        middle,
        right,
    ) in zip(
        answer_terms,
        answer_terms[
            1:
        ],
        answer_terms[
            2:
        ],
    ):

        if (
            left
            ==
            middle
            ==
            right
        ):

            return _localized_rejection(
                language,
                "repeated_answer",
                "Model output contains excessive repetition.",
            )


    # ---------------------------------------------------------
    # QUESTION ECHO
    # ---------------------------------------------------------

    query_terms = set(
        _terms(
            query
        )
    )


    meaningful_new = [
        term
        for term
        in answer_terms

        if (
            term
            not in
            query_terms
            and
            len(term)
            >= 4
        )
    ]


    if (
        answer_terms
        and
        not meaningful_new
        and
        set(
            answer_terms
        ).issubset(
            query_terms
        )
    ):

        return _localized_rejection(
            language,
            "question_echo",
            "Model repeated the question instead of answering it.",
        )


    # ---------------------------------------------------------
    # SCRIPT
    # ---------------------------------------------------------

    target_script = (
        _TARGET_SCRIPT.get(
            language
        )
    )


    has_target_script = bool(
        target_script
        and
        target_script.search(
            cleaned
        )
    )


    has_latin_or_numbers = bool(
        re.search(
            r"[a-zA-Z0-9]",
            cleaned,
        )
    )


    if not (
        has_target_script
        or
        has_latin_or_numbers
        or
        _NUMERIC_ANSWER_RE
        .fullmatch(
            cleaned
        )
    ):

        return _localized_rejection(
            language,
            "wrong_language",
            "Model answered in an unsupported script.",
        )


    # ---------------------------------------------------------
    # ACTUAL GROUNDING
    # ---------------------------------------------------------

    if (
        context
        and
        not _supported_by_context(
            cleaned,
            context,
            query=
                query,
            language=
                language,
        )
    ):

        return _localized_rejection(
            language,
            "unsupported_answer",
            (
                "Generated answer is not supported by "
                "a query-aligned evidence sentence."
            ),
        )


    return (
        cleaned,

        GuardrailResult(
            True,
            stage,
        ),
    )


# =============================================================================
# USER-FACING NOT FOUND
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


guardrails = Guardrails()