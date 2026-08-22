"""
Cheap deterministic evidence-quality utilities.

IMPORTANT:
- Does NOT modify dense retrieval.
- Does NOT modify BM25 retrieval.
- Does NOT modify weighted RRF.
- Does NOT invoke another model.

Its job is only to:
1. rank already-retrieved evidence for prompt packing,
2. isolate the best sentence/list item,
3. identify when evidence is strong enough that a tiny generator
   should be told to extract rather than abstain.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


# =============================================================================
# QUESTION / RELATION WORDS
# =============================================================================

# Only low-information interrogatives / generic relation words belong here.
#
# IMPORTANT:
# Semantic modifiers such as:
#   Tamil: ஒரே, மிக, தேசிய
#   Hindi: सबसे, एकमात्र, राष्ट्रीय
#
# MUST NOT be removed. They can change the factual answer.
QUESTION_WORDS = {
    "ta": {
        "எது",
        "என்ன",
        "எந்த",
        "எத்தனை",
        "யார்",
        "எங்கே",
        "எப்படி",
        "உள்ளது",
        "அமைந்துள்ளது",
        "ஆகும்",
        "என்பது",
        "செய்யும்",
        "செய்கிறது",
        "செய்கிற",
        "பயன்படுத்தும்",
        "பயன்படுத்துகிறது",
        "கொண்டுள்ளது",
    },

    "hi": {
        "क्या",
        "कौन",
        "कौनसा",
        "कौनसी",
        "कितना",
        "कितने",
        "कितनी",
        "कहाँ",
        "कैसे",
        "है",
        "हैं",
        "होता",
        "होती",
        "होते",
        "करता",
        "करती",
        "करते",
        "करने",
        "स्थित",
        "इस्तेमाल",
        "उपयोग",
    },
}


# =============================================================================
# HARD QUERY CONSTRAINTS
# =============================================================================

# These are not domain answers.
# They are generic linguistic/unit constraints.
#
# If a user explicitly asks for Celsius, an evidence sentence containing
# only Fahrenheit must NOT be classified as "strong support".
#
# Similarly:
#   national bird != merely a bird
#   only satellite != merely a satellite
HARD_CONSTRAINT_GROUPS = {
    "ta": (
        (
            "செல்சியஸ்",
            "celsius",
            "°c",
        ),
        (
            "ஃபாரன்ஹீட்",
            "பாரன்ஹீட்",
            "fahrenheit",
            "°f",
        ),
        (
            "தேசிய",
        ),
        (
            "ஒரே",
            "மட்டுமே",
        ),
        (
            "மிக",
            "மிகவும்",
        ),
    ),

    "hi": (
        (
            "सेल्सियस",
            "celsius",
            "°c",
        ),
        (
            "फ़ारेनहाइट",
            "फारेनहाइट",
            "fahrenheit",
            "°f",
        ),
        (
            "राष्ट्रीय",
        ),
        (
            "एकमात्र",
            "केवल",
            "सिर्फ",
        ),
        (
            "सबसे",
        ),
    ),
}


# =============================================================================
# REGEX
# =============================================================================

_LIST_ITEM_RE = re.compile(
    r"^\s*(?:\d{1,2}|[A-Da-d])[\.\)]\s+"
)

_LIST_BOUNDARY_RE = re.compile(
    r"\s+(?=(?:\d{1,2}|[A-Da-d])[\.\)]\s+)"
)

_NORMAL_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[?!।])\s+|(?<=\.)\s+"
)


# =============================================================================
# TYPES
# =============================================================================

@dataclass(frozen=True, slots=True)
class SupportSignal:
    strong: bool
    score: float
    coverage: float
    matched_terms: int
    total_terms: int
    sentence: str
    list_item: bool


# =============================================================================
# TOKENIZATION
# =============================================================================

def split_terms(
    text: str,
) -> list[str]:
    """
    Unicode-aware tokenizer suitable for Tamil / Hindi.

    Keeps letters, combining marks and digits together.
    """

    terms: list[str] = []
    current: list[str] = []

    for char in str(text).casefold():

        category = unicodedata.category(
            char
        )

        if (
            category
            and
            category[0] in {
                "L",
                "M",
                "N",
            }
        ):
            current.append(char)

        elif current:

            terms.append(
                "".join(current)
            )

            current = []

    if current:

        terms.append(
            "".join(current)
        )

    return terms


def informative_query_terms(
    query: str,
    language: str,
) -> list[str]:
    """
    Remove only low-information question/relation words.
    """

    stopwords = QUESTION_WORDS.get(
        language,
        set(),
    )

    return [
        term
        for term in split_terms(query)
        if (
            len(term) >= 2
            and
            term not in stopwords
        )
    ]


# =============================================================================
# FUZZY MORPHOLOGICAL MATCHING
# =============================================================================

def term_matches(
    left: str,
    right: str,
) -> bool:
    """
    Very cheap morphology-tolerant match.

    Exact match first.

    For words >=4 chars, allow containment:
        பூமி      ↔ பூமியின்
        மனித     ↔ மனிதர்கள்

    This is intentionally conservative.
    """

    left = str(left).casefold()
    right = str(right).casefold()

    if left == right:
        return True

    if (
        min(
            len(left),
            len(right),
        )
        >= 4
    ):
        return (
            left in right
            or
            right in left
        )

    return False


def query_overlap_stats(
    query: str,
    text: str,
    language: str,
) -> tuple[int, int, float]:
    """
    Return:
        matched informative query terms,
        total informative query terms,
        coverage ratio
    """

    query_terms = informative_query_terms(
        query,
        language,
    )

    if not query_terms:
        return 0, 0, 0.0

    text_terms = split_terms(
        text
    )

    matched = 0

    for query_term in query_terms:

        if any(
            term_matches(
                query_term,
                text_term,
            )
            for text_term
            in text_terms
        ):
            matched += 1

    coverage = (
        matched
        /
        len(query_terms)
    )

    return (
        matched,
        len(query_terms),
        coverage,
    )


# =============================================================================
# HARD CONSTRAINT CHECK
# =============================================================================

def hard_constraints_satisfied(
    query: str,
    text: str,
    language: str,
) -> bool:
    """
    Ensure explicit high-value constraints in the question are not lost.

    Examples:
        Question asks Celsius
        Evidence only says Fahrenheit
        -> False

        Question asks national bird
        Evidence merely describes a bird
        -> False

        Question asks ONLY natural satellite
        Evidence merely mentions a planet
        -> False
    """

    query_cf = str(
        query
    ).casefold()

    text_cf = str(
        text
    ).casefold()

    groups = HARD_CONSTRAINT_GROUPS.get(
        language,
        (),
    )

    for group in groups:

        query_mentions_group = any(
            marker.casefold()
            in query_cf
            for marker
            in group
        )

        if not query_mentions_group:
            continue

        text_mentions_group = any(
            marker.casefold()
            in text_cf
            for marker
            in group
        )

        if not text_mentions_group:
            return False

    # Explicit numeric conditions in the question should also appear
    # in the supporting sentence.
    query_digits = set(
        re.findall(
            r"\d+",
            query_cf,
        )
    )

    if query_digits:

        text_digits = set(
            re.findall(
                r"\d+",
                text_cf,
            )
        )

        if not query_digits.issubset(
            text_digits
        ):
            return False

    return True


# =============================================================================
# PASSAGE RELEVANCE
# =============================================================================

def evidence_relevance_score(
    query: str,
    text: str,
    language: str,
) -> float:
    """
    Cheap informative query coverage from 0..1.
    """

    _, _, coverage = (
        query_overlap_stats(
            query,
            text,
            language,
        )
    )

    return coverage


# =============================================================================
# SENTENCE / LIST SPLITTING
# =============================================================================

def is_list_item(
    text: str,
) -> bool:
    return bool(
        _LIST_ITEM_RE.match(
            str(text)
        )
    )


def split_sentences(
    text: str,
) -> list[str]:
    """
    Sentence splitter designed for:
        Tamil
        Hindi
        Latin punctuation
        numbered-list passages

    Critical difference from the previous version:
    numbered items are isolated as independent evidence units instead
    of leaving:

        1. Mercury
        2. Venus
        3. Earth
        4. ...

    inside one giant sentence.
    """

    raw = str(
        text
    ).replace(
        "\r",
        "\n",
    ).strip()

    if not raw:
        return []

    # Put numbered / A-D list items on their own logical line.
    raw = _LIST_BOUNDARY_RE.sub(
        "\n",
        raw,
    )

    sentences: list[str] = []

    for line in re.split(
        r"\n+",
        raw,
    ):

        line = " ".join(
            line.split()
        ).strip()

        if not line:
            continue

        # Keep a complete numbered list item as one unit.
        if is_list_item(
            line
        ):
            sentences.append(
                line
            )
            continue

        parts = (
            _NORMAL_SENTENCE_SPLIT_RE
            .split(
                line
            )
        )

        for part in parts:

            part = " ".join(
                part.split()
            ).strip()

            if part:
                sentences.append(
                    part
                )

    if sentences:
        return sentences

    normalized = " ".join(
        raw.split()
    )

    return (
        [normalized]
        if normalized
        else []
    )


# =============================================================================
# SENTENCE SCORING
# =============================================================================

def score_sentence(
    query: str,
    sentence: str,
    language: str,
) -> float:
    """
    Rank evidence sentences for a tiny extractive generator.

    Rewards:
        informative query coverage
        multiple informative terms

    Penalizes:
        numbered distractor items
        violation of explicit hard constraints
    """

    (
        matched,
        total,
        coverage,
    ) = query_overlap_stats(
        query,
        sentence,
        language,
    )

    if total == 0:
        return 0.0

    if matched == 0:
        return 0.0

    score = (
        coverage
        *
        10.0
    )

    # Reward evidence where multiple parts of the question
    # co-occur inside one sentence.
    score += (
        min(
            matched,
            3,
        )
        *
        0.40
    )

    if not hard_constraints_satisfied(
        query,
        sentence,
        language,
    ):
        score -= 4.0

    if is_list_item(
        sentence
    ):
        score -= 0.75

    # Extra penalty if badly formatted text still contains many items.
    numbered_items = len(
        re.findall(
            r"(?:^|\s)\d{1,2}[\.\)]\s+",
            sentence,
        )
    )

    if numbered_items >= 2:
        score -= (
            1.5
            *
            (
                numbered_items
                -
                1
            )
        )

    return max(
        score,
        0.0,
    )


def evidence_pack_score(
    query: str,
    text: str,
    language: str,
) -> float:
    """
    Score an already-retrieved child for context packing.

    The best individual sentence matters much more than broad
    passage-level overlap.
    """

    sentences = split_sentences(
        text
    )

    if not sentences:
        return 0.0

    best_sentence_score = max(
        score_sentence(
            query,
            sentence,
            language,
        )
        for sentence
        in sentences
    )

    passage_coverage = (
        evidence_relevance_score(
            query,
            text,
            language,
        )
    )

    return (
        best_sentence_score
        +
        1.5
        *
        passage_coverage
    )


# =============================================================================
# STRONG SUPPORT DETECTION
# =============================================================================

def strongest_supporting_sentence(
    query: str,
    text: str,
    language: str,
) -> SupportSignal:
    """
    Find the best sentence and decide whether it is strong enough
    to switch the generator from:

        "answer or NOT_FOUND"

    into:

        "the evidence contains the answer; extract it"

    This is deliberately conservative.

    Strong support requires:
    - query terms co-occurring in ONE sentence,
    - hard constraints satisfied,
    - sufficient informative overlap,
    - stronger requirements for numbered-list items.
    """

    sentences = split_sentences(
        text
    )

    if not sentences:

        return SupportSignal(
            strong=False,
            score=0.0,
            coverage=0.0,
            matched_terms=0,
            total_terms=0,
            sentence="",
            list_item=False,
        )

    best_signal: SupportSignal | None = None

    for sentence in sentences:

        (
            matched,
            total,
            coverage,
        ) = query_overlap_stats(
            query,
            sentence,
            language,
        )

        score = score_sentence(
            query,
            sentence,
            language,
        )

        list_item = is_list_item(
            sentence
        )

        constraints_ok = (
            hard_constraints_satisfied(
                query,
                sentence,
                language,
            )
        )

        strong = False

        if (
            total > 0
            and
            constraints_ok
        ):

            # Very short factual questions:
            # require all informative terms.
            if total <= 2:

                strong = (
                    matched
                    ==
                    total
                    and
                    coverage >= 0.99
                )

            else:

                strong = (
                    matched >= 2
                    and
                    coverage >= 0.50
                )

            # A list item is especially dangerous for a 0.6B model.
            # Only treat it as strong if almost the entire question
            # aligns with that same item.
            if (
                list_item
                and
                coverage < 0.80
            ):
                strong = False

        signal = SupportSignal(
            strong=strong,
            score=score,
            coverage=coverage,
            matched_terms=matched,
            total_terms=total,
            sentence=sentence,
            list_item=list_item,
        )

        if (
            best_signal is None
            or
            (
                signal.strong,
                signal.score,
                signal.coverage,
                signal.matched_terms,
            )
            >
            (
                best_signal.strong,
                best_signal.score,
                best_signal.coverage,
                best_signal.matched_terms,
            )
        ):
            best_signal = signal

    assert best_signal is not None

    return best_signal