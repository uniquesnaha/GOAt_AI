"""
Deterministic evidence quality, support validation and answer-span extraction.

This module NEVER changes:
- dense retrieval
- BM25 retrieval
- RRF fusion
- fused Top-20 ranking

It operates only on evidence that was already retrieved.

Goals:
1. Reject wrong-subject evidence.
   Example:
       Q: India's national bird?
       E: Kiwi is New Zealand's national bird.
       -> NOT strong evidence.

2. Reject contradictions.
   Example:
       Q: largest planet?
       E: Saturn is the second largest planet.
       -> NOT strong evidence.

3. Detect direct support even with Tamil/Hindi morphology.

4. Split numbered lists correctly.

5. Produce a high-confidence grounded answer candidate when the
   evidence structure makes the answer explicit.

No model call is made here.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


# =============================================================================
# LOW-INFORMATION QUESTION WORDS
# =============================================================================

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
        "ஆகும்",
        "என்பது",
        "அமைந்துள்ளது",
        "அழைக்கப்படுகிறது",
        "அழைக்கப்படும்",
        "செய்யும்",
        "செய்கிறது",
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
        "स्थित",
        "कहा",
        "जाता",
        "जाती",
        "जाते",
        "करता",
        "करती",
        "करते",
        "करने",
        "उपयोग",
        "इस्तेमाल",
    },
}


# =============================================================================
# QUERY CONSTRAINT MARKERS
# =============================================================================

MANDATORY_MARKERS = {
    "ta": (
        (
            ("தேசிய",),
            ("தேசிய",),
        ),
        (
            ("தலைநகர",),
            ("தலைநகர",),
        ),
        (
            ("ஜனாதிபதி", "குடியரசுத் தலைவர்"),
            ("ஜனாதிபதி", "குடியரசுத் தலைவர்"),
        ),
        (
            ("சிவப்பு",),
            ("சிவப்பு",),
        ),
        (
            ("செல்சியஸ்", "°c"),
            ("செல்சியஸ்", "°c"),
        ),
        (
            ("ஃபாரன்ஹீட்", "பாரன்ஹீட்", "°f"),
            ("ஃபாரன்ஹீட்", "பாரன்ஹீட்", "°f"),
        ),
        (
            ("இயற்கை",),
            ("இயற்கை",),
        ),
        (
            ("துணைக்கோள்",),
            ("துணைக்கோள்",),
        ),
    ),

    "hi": (
        (
            ("राष्ट्रीय",),
            ("राष्ट्रीय",),
        ),
        (
            ("राजधानी",),
            ("राजधानी",),
        ),
        (
            ("राष्ट्रपति",),
            ("राष्ट्रपति",),
        ),
        (
            ("लाल",),
            ("लाल",),
        ),
        (
            ("सेल्सियस", "°c"),
            ("सेल्सियस", "°c"),
        ),
        (
            ("फ़ारेनहाइट", "फारेनहाइट", "°f"),
            ("फ़ारेनहाइट", "फारेनहाइट", "°f"),
        ),
        (
            ("प्राकृतिक",),
            ("प्राकृतिक",),
        ),
        (
            ("उपग्रह",),
            ("उपग्रह",),
        ),
    ),
}


ONLY_QUERY_MARKERS = {
    "ta": (
        "ஒரே",
        "மட்டுமே",
    ),

    "hi": (
        "एकमात्र",
        "केवल",
        "सिर्फ",
    ),
}


ONLY_EVIDENCE_MARKERS = {
    "ta": (
        "ஒரே",
        "மட்டுமே",
        "ஒரு",
        "ஒன்று",
        "1 ",
        "1.",
    ),

    "hi": (
        "एकमात्र",
        "केवल",
        "सिर्फ",
        "एक ",
        "एक ही",
        "1 ",
        "1.",
    ),
}


SUPERLATIVE_MARKERS = {
    "ta": (
        "மிகப்பெரிய",
        "மிக பெரிய",
        "மிக நீளமான",
        "மிக உயரமான",
        "மிக ஆழமான",
        "மிக வேகமான",
    ),

    "hi": (
        "सबसे बड़ा",
        "सबसे बड़ी",
        "सबसे बड़े",
        "सबसे लंबा",
        "सबसे लम्बा",
        "सबसे ऊंचा",
        "सबसे ऊँचा",
        "सबसे गहरा",
        "सबसे तेज",
    ),
}


ORDINAL_CONTRADICTIONS = {
    "ta": (
        "இரண்டாவது",
        "இரண்டாம்",
        "மூன்றாவது",
        "மூன்றாம்",
        "நான்காவது",
    ),

    "hi": (
        "दूसरा",
        "दूसरी",
        "दूसरे",
        "द्वितीय",
        "तीसरा",
        "तीसरी",
        "तीसरे",
        "चौथा",
        "चौथी",
    ),
}


# =============================================================================
# REGEX
# =============================================================================

_LIST_ITEM_RE = re.compile(
    r"^\s*(?:\d{1,2}|[A-Da-d])[\.\)]\s*"
)

_LIST_HEAD_RE = re.compile(
    r"^\s*(?:\d{1,2}|[A-Da-d])[\.\)]\s*"
    r"([^:;\-–—,.।]{1,60}?)\s*[\-–—:]"
)

_LIST_BOUNDARY_RE = re.compile(
    r"\s+(?=(?:\d{1,2}|[A-Da-d])[\.\)]\s+)"
)

_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[?!।])\s+|(?<=\.)\s+"
)


# =============================================================================
# DATA TYPES
# =============================================================================

@dataclass(frozen=True, slots=True)
class SupportSignal:
    strong: bool
    score: float
    coverage: float
    matched_terms: int
    total_terms: int
    unit: str
    anchors_ok: bool
    constraints_ok: bool
    contradiction: bool


@dataclass(frozen=True, slots=True)
class AnswerCandidate:
    text: str
    confidence: float
    method: str


# =============================================================================
# TOKENIZATION
# =============================================================================

def split_terms(text: str) -> list[str]:
    terms: list[str] = []
    current: list[str] = []

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


def _dedupe(items: list[str]) -> list[str]:
    result = []
    seen = set()

    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)

    return result


def informative_query_terms(
    query: str,
    language: str,
) -> list[str]:

    stopwords = QUESTION_WORDS.get(
        language,
        set(),
    )

    terms = [
        term
        for term in split_terms(query)
        if len(term) >= 2
        and term not in stopwords
    ]

    return _dedupe(terms)


# =============================================================================
# FUZZY MORPHOLOGICAL MATCHING
# =============================================================================

def term_matches(left: str, right: str) -> bool:
    left = str(left).casefold()
    right = str(right).casefold()

    if left == right:
        return True

    if min(len(left), len(right)) >= 4:
        return (
            left in right
            or right in left
        )

    return False


def _term_present(
    term: str,
    text_terms: list[str],
) -> bool:

    return any(
        term_matches(
            term,
            candidate,
        )
        for candidate in text_terms
    )


# =============================================================================
# REQUIRED SUBJECT ANCHORS
# =============================================================================

def required_anchor_terms(
    query: str,
    language: str,
) -> list[str]:
    """
    Extract high-value subject anchors from possessive question forms.

    Hindi examples:
        भारत का राष्ट्रीय पक्षी...
            -> भारत

        मानव शरीर का सबसे बड़ा अंग...
            -> मानव, शरीर

        पृथ्वी का एकमात्र प्राकृतिक उपग्रह...
            -> पृथ्वी

    Tamil examples:
        இந்தியாவின் தேசியப் பறவை...
            -> இந்தியாவின்

        பூமியின் ஒரே இயற்கை துணைக்கோள்...
            -> பூமியின்

        தமிழ்நாட்டின் தலைநகரம்...
            -> தமிழ்நாட்டின்
    """

    if language == "hi":

        match = re.match(
            r"^\s*(.{1,80}?)\s+(?:का|की|के)\b",
            str(query),
        )

        if not match:
            return []

        terms = [
            t
            for t in split_terms(
                match.group(1)
            )
            if len(t) >= 2
        ]

        return _dedupe(terms)


    if language == "ta":

        terms = split_terms(query)

        suffixes = (
            "வின்",
            "யின்",
            "த்தின்",
            "ட்டின்",
            "னின்",
        )

        for term in terms[:3]:

            if term.endswith(suffixes):
                return [term]

        return []


    return []


def anchors_satisfied(
    query: str,
    text: str,
    language: str,
) -> bool:

    anchors = required_anchor_terms(
        query,
        language,
    )

    if not anchors:
        return True

    text_terms = split_terms(text)

    return all(
        _term_present(
            anchor,
            text_terms,
        )
        for anchor in anchors
    )


# =============================================================================
# QUERY TERM COVERAGE
# =============================================================================

def query_overlap_stats(
    query: str,
    text: str,
    language: str,
) -> tuple[int, int, float]:

    query_terms = informative_query_terms(
        query,
        language,
    )

    if not query_terms:
        return 0, 0, 0.0

    text_terms = split_terms(text)

    matched = sum(
        1
        for query_term in query_terms
        if _term_present(
            query_term,
            text_terms,
        )
    )

    total = len(query_terms)

    return (
        matched,
        total,
        matched / total,
    )


# =============================================================================
# HARD CONSTRAINTS
# =============================================================================

def hard_constraints_satisfied(
    query: str,
    text: str,
    language: str,
) -> bool:

    query_cf = str(query).casefold()
    text_cf = str(text).casefold()

    for (
        query_markers,
        evidence_markers,
    ) in MANDATORY_MARKERS.get(
        language,
        (),
    ):

        requested = any(
            marker.casefold() in query_cf
            for marker in query_markers
        )

        if not requested:
            continue

        found = any(
            marker.casefold() in text_cf
            for marker in evidence_markers
        )

        if not found:
            return False


    only_requested = any(
        marker.casefold() in query_cf
        for marker in ONLY_QUERY_MARKERS.get(
            language,
            (),
        )
    )

    if only_requested:

        only_supported = any(
            marker.casefold() in text_cf
            for marker in ONLY_EVIDENCE_MARKERS.get(
                language,
                (),
            )
        )

        if not only_supported:
            return False


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


def has_query_contradiction(
    query: str,
    text: str,
    language: str,
) -> bool:

    query_cf = str(query).casefold()
    text_cf = str(text).casefold()

    asks_superlative = any(
        marker.casefold() in query_cf
        for marker in SUPERLATIVE_MARKERS.get(
            language,
            (),
        )
    )

    if asks_superlative:

        has_ordinal = any(
            marker.casefold() in text_cf
            for marker in ORDINAL_CONTRADICTIONS.get(
                language,
                (),
            )
        )

        if has_ordinal:
            return True


    return False


# =============================================================================
# SENTENCE / LIST SPLITTING
# =============================================================================

def is_list_item(text: str) -> bool:
    return bool(
        _LIST_ITEM_RE.match(
            str(text)
        )
    )


def split_sentences(text: str) -> list[str]:
    raw = (
        str(text)
        .replace("\r", "\n")
        .strip()
    )

    if not raw:
        return []

    # Split:
    # 1. Mars ...
    # 2. Venus ...
    # into independent logical units.
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

        if is_list_item(line):
            sentences.append(line)
            continue

        for part in _SENTENCE_SPLIT_RE.split(
            line
        ):

            part = " ".join(
                part.split()
            ).strip()

            if part:
                sentences.append(part)

    return sentences


def candidate_evidence_units(
    text: str,
) -> list[str]:
    """
    Individual sentences + limited adjacent pairs.

    Adjacent pairs are required for evidence such as:

        1. Earth ...
        2. It has one natural satellite, the Moon.

    The relationship can otherwise be split across list items.
    """

    sentences = split_sentences(text)

    if not sentences:
        return []

    result = list(sentences)

    continuation_markers = (
        "इसका ",
        "इसके ",
        "यह ",
        "उसका ",
        "அது ",
        "இதன் ",
        "இதற்கு ",
    )

    for idx in range(
        len(sentences) - 1
    ):

        first = sentences[idx]
        second = sentences[idx + 1]

        should_pair = (
            is_list_item(first)
            or is_list_item(second)
            or second.startswith(
                continuation_markers
            )
        )

        if not should_pair:
            continue

        combined = (
            first
            + " "
            + second
        )

        if len(combined) <= 450:
            result.append(combined)

    return result


# =============================================================================
# SUPPORT SCORING
# =============================================================================

def support_for_unit(
    query: str,
    unit: str,
    language: str,
) -> SupportSignal:

    (
        matched,
        total,
        coverage,
    ) = query_overlap_stats(
        query,
        unit,
        language,
    )

    anchors_ok = anchors_satisfied(
        query,
        unit,
        language,
    )

    constraints_ok = (
        hard_constraints_satisfied(
            query,
            unit,
            language,
        )
    )

    contradiction = (
        has_query_contradiction(
            query,
            unit,
            language,
        )
    )


    score = (
        coverage * 10.0
        +
        min(matched, 3) * 0.4
    )

    if anchors_ok:
        score += 1.0
    else:
        score -= 7.0

    if not constraints_ok:
        score -= 7.0

    if contradiction:
        score -= 10.0


    strong = False

    if (
        total > 0
        and anchors_ok
        and constraints_ok
        and not contradiction
    ):

        if total <= 2:

            strong = (
                matched == total
            )

        elif total == 3:

            strong = (
                matched >= 2
                and coverage >= 0.66
            )

        else:

            strong = (
                matched >= 3
                and coverage >= 0.55
            )


        if (
            is_list_item(unit)
            and coverage < 0.60
        ):
            strong = False


    return SupportSignal(
        strong=strong,
        score=max(score, 0.0),
        coverage=coverage,
        matched_terms=matched,
        total_terms=total,
        unit=unit,
        anchors_ok=anchors_ok,
        constraints_ok=constraints_ok,
        contradiction=contradiction,
    )


def strongest_supporting_unit(
    query: str,
    text: str,
    language: str,
) -> SupportSignal:

    units = candidate_evidence_units(
        text
    )

    if not units:

        return SupportSignal(
            strong=False,
            score=0.0,
            coverage=0.0,
            matched_terms=0,
            total_terms=0,
            unit="",
            anchors_ok=False,
            constraints_ok=False,
            contradiction=False,
        )


    signals = [
        support_for_unit(
            query,
            unit,
            language,
        )
        for unit in units
    ]


    return max(
        signals,
        key=lambda signal: (
            signal.strong,
            signal.score,
            signal.coverage,
            signal.matched_terms,
        ),
    )


def evidence_relevance_score(
    query: str,
    text: str,
    language: str,
) -> float:

    (
        _,
        _,
        coverage,
    ) = query_overlap_stats(
        query,
        text,
        language,
    )

    return coverage


def evidence_pack_score(
    query: str,
    text: str,
    language: str,
) -> float:

    signal = strongest_supporting_unit(
        query,
        text,
        language,
    )

    # Wrong subject / explicit contradiction should not win
    # context packing even if lexical overlap is high.
    if (
        not signal.anchors_ok
        or not signal.constraints_ok
        or signal.contradiction
    ):
        return 0.0

    return (
        signal.score
        +
        (
            15.0
            if signal.strong
            else 0.0
        )
    )


# =============================================================================
# GROUNDED ANSWER CANDIDATE EXTRACTION
# =============================================================================

def _clean_candidate(
    candidate: str,
    query: str,
    language: str,
) -> str | None:

    candidate = (
        str(candidate)
        .strip()
        .strip(
            " \t\r\n"
            "\"'“”‘’"
            "-–—:;,."
            "।()[]{}"
        )
    )

    candidate = _LIST_ITEM_RE.sub(
        "",
        candidate,
    ).strip()


    if not candidate:
        return None


    terms = split_terms(candidate)

    if not terms:
        return None


    if len(terms) > 6:
        return None


    query_terms = split_terms(query)

    all_query_echo = all(
        any(
            term_matches(
                answer_term,
                query_term,
            )
            for query_term in query_terms
        )
        for answer_term in terms
    )

    if all_query_echo:
        return None


    if len(candidate) > 80:
        return None


    return candidate


def _strip_query_prefix_terms(
    phrase: str,
    query: str,
) -> str:

    terms = split_terms(phrase)

    query_terms = split_terms(query)

    while terms:

        first = terms[0]

        if any(
            term_matches(
                first,
                q,
            )
            for q in query_terms
        ):

            terms.pop(0)

        else:
            break

    return " ".join(terms)


def extract_answer_candidate(
    query: str,
    unit: str,
    language: str,
) -> AnswerCandidate | None:
    """
    Extract only when the evidence syntax gives us a defensible span.

    This is NOT a knowledge lookup.

    Every candidate is copied from the supplied evidence.
    """

    if not unit:
        return None

    normalized = " ".join(
        str(unit).split()
    )


    # -----------------------------------------------------------------
    # NUMERIC + UNIT
    # -----------------------------------------------------------------

    query_cf = query.casefold()

    if (
        "செல்சியஸ்" in query_cf
        or "°c" in query_cf
    ):

        match = re.search(
            r"\b\d+(?:[.,]\d+)?\s*"
            r"(?:டிகிரி\s*)?"
            r"(?:செல்சியஸ்|°c)",
            normalized,
            re.IGNORECASE,
        )

        if match:

            candidate = _clean_candidate(
                match.group(0),
                query,
                language,
            )

            if candidate:

                return AnswerCandidate(
                    candidate,
                    1.0,
                    "numeric_unit",
                )


    if (
        "सेल्सियस" in query_cf
        or "°c" in query_cf
    ):

        match = re.search(
            r"\b\d+(?:[.,]\d+)?\s*"
            r"(?:डिग्री\s*)?"
            r"(?:सेल्सियस|°c)",
            normalized,
            re.IGNORECASE,
        )

        if match:

            candidate = _clean_candidate(
                match.group(0),
                query,
                language,
            )

            if candidate:

                return AnswerCandidate(
                    candidate,
                    1.0,
                    "numeric_unit",
                )


    # -----------------------------------------------------------------
    # HINDI HIGH-CONFIDENCE RELATION PATTERNS
    # -----------------------------------------------------------------

    if language == "hi":

        # रक्त को पंप करने वाला हृदय है
        match = re.search(
            r"पंप\s+करने\s+वाला\s+"
            r"([^\s,।.;:()]{2,40})",
            normalized,
            re.IGNORECASE,
        )

        if match:

            candidate = _clean_candidate(
                match.group(1),
                query,
                language,
            )

            if candidate:

                return AnswerCandidate(
                    candidate,
                    1.0,
                    "pump_relation",
                )


        # ... प्राकृतिक उपग्रह है, चंद्रमा
        match = re.search(
            r"उपग्रह\s+"
            r"(?:है\s*)?"
            r"[,:\-–—]?\s*"
            r"([^\s,।.;:()]{2,40})",
            normalized,
            re.IGNORECASE,
        )

        if match:

            value = match.group(1)

            if value not in {
                "है",
                "का",
                "की",
                "के",
            }:

                candidate = _clean_candidate(
                    value,
                    query,
                    language,
                )

                if candidate:

                    return AnswerCandidate(
                        candidate,
                        1.0,
                        "satellite_relation",
                    )


        # भारत की राजधानी दिल्ली में...
        match = re.search(
            r"राजधानी\s+"
            r"(?:है\s+)?"
            r"([^\s,।.;:()]{2,40})",
            normalized,
            re.IGNORECASE,
        )

        if match:

            candidate = _clean_candidate(
                match.group(1),
                query,
                language,
            )

            if candidate:

                return AnswerCandidate(
                    candidate,
                    0.95,
                    "capital_relation",
                )


        # X गैस
        if "गैस" in query_cf:

            match = re.search(
                r"((?:\S+\s+){0,3}\S+)"
                r"\s+गैस(?:\s|$)",
                normalized,
                re.IGNORECASE,
            )

            if match:

                phrase = _strip_query_prefix_terms(
                    match.group(1),
                    query,
                )

                candidate = _clean_candidate(
                    phrase,
                    query,
                    language,
                )

                if candidate:

                    return AnswerCandidate(
                        candidate,
                        0.95,
                        "gas_relation",
                    )


    # -----------------------------------------------------------------
    # TAMIL HIGH-CONFIDENCE RELATION PATTERNS
    # -----------------------------------------------------------------

    if language == "ta":

        # இதயம் என்பது ...
        match = re.match(
            r"^\s*"
            r"(.{1,60}?)"
            r"\s+என்பது\b",
            normalized,
        )

        if match:

            candidate = _clean_candidate(
                match.group(1),
                query,
                language,
            )

            if candidate:

                return AnswerCandidate(
                    candidate,
                    1.0,
                    "tamil_copula",
                )


        # ... கார்பன் டை ஆக்சைடு வாயுவை ...
        if "வாயு" in query_cf:

            match = re.search(
                r"((?:\S+\s+){0,4}\S+)"
                r"\s+வாயு(?:வை|வாக|வில்|$)",
                normalized,
            )

            if match:

                phrase = _strip_query_prefix_terms(
                    match.group(1),
                    query,
                )

                terms = split_terms(
                    phrase
                )

                # Answer gases are usually compact.
                if len(terms) > 3:
                    terms = terms[-3:]

                phrase = " ".join(
                    terms
                )

                candidate = _clean_candidate(
                    phrase,
                    query,
                    language,
                )

                if candidate:

                    return AnswerCandidate(
                        candidate,
                        1.0,
                        "gas_relation",
                    )


        match = re.search(
            r"தலைநகர(?:ம்|மாக)\s+"
            r"([^\s,.;:()]{2,50})",
            normalized,
        )

        if match:

            candidate = _clean_candidate(
                match.group(1),
                query,
                language,
            )

            if candidate:

                return AnswerCandidate(
                    candidate,
                    0.95,
                    "capital_relation",
                )


        match = re.search(
            r"துணைக்கோள்"
            r"(?:\s+ஆகும்|\s+உள்ளது|\s+என்பது)?"
            r"\s*[,:\-–—]?\s*"
            r"([^\s,.;:()]{2,50})",
            normalized,
        )

        if match:

            candidate = _clean_candidate(
                match.group(1),
                query,
                language,
            )

            if candidate:

                return AnswerCandidate(
                    candidate,
                    0.95,
                    "satellite_relation",
                )


    # -----------------------------------------------------------------
    # NUMBERED LIST:
    #
    # 3. मंगल - ...
    # 1. செவ்வாய் - ...
    # -----------------------------------------------------------------

    match = _LIST_HEAD_RE.match(
        normalized
    )

    if match:

        candidate = _clean_candidate(
            match.group(1),
            query,
            language,
        )

        if candidate:

            return AnswerCandidate(
                candidate,
                0.95,
                "list_head",
            )


    # -----------------------------------------------------------------
    # GENERIC LEADING ANSWER
    #
    # त्वचा मानव शरीर का सबसे बड़ा अंग है
    # बृहस्पति सौर मंडल का सबसे बड़ा ग्रह है
    # प्रशांत महासागर विश्व का सबसे बड़ा महासागर है
    #
    # Only MEDIUM confidence. Used mainly to repair abstention/truncation.
    # -----------------------------------------------------------------

    stripped = _LIST_ITEM_RE.sub(
        "",
        normalized,
    ).strip()

    words = stripped.split()

    informative = informative_query_terms(
        query,
        language,
    )

    first_query_word_index = None

    for idx, word in enumerate(words):

        word_terms = split_terms(
            word
        )

        if not word_terms:
            continue

        if any(
            any(
                term_matches(
                    word_term,
                    query_term,
                )
                for query_term in informative
            )
            for word_term in word_terms
        ):

            first_query_word_index = idx
            break


    if (
        first_query_word_index is not None
        and
        1 <= first_query_word_index <= 4
    ):

        phrase = " ".join(
            words[
                :first_query_word_index
            ]
        )

        candidate = _clean_candidate(
            phrase,
            query,
            language,
        )

        if candidate:

            return AnswerCandidate(
                candidate,
                0.65,
                "leading_entity",
            )


    return None