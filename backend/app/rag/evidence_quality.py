"""Cheap evidence-quality scoring for context packing.

This module is intentionally dependency-free. It never changes dense, BM25,
or RRF ranking; it only selects among already-retrieved evidence.
"""

from __future__ import annotations

import unicodedata


QUESTION_WORDS = {
    "ta": {
        # interrogatives
        "எது", "என்ன", "எந்த", "எத்தனை", "யார்", "எங்கே", "எப்படி",
        "உள்ளது", "அமைந்துள்ளது", "ஆகும்",
        # superlatives / quantifiers (no retrieval signal when in query)
        "மிக", "மிகவும்", "ஒரே", "முதல்",
    },
    "hi": {
        # interrogatives
        "क्या", "कौन", "कौनसा", "कौनसी", "कितना", "कितने", "कितनी",
        "कहाँ", "कैसे", "है", "हैं", "होता", "होती", "स्थित",
        # superlatives / quantifiers (no retrieval signal when in query)
        "सबसे", "सबसेबड़ा", "सबसेछोटा", "सबसेलंबा", "सबसेतेज",
        "सबसेबड़ी", "सबसेछोटी", "सबसेलम्बी",
    },
}



def split_terms(text: str) -> list[str]:
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


def evidence_relevance_score(
    query: str,
    text: str,
    language: str,
) -> float:
    """Return informative query-term coverage for one evidence passage."""

    stopwords = QUESTION_WORDS.get(language, set())
    query_terms = [
        term
        for term in split_terms(query)
        if len(term) >= 2 and term not in stopwords
    ]

    if not query_terms:
        return 0.0

    text_terms = set(split_terms(text))

    def matches(query_term: str) -> bool:
        return any(
            query_term == text_term
            or (
                min(len(query_term), len(text_term)) >= 4
                and (
                    query_term in text_term
                    or text_term in query_term
                )
            )
            for text_term in text_terms
        )

    return sum(matches(term) for term in query_terms) / len(query_terms)
