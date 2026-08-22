"""
RAG Failure Attribution & Diagnostic Tool for GOAt AI.

Evaluates test queries and classifies any failure into the exact pipeline stage:
- CORPUS_MISS: Expected answer is not present anywhere in the Parquet corpus.
- RETRIEVAL_MISS: Expected answer exists in corpus but was not retrieved in Top-20 parents.
- CONTEXT_PACKING_MISS: Answer was in Top-20 but excluded from packed generation context.
- GENERATION_MISS: Answer was in generation context but model generated wrong answer / NOT_FOUND.
- GROUNDING_REJECTION: Model generated the correct answer but output guardrail rejected it.
- PASS: End-to-end correct grounded answer.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

# Add backend directory to sys.path
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.guardrails.guardrails import (
    _normalize_grounding_text,
    _strip_answer_suffixes,
    apply_output_guardrail,
)
from app.rag.engine import (
    CHUNKS,
    CONTEXT_CHAR_BUDGET,
    MAX_CONTEXT_PARENTS,
    PER_CHUNK_CHARS,
    FullRAG,
)


EVALUATION_SET = [
    # Tamil queries
    {
        "query": "தாவரங்கள் ஒளிச்சேர்க்கைக்கு பயன்படுத்தும் வாயு எது?",
        "language": "ta",
        "expected": "கார்பன் டை ஆக்சைடு",
    },
    {
        "query": "மனித உடலில் இரத்தத்தை பம்ப் செய்யும் உறுப்பு எது?",
        "language": "ta",
        "expected": "இதயம்",
    },
    {
        "query": "பூமியின் ஒரே இயற்கை துணைக்கோள் எது?",
        "language": "ta",
        "expected": "சந்திரன்",
    },
    {
        "query": "சூரிய குடும்பத்தில் மிகப்பெரிய கோள் எது?",
        "language": "ta",
        "expected": "வியாழன்",
    },
    {
        "query": "இந்தியாவின் தேசியப் பறவை எது?",
        "language": "ta",
        "expected": "மயில்",
    },
    {
        "query": "எந்த கிரகம் சிவப்பு கிரகம் என்று அழைக்கப்படுகிறது?",
        "language": "ta",
        "expected": "செவ்வாய்",
    },
    {
        "query": "தமிழ்நாட்டின் தலைநகரம் எது?",
        "language": "ta",
        "expected": "சென்னை",
    },
    {
        "query": "நீரின் கொதிநிலை எத்தனை டிகிரி செல்சியஸ்?",
        "language": "ta",
        "expected": "100",
    },
    {
        "query": "சிந்து சமவெளி நாகரிகத்தின் பெரிய குளியல் குளம் எங்கு கண்டுபிடிக்கப்பட்டது?",
        "language": "ta",
        "expected": "மோஹென்ஜோ",
    },
    {
        "query": "ஒளியின் வேகம் ஒரு வினாடிக்கு தோராயமாக எத்தனை கிலோமீட்டர்?",
        "language": "ta",
        "expected": "300000",
    },
    # Hindi queries
    {
        "query": "सौरमंडल का सबसे बड़ा ग्रह कौन सा है?",
        "language": "hi",
        "expected": "बृहस्पति",
    },
    {
        "query": "मानव शरीर में रक्त को पंप करने वाला अंग कौन सा है?",
        "language": "hi",
        "expected": "हृदय",
    },
    {
        "query": "मानव शरीर का सबसे बड़ा अंग कौन सा है?",
        "language": "hi",
        "expected": "त्वचा",
    },
    {
        "query": "पृथ्वी का एकमात्र प्राकृतिक उपग्रह कौन सा है?",
        "language": "hi",
        "expected": "चंद्रमा",
    },
    {
        "query": "भारत का राष्ट्रीय पक्षी कौन सा है?",
        "language": "hi",
        "expected": "मोर",
    },
    {
        "query": "सौरमंडल में किस ग्रह को लाल ग्रह कहा जाता है?",
        "language": "hi",
        "expected": "मंगल",
    },
    {
        "query": "पानी का क्वथनांक कितने डिग्री सेल्सियस होता है?",
        "language": "hi",
        "expected": "100",
    },
    {
        "query": "भारत की राजधानी क्या है?",
        "language": "hi",
        "expected": "दिल्ली",
    },
]


def text_contains_expected(expected: str, text: str) -> bool:
    if not expected or not text:
        return False
    exp_norm = _strip_answer_suffixes(expected)
    text_norm = _normalize_grounding_text(text)
    return exp_norm in text_norm


def search_corpus_for_answer(engine: FullRAG, language: str, expected: str) -> bool:
    """Check if the expected answer appears in any raw chunk text for the language."""
    bm25 = engine.bm25.get(language)
    if not bm25 or not hasattr(bm25, "chunk_texts"):
        return True  # Assume present if corpus not loaded in memory

    exp_norm = _strip_answer_suffixes(expected)
    for text in bm25.chunk_texts:
        if exp_norm in _normalize_grounding_text(text):
            return True
    return False


def classify_failure(
    *,
    expected: str,
    corpus_hit: bool,
    retrieval_hit: bool,
    context_hit: bool,
    raw_generation_hit: bool,
    guardrail_allowed: bool,
    final_answer_hit: bool,
) -> str:
    if final_answer_hit and guardrail_allowed:
        return "PASS"
    if not corpus_hit:
        return "CORPUS_MISS"
    if not retrieval_hit:
        return "RETRIEVAL_MISS"
    if not context_hit:
        return "CONTEXT_PACKING_MISS"
    if not raw_generation_hit:
        return "GENERATION_MISS"
    if not guardrail_allowed or not final_answer_hit:
        return "GROUNDING_REJECTION"
    return "UNKNOWN"


def run_diagnostics(output_csv: str | None = None) -> list[dict]:
    print("=" * 80)
    print("GOAt AI RAG Failure Attribution & Diagnostic Suite")
    print("=" * 80)

    print("Initializing FullRAG engine...")
    engine = FullRAG()
    engine.warmup()
    print("Engine ready.\n")

    results = []
    category_counts = {
        "PASS": 0,
        "CORPUS_MISS": 0,
        "RETRIEVAL_MISS": 0,
        "CONTEXT_PACKING_MISS": 0,
        "GENERATION_MISS": 0,
        "GROUNDING_REJECTION": 0,
        "UNKNOWN": 0,
    }

    for item in EVALUATION_SET:
        query = item["query"]
        lang = item["language"]
        expected = item["expected"]

        # 1. Corpus presence
        corpus_hit = search_corpus_for_answer(engine, lang, expected)

        # 2. Retrieval
        retrieval_start = time.perf_counter()
        retrieval = engine.retrieve(query, lang)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        # Check retrieval Top-20 text
        retrieved_texts = []
        for p in retrieval.get("parents", []):
            p_str = str(p)
            candidates = engine.contexts._all_parent_candidates(
                lang, p_str, retrieval.get("evidence_by_parent", {})
            )
            for c in candidates:
                retrieved_texts.append(c.get("text", ""))

        retrieval_hit = any(text_contains_expected(expected, t) for t in retrieved_texts)

        # 3. Context packing
        context, _, _, _ = engine.contexts.build(
            lang,
            query,
            retrieval["parents"],
            CONTEXT_CHAR_BUDGET,
            MAX_CONTEXT_PARENTS,
            PER_CHUNK_CHARS,
            evidence_by_parent=retrieval.get("evidence_by_parent", {}),
        )
        context_hit = text_contains_expected(expected, context)

        # 4. Generation
        gen_res = engine.generate(query, context, language=lang)
        raw_answer = gen_res.get("raw_answer", "")
        model_ttft_ms = gen_res.get("model_first_token_ms")
        total_ttft_ms = (
            retrieval_ms + model_ttft_ms
            if model_ttft_ms is not None
            else retrieval_ms + gen_res.get("generation_stage_ttft_ms", 0.0)
        )
        raw_generation_hit = text_contains_expected(expected, raw_answer)

        # 5. Output Guardrail
        final_answer, guardrail_res = apply_output_guardrail(
            gen_res["answer"],
            lang,
            query=query,
            context=context,
            possibly_truncated=gen_res.get("possibly_truncated", False),
            strong_evidence=gen_res.get("strong_evidence", False),
        )
        final_answer_hit = text_contains_expected(expected, final_answer)

        # 6. Classification
        classification = classify_failure(
            expected=expected,
            corpus_hit=corpus_hit,
            retrieval_hit=retrieval_hit,
            context_hit=context_hit,
            raw_generation_hit=raw_generation_hit,
            guardrail_allowed=guardrail_res.allowed,
            final_answer_hit=final_answer_hit,
        )

        category_counts[classification] += 1

        record = {
            "query": query,
            "language": lang,
            "expected": expected,
            "corpus_hit": corpus_hit,
            "retrieval_hit": retrieval_hit,
            "packed_context_hit": context_hit,
            "raw_answer": raw_answer,
            "final_answer": final_answer,
            "guardrail_code": guardrail_res.code or "allowed",
            "failure_type": classification,
            "retrieval_ms": round(retrieval_ms, 2),
            "model_ttft_ms": round(model_ttft_ms, 2) if model_ttft_ms else None,
            "total_ttft_ms": round(total_ttft_ms, 2),
        }
        results.append(record)

        status_icon = "✅" if classification == "PASS" else "❌"
        print(f"{status_icon} [{lang.upper()}] {query[:40]}... -> {classification} (Raw: {raw_answer[:30]} | Final: {final_answer[:30]})")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY REPORT")
    print("=" * 80)
    total = len(results)
    for cat, count in category_counts.items():
        pct = (count / total * 100) if total else 0
        print(f"  {cat:<25}: {count:>3}/{total} ({pct:>5.1f}%)")

    if output_csv:
        csv_path = Path(output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nDetailed CSV exported to: {csv_path.resolve()}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose GOAt AI RAG pipeline failures.")
    parser.add_argument("--csv", type=str, default="rag_diagnosis_report.csv", help="Output CSV path")
    args = parser.parse_args()
    run_diagnostics(output_csv=args.csv)
