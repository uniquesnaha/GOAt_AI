from __future__ import annotations

import unittest

from app.rag.evidence_quality import evidence_relevance_score


class EvidenceQualityTests(unittest.TestCase):

    def test_relevant_tamil_fact_outranks_unrelated_city(self):
        query = "இந்தியாவின் தலைநகரம் எது?"
        relevant = "இந்தியாவின் தலைநகரம் புதுதில்லி."
        unrelated = "இந்தியானாபோலிஸ் ஒரு பெரிய நகரமாகும்."

        self.assertGreater(
            evidence_relevance_score(query, relevant, "ta"),
            evidence_relevance_score(query, unrelated, "ta"),
        )

    def test_relevant_hindi_fact_outranks_unrelated_temperature(self):
        query = "पानी का क्वथनांक कितने डिग्री सेल्सियस है?"
        relevant = "पानी का क्वथनांक 100 डिग्री सेल्सियस है।"
        unrelated = "आज का तापमान 32 डिग्री सेल्सियस है।"

        self.assertGreater(
            evidence_relevance_score(query, relevant, "hi"),
            evidence_relevance_score(query, unrelated, "hi"),
        )

    def test_question_words_do_not_create_false_relevance(self):
        self.assertEqual(
            evidence_relevance_score(
                "எது?",
                "இது தொடர்பில்லாத உரை.",
                "ta",
            ),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
