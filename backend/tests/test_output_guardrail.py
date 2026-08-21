from __future__ import annotations

import unittest

from app.guardrails.guardrails import apply_output_guardrail


class OutputGuardrailTests(unittest.TestCase):

    def test_cleans_supported_tamil_answer_and_citation(self):
        answer, result = apply_output_guardrail(
            "[1] **புதுதில்லி**",
            "ta",
            query="இந்தியாவின் தலைநகரம் எது?",
            context="இந்தியாவின் தலைநகரம் புதுதில்லி.",
        )

        self.assertTrue(result.allowed)
        self.assertEqual(answer, "புதுதில்லி")

    def test_rejects_citation_only(self):
        _, result = apply_output_guardrail(
            "[2]",
            "ta",
            query="சூரிய குடும்பத்தில் மிகப்பெரிய கோள் எது?",
            context="வியாழன் மிகப்பெரிய கோள்.",
        )

        self.assertFalse(result.allowed)
        self.assertEqual(result.code, "citation_only")

    def test_rejects_wrong_script(self):
        _, result = apply_output_guardrail(
            "Puthalini",
            "ta",
            query="இந்தியாவின் தலைநகரம் எது?",
            context="இந்தியாவின் தலைநகரம் புதுதில்லி.",
        )

        self.assertFalse(result.allowed)
        self.assertEqual(result.code, "wrong_language")

    def test_rejects_question_echo(self):
        _, result = apply_output_guardrail(
            "சூரிய குடும்பத்தில் மிகப்பெரிய கோள்",
            "ta",
            query="சூரிய குடும்பத்தில் மிகப்பெரிய கோள் எது?",
            context="வியாழன் சூரிய குடும்பத்தில் மிகப்பெரிய கோள்.",
        )

        self.assertFalse(result.allowed)
        self.assertEqual(result.code, "question_echo")

    def test_rejects_unsupported_answer(self):
        _, result = apply_output_guardrail(
            "சென்னை",
            "ta",
            query="இந்தியாவின் தலைநகரம் எது?",
            context="இந்தியாவின் தலைநகரம் புதுதில்லி.",
        )

        self.assertFalse(result.allowed)
        self.assertEqual(result.code, "unsupported_answer")

    def test_accepts_supported_hindi_numeric_answer(self):
        answer, result = apply_output_guardrail(
            "100 डिग्री सेल्सियस",
            "hi",
            query="पानी का क्वथनांक कितने डिग्री सेल्सियस है?",
            context="पानी का क्वथनांक 100 डिग्री सेल्सियस है।",
        )

        self.assertTrue(result.allowed)
        self.assertEqual(answer, "100 डिग्री सेल्सियस")

    def test_rejects_token_limit_output(self):
        _, result = apply_output_guardrail(
            "வியாழன்",
            "ta",
            query="சூரிய குடும்பத்தில் மிகப்பெரிய கோள் எது?",
            context="வியாழன் மிகப்பெரிய கோள்.",
            possibly_truncated=True,
        )

        self.assertFalse(result.allowed)
        self.assertEqual(result.code, "truncated_answer")


if __name__ == "__main__":
    unittest.main()
