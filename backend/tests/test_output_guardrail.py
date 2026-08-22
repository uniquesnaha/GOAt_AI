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


    def test_model_abstained_returns_abstained_code(self):
        _, result = apply_output_guardrail(
            "NOT_FOUND",
            "ta",
            query="இந்தியாவின் தேசியப் பறவை எது?",
            context="சில தகவல்கள் இங்கே உள்ளன.",
        )

        self.assertFalse(result.allowed)
        self.assertEqual(result.code, "model_abstained")

    def test_rejects_distractor_in_unrelated_sentence(self):
        # Mercury ("புதன்") appears in context, but in the sentence about Mercury,
        # not the sentence answering Earth's satellite. Same-sentence support must reject it.
        _, result = apply_output_guardrail(
            "புதன்",
            "ta",
            query="பூமியின் ஒரே இயற்கை துணைக்கோள் எது?",
            context=(
                "1. புதன் - சூரிய மண்டலத்தில் மிக அருகில் உள்ள கிரகம். "
                "2. பூமி - சூரிய மண்டலத்தில் உயிர் உள்ள கிரகம். "
                "3. இது ஒரு இயற்கை சந்திரனைக் கொண்டுள்ளது."
            ),
        )

        self.assertFalse(result.allowed)
        self.assertEqual(result.code, "unsupported_answer")

    def test_accepts_supported_photosynthesis_gas(self):
        answer, result = apply_output_guardrail(
            "கார்பன் டை ஆக்சைடு",
            "ta",
            query="தாவரங்கள் ஒளிச்சேர்க்கைக்கு பயன்படுத்தும் வாயு எது?",
            context="தாவரங்கள் ஒளிச்சேர்க்கைக்கு கார்பன் டை ஆக்சைடு வாயுவை உள்வாங்கி பயன்படுத்துகின்றன.",
        )

        self.assertTrue(result.allowed)
        self.assertEqual(answer, "கார்பன் டை ஆக்சைடு")


    def test_peacock_supported_generation(self):
        answer, result = apply_output_guardrail(
            "இந்திய மயில்",
            "ta",
            query="இந்தியாவின் தேசியப் பறவை எது?",
            context="இந்தியாவின் தேசியப் பறவை இந்திய மயில் ஆகும்.",
        )
        self.assertTrue(result.allowed)
        self.assertIn("மயில்", answer)

    def test_tamil_inflection_grounding_heart(self):
        from app.guardrails.guardrails import answer_is_grounded
        self.assertTrue(
            answer_is_grounded(
                "இதயம்",
                "இதயத்தில் இரண்டு பம்புகள் உள்ளன.",
            )
        )

    def test_irrelevant_retrieval_abstains(self):
        answer, result = apply_output_guardrail(
            "NOT_FOUND",
            "ta",
            query="பூமியின் இயற்கை துணைக்கோள் எது?",
            context="பூமியின் சுற்றுப்பாதை தளம் சூரியனைச் சுற்றி அமைகிறது.",
        )
        self.assertFalse(result.allowed)
        self.assertEqual(result.code, "model_abstained")

    def test_numeric_answer_type_validation(self):
        from app.guardrails.guardrails import answer_type_is_valid
        question = "ஒளியின் வேகம் ஒரு வினாடிக்கு தோராயமாக எத்தனை கிலோமீட்டர்?"
        self.assertFalse(
            answer_type_is_valid(
                question,
                "ஒளி",
                "ta",
            )
        )
        self.assertTrue(
            answer_type_is_valid(
                question,
                "300000",
                "ta",
            )
        )

    def test_great_bath_mohenjodaro(self):
        question = "சிந்து சமவெளி நாகரிகத்தின் பெரிய குளியல் குளம் எங்கு கண்டுபிடிக்கப்பட்டது?"
        context = (
            "பாகிஸ்தானின் சிந்து மாகாணத்தில் உள்ள மோஹென்ஜோ-தாரோவின் "
            "அகழ்வாராய்ச்சி செய்யப்பட்ட இடிபாடுகள், முன்புறத்தில் பெரிய குளியல் குளத்தைக் காட்டுகின்றன."
        )
        answer, result = apply_output_guardrail(
            "மோஹென்ஜோ-தாரோ",
            "ta",
            query=question,
            context=context,
        )
        self.assertTrue(result.allowed)
        self.assertIn("மோஹென்ஜோ", answer)


if __name__ == "__main__":
    unittest.main()


