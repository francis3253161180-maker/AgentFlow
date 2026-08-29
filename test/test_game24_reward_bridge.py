import unittest
from types import SimpleNamespace
from unittest.mock import patch

import train.utils as reward_utils
from agentflow.models.structured_outputs import extract_final_answer, game24_reward_decision


class Game24RewardBridgeTest(unittest.TestCase):
    question = "Using the numbers [3, 8, 12, 13], create an expression that equals 24."

    def test_bare_valid_expression_is_strictly_accepted(self):
        decision, details = game24_reward_decision(
            self.question, "3 * 8 * (13 - 12) = 24"
        )
        self.assertTrue(decision)
        self.assertEqual(details["reason"], "proved_fraction_24")
        self.assertTrue(
            reward_utils.compute_score(
                self.question,
                "24",
                "3 * 8 * (13 - 12) = 24",
            )
        )

    def test_tagged_and_json_valid_expressions_remain_accepted(self):
        tagged, _ = game24_reward_decision(
            self.question, "<answer>3 * 8 * (13 - 12)</answer>"
        )
        structured, _ = game24_reward_decision(
            self.question, '{"expression":"3 * 8 * (13 - 12)"}'
        )
        self.assertTrue(tagged)
        self.assertTrue(structured)

    def test_bare_invalid_expression_and_wrong_multiset_fail(self):
        not_24, not_24_details = game24_reward_decision(
            self.question, "3 + 8 + 12 + 13"
        )
        wrong_numbers, wrong_details = game24_reward_decision(
            self.question, "6 * 4"
        )
        self.assertFalse(not_24)
        self.assertEqual(not_24_details["reason"], "not_24")
        self.assertFalse(wrong_numbers)
        self.assertEqual(wrong_details["reason"], "wrong_number_multiset")

    def test_unrelated_text_is_not_promoted(self):
        decision, details = game24_reward_decision(
            self.question, "The numbers are difficult; no expression was found."
        )
        self.assertFalse(decision)
        self.assertNotEqual(details["reason"], "proved_fraction_24")

    def test_extraction_prefers_last_answer_tag(self):
        output = (
            "Reasoning <answer>3 + 8 + 12 + 13</answer>\n"
            "Final <answer>3 * 8 * (13 - 12) = 24</answer>"
        )
        self.assertEqual(extract_final_answer(output), "3 * 8 * (13 - 12) = 24")

    def test_extraction_accepts_explicit_answer_section_only(self):
        output = """### Process Summary
The incidental expression 3 * 8 * (13 - 12) = 24 is discussed above.

### Answer
3 * 8 * (13 - 12) = 24
"""
        extracted = extract_final_answer(output)
        self.assertEqual(extracted, "3 * 8 * (13 - 12) = 24")
        self.assertTrue(game24_reward_decision(self.question, extracted)[0])

    def test_incidental_prose_is_not_mined_for_an_expression(self):
        output = "The incidental expression 3 * 8 * (13 - 12) = 24 is not my answer."
        extracted = extract_final_answer(output)
        self.assertEqual(extracted, output)
        self.assertFalse(game24_reward_decision(self.question, extracted)[0])

    def test_explicit_answer_section_keeps_wrong_expression_strict(self):
        output = "**Answer:** 3 + 8 + 12 + 13"
        extracted = extract_final_answer(output)
        self.assertEqual(extracted, "3 + 8 + 12 + 13")
        self.assertFalse(game24_reward_decision(self.question, extracted)[0])

    def test_explicit_answer_section_keeps_wrong_multiset_strict(self):
        output = "### Answer\n6 * 4"
        extracted = extract_final_answer(output)
        self.assertEqual(extracted, "6 * 4")
        self.assertFalse(game24_reward_decision(self.question, extracted)[0])

    def test_numbered_explicit_answer_section_is_delimited(self):
        output = "1. **Process Summary:** details\n2. **Answer:** 3 * 8 * (13 - 12) = 24"
        extracted = extract_final_answer(output)
        self.assertEqual(extracted, "3 * 8 * (13 - 12) = 24")

    def test_non_game24_compute_score_still_uses_generic_scorer(self):
        stub = SimpleNamespace(
            score_with_metadata=lambda question, groundtruth, answer: SimpleNamespace(
                score=True
            )
        )
        with patch.object(reward_utils, "_default_reward_scorer", stub):
            self.assertTrue(
                reward_utils.compute_score(
                    "What is the capital of France?",
                    "Paris",
                    "The answer is Paris.",
                )
            )


if __name__ == "__main__":
    unittest.main()
