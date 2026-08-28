import unittest

from agentflow.models.structured_outputs import (
    Game24Answer,
    StructuredToolCall,
    StructuredToolResponse,
    StructuredVerifierFeedback,
    game24_prompt,
    parse_game24_answer,
    parse_strict_json,
    select_valid_candidate,
    validate_game24_expression,
)


class StructuredOutputHarnessTest(unittest.TestCase):
    numbers = (1, 2, 3, 4)

    def test_valid_json_and_fraction_validation(self):
        answer, result = parse_game24_answer('{"expression":"(1+2+3)*4"}', self.numbers)
        self.assertEqual(answer.expression, "(1+2+3)*4")
        self.assertEqual(result["reason"], "proved_fraction_24")

    def test_malformed_extra_prose_and_extra_key_are_rejected(self):
        for raw in ("prose {\"expression\":\"(1+2+3)*4\"}", '{"expression":"x", "extra": 1}'):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_strict_json(raw, Game24Answer)

    def test_wrong_multiset_duplicate_missing_and_divide_by_zero(self):
        cases = {
            "(1+2+3)*5": "wrong_number_multiset",
            "(1+1+3)*4": "wrong_number_multiset",
            "1/(2-2)+3+4": "divide_by_zero",
        }
        for expression, reason in cases.items():
            with self.subTest(expression=expression):
                self.assertEqual(validate_game24_expression(expression, self.numbers)["reason"], reason)

    def test_exact_fraction_and_unsupported_syntax(self):
        self.assertTrue(validate_game24_expression("8/(3-8/3)", (3, 3, 8, 8))["valid"])
        self.assertIn(validate_game24_expression("1**2+3+4", self.numbers)["reason"], {"unsupported_syntax", "unsupported_expression"})

    def test_marked_candidate_selection_does_not_promote_prose(self):
        text = "The example is 1+2. Answer: (1+2+3)*4"
        candidate, result = select_valid_candidate(text, self.numbers)
        self.assertEqual(candidate, "(1+2+3)*4")
        self.assertTrue(result["valid"])
        self.assertIsNone(select_valid_candidate("The answer mentions 24 only.", self.numbers)[0])

    def test_structured_retry_feedback_is_explicit(self):
        prompt = game24_prompt("Using the numbers [1, 2, 3, 4], create 24.", "{}", "reason=wrong_number_multiset")
        self.assertIn('Return one JSON object and nothing else.', prompt)
        self.assertIn('reason=wrong_number_multiset', prompt)

    def test_role_schemas_are_strict(self):
        self.assertEqual(StructuredToolCall(tool_name="tool", query="q").query, "q")
        self.assertEqual(StructuredToolResponse(success=True, result="ok").result, "ok")
        self.assertFalse(StructuredVerifierFeedback(stop=False, reason="retry").stop)
        with self.assertRaises(Exception):
            StructuredToolCall(tool_name="tool", query="q", unexpected=True)

    def test_vllm_guided_json_schema_conversion(self):
        from agentflow.engine.vllm import ChatVLLM

        schema = ChatVLLM._guided_json_schema(Game24Answer)
        self.assertEqual(schema["type"], "object")
        self.assertIn("expression", schema["properties"])

    def test_validated_expression_keeps_scorer_answer_boundary(self):
        # The production rollout bridge wraps a validated expression in the
        # same explicit boundary used by the deterministic numeric scorer.
        from train.utils import deterministic_decision

        expression = "(12-8)*(7-1)"
        self.assertIsNone(deterministic_decision("24", expression).value)
        self.assertTrue(
            deterministic_decision("24", f"<answer>{expression}</answer>").value
        )


if __name__ == "__main__":
    unittest.main()
