import os
import tempfile
import unittest


# Ordinary unit tests must never spend on a live provider.
os.environ["AGENTFLOW_REWARD_JUDGE_ENABLED"] = "0"

import train.utils as reward_utils
from train.reward_judge import HybridRewardScorer, RewardJudgeCache, parse_judge_response
from test.hybrid_reward_cases import SYNTHETIC_CASES


class RecordingJudge:
    def __init__(self, verdict=True):
        self.verdict = verdict
        self.calls = []

    def __call__(self, question, groundtruth, answer):
        self.calls.append((question, groundtruth, answer))
        return {"true_false": self.verdict, "analysis": "mock"}


class HybridRewardScorerTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.judge = RecordingJudge()
        self.scorer = HybridRewardScorer(
            judge=self.judge,
            cache=RewardJudgeCache(self.tempdir.name),
            enabled=True,
        )
        reward_utils._default_reward_scorer = self.scorer

    def tearDown(self):
        reward_utils._default_reward_scorer = None
        self.tempdir.cleanup()

    def assertScore(self, groundtruth, answer, expected=True):
        self.judge.verdict = expected
        self.assertEqual(reward_utils.compute_score("offline test", groundtruth, answer), expected)

    def test_seen_natural_language_answers_use_judge(self):
        cases = [
            ("Oscar the Grouch", "Oscar the Grouch lives in the trash can on Sesame Street."),
            ("Daimler-Benz", "BMW was nearly sold to Daimler-Benz in 1959."),
            ("John McCrae", "The poem was written by Lieutenant Colonel John McCrae."),
            ("bust", "Going over 21 in blackjack is called a bust."),
            ("Chicago's Grant Park", "The Art Institute of Chicago is located in Grant Park."),
        ]
        for groundtruth, answer in cases:
            with self.subTest(groundtruth=groundtruth):
                self.assertScore(groundtruth, answer)
        self.assertEqual(len(self.judge.calls), len(cases))

    def test_explicit_answer_and_math_are_local(self):
        self.assertScore("Daimler-Benz", "Reasoning mentions it. **Answer:** Daimler-Benz")
        self.assertScore("Oscar the Grouch", "<answer>Oscar the Grouch</answer>")
        self.assertScore(r"\dfrac{1}{2}", r"\frac{1}{2}")
        self.assertScore(r"1 + \sqrt{2}", "√2 + 1")
        self.assertScore("x + 1", r"f(x) = x + 1")
        self.assertEqual(len(self.judge.calls), 0)

    def test_explicit_numeric_arithmetic_expression_is_local(self):
        # Game24 stores the scalar target ``24``.  A tagged arithmetic
        # expression must be evaluated as a whole, not rejected because its
        # intermediate numbers are also present in the answer.
        result = self.scorer.score_with_metadata(
            "Game24 arithmetic task", "24", "<answer>(12 * (13 - (3 + 8)))</answer>"
        )
        self.assertTrue(result.score)
        self.assertEqual(result.route, "deterministic")
        self.assertEqual(result.reason, "proved_numeric_expression")

        wrong = self.scorer.score_with_metadata(
            "Game24 arithmetic task", "24", "<answer>(11 - 2) * (2 + 1)</answer>"
        )
        self.assertFalse(wrong.score)
        self.assertEqual(wrong.route, "deterministic")
        self.assertEqual(wrong.reason, "numeric_expression_mismatch")
        self.assertEqual(len(self.judge.calls), 0)

    def test_numeric_expression_does_not_promote_unmarked_prose(self):
        # Without a final-answer marker, the full response remains on the
        # conservative numeric-token path; an incidental 24 is not a proof.
        self.judge.verdict = False
        result = self.scorer.score_with_metadata(
            "Game24 arithmetic task", "24", "I tried (12 * 2), but the final answer is unknown."
        )
        self.assertFalse(result.score)
        self.assertNotEqual(result.reason, "proved_numeric_expression")

    def test_math_conclusion_extraction_is_local_and_conservative(self):
        # An affirmative prose wrapper around an equivalent equality is a
        # high-confidence math claim, not an open-ended entity answer.
        result = self.scorer.score_with_metadata(
            "offline equality", "a = b", r"Yes, it is true that \(a = b\)."
        )
        self.assertTrue(result.score)
        self.assertEqual(result.route, "deterministic")
        self.assertEqual(result.reason, "proved_math_equivalence")
        self.assertEqual(len(self.judge.calls), 0)

        # The same target being rejected must not become a local positive.
        self.judge.verdict = False
        rejected = self.scorer.score_with_metadata(
            "offline equality", "a = b", r"No, \(a = b\) is false; the answer is \(a = c\)."
        )
        self.assertFalse(rejected.score)
        self.assertEqual(rejected.route, "deterministic")
        self.assertEqual(rejected.reason, "safe_math_mismatch")
        self.assertEqual(len(self.judge.calls), 0)

        # A long derivation with a final equivalent formula is still safe when
        # the answer block contains the final mathematical conclusion.
        final = self.scorer.score_with_metadata(
            "offline group order",
            r"2^{n+1}",
            r"The derivation has several intermediate relations. **Answer:** "
            r"\[|H_n| = 2^{n+1}.\]",
        )
        self.assertTrue(final.score)
        self.assertEqual(final.route, "deterministic")
        self.assertEqual(len(self.judge.calls), 0)

    def test_math_mentions_without_final_claim_do_not_get_promoted(self):
        self.judge.verdict = False
        wrong_relation = self.scorer.score_with_metadata(
            "offline equality", "a = b", "c = b"
        )
        self.assertFalse(wrong_relation.score)
        self.assertEqual(wrong_relation.route, "deterministic")

        result = self.scorer.score_with_metadata(
            "offline equality", "a = b", "The possibilities include a = b and c = d."
        )
        self.assertFalse(result.score)
        # The local route remains conservative for an unmarked multi-candidate
        # statement; a real hybrid scorer would send it to the judge.
        self.assertEqual(result.route, "judge")
        self.assertEqual(len(self.judge.calls), 1)

        explicit_list = self.scorer.score_with_metadata(
            "offline equality", "a = b", "Answer: a = b or c = b"
        )
        self.assertFalse(explicit_list.score)
        self.assertEqual(explicit_list.route, "judge")
        self.assertEqual(len(self.judge.calls), 2)

    def test_yes_no_and_negation(self):
        self.assertScore("Yes", "Yes, such a natural number exists.")
        self.assertScore("No", "No, the proposed statement is false.")
        self.assertScore("Yes", "No, such a number does not exist.", expected=False)
        self.assertScore("Oscar the Grouch", "The answer is not Oscar the Grouch; it is Big Bird.", expected=False)
        self.assertScore("Yes", "The answer is not yes.", expected=False)
        self.assertScore("Oscar the Grouch", "Oscar the Grouch is not the answer.", expected=False)

    def test_dates_and_numbers_are_complete_values(self):
        self.assertScore("October 3, 2017", "The album was May 19, 2017.", expected=False)
        self.assertScore("October 3, 2017", "The album was May 19, 2017; the title track was October 3, 2017.")
        self.assertScore("7 April 2018", "The maiden voyage was March 31, 2018.", expected=False)
        self.assertScore("13", "20", expected=False)
        self.assertScore("13", "The final answer is 13.")
        self.assertScore("13", "The answer is 20, not 13.", expected=False)

    def test_cache_deduplicates_judge_calls(self):
        first = self.scorer.score_with_metadata("q", "a target", "a long answer mentioning the target")
        second = self.scorer.score_with_metadata("q", "a target", "a long answer mentioning the target")
        self.assertTrue(first.score)
        self.assertTrue(second.score)
        self.assertEqual(first.route, "judge")
        self.assertEqual(second.route, "judge_cache")
        self.assertEqual(len(self.judge.calls), 1)
        cached_files = [name for name in os.listdir(self.tempdir.name) if name.endswith(".json")]
        self.assertEqual(len(cached_files), 1)
        with open(os.path.join(self.tempdir.name, cached_files[0]), encoding="utf-8") as handle:
            self.assertNotIn("a long answer", handle.read())

    def test_strict_judge_parser(self):
        self.assertTrue(parse_judge_response('{"true_false": true}').true_false)
        self.assertFalse(parse_judge_response('```json\n{"true_false": false}\n```').true_false)
        with self.assertRaises(ValueError):
            parse_judge_response("The answer is probably true.")

    def test_api_failure_is_conservative(self):
        failures = [
            ("TimeoutError", lambda *args: (_ for _ in ()).throw(TimeoutError("simulated timeout"))),
            ("RuntimeError", lambda *args: (_ for _ in ()).throw(RuntimeError("simulated HTTP error"))),
            ("ValueError", lambda *args: "not valid JSON"),
        ]
        for index, (error_type, failing_judge) in enumerate(failures):
            with self.subTest(error_type=error_type):
                scorer = HybridRewardScorer(
                    judge=failing_judge,
                    cache=RewardJudgeCache(os.path.join(self.tempdir.name, str(index))),
                    enabled=True,
                )
                result = scorer.score_with_metadata(
                    "q", "target", "a long answer that mentions target"
                )
                self.assertFalse(result.score)
                self.assertEqual(result.route, "conservative_fallback")
                self.assertEqual(result.judge_error, error_type)
                self.assertIn(float(result.score), {0.0, 1.0})

    def test_synthetic_adversarial_accuracy_and_routing(self):
        for case in SYNTHETIC_CASES:
            judge = RecordingJudge(case.expected)
            scorer = HybridRewardScorer(judge=judge, enabled=True)
            result = scorer.score_with_metadata(
                "full synthetic question: " + case.name,
                case.groundtruth,
                case.answer,
            )
            with self.subTest(case=case.name):
                self.assertEqual(result.score, case.expected)
                self.assertEqual(result.route, case.expected_route)
                if case.expected_route == "judge":
                    self.assertEqual(len(judge.calls), 1)
                    self.assertEqual(judge.calls[0][0], "full synthetic question: " + case.name)
                else:
                    self.assertEqual(len(judge.calls), 0)

    def test_eval_signature_returns_binary_float(self):
        self.judge.verdict = True
        self.assertEqual(reward_utils.eval("question", "Oscar the Grouch", "Oscar the Grouch lives there."), 1.0)
        self.judge.verdict = False
        self.assertEqual(reward_utils.eval("question", "Oscar the Grouch", "Big Bird lives there."), 0.0)


if __name__ == "__main__":
    unittest.main()
