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
        def failing_judge(*args):
            raise TimeoutError("simulated timeout")

        scorer = HybridRewardScorer(
            judge=failing_judge,
            cache=RewardJudgeCache(self.tempdir.name),
            enabled=True,
        )
        result = scorer.score_with_metadata("q", "target", "a long answer that mentions target")
        self.assertFalse(result.score)
        self.assertEqual(result.route, "conservative_fallback")
        self.assertEqual(result.judge_error, "TimeoutError")

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
