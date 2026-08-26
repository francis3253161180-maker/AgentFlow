import os
import unittest


os.environ.pop("AGENTFLOW_USE_LLM_SCORER", None)

from train.utils import compute_score, eval


class DeterministicRewardScorerTest(unittest.TestCase):
    def assertScore(self, groundtruth, answer, expected=True):
        self.assertEqual(compute_score("offline test", groundtruth, answer), expected)

    def test_natural_language_short_answers(self):
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

    def test_answer_block_is_preferred(self):
        self.assertScore("Daimler-Benz", "Reasoning mentions it. **Answer:** Daimler-Benz")
        self.assertScore("Oscar the Grouch", "<answer>Oscar the Grouch</answer>")

    def test_yes_no_and_negation(self):
        self.assertScore("Yes", "Yes, such a natural number exists.")
        self.assertScore("No", "No, the proposed statement is false.")
        self.assertScore("Yes", "No, such a number does not exist.", expected=False)
        self.assertScore("Oscar the Grouch", "The answer is not Oscar the Grouch; it is Big Bird.", expected=False)
        self.assertScore("Yes", "The answer is not yes.", expected=False)
        self.assertScore("Oscar the Grouch", "Oscar the Grouch is not the answer.", expected=False)

    def test_dates_are_complete_values(self):
        self.assertScore("October 3, 2017", "The album was released on May 19, 2017.", expected=False)
        self.assertScore("October 3, 2017", "The album was May 19, 2017; the title track was October 3, 2017.")
        self.assertScore("7 April 2018", "The maiden voyage was March 31, 2018.", expected=False)

    def test_numbers_do_not_share_intermediate_tokens(self):
        self.assertScore("13", "20", expected=False)
        self.assertScore("13", "The final answer is 13.")
        self.assertScore("13", "The answer is 20, not 13.", expected=False)

    def test_math_equivalence_and_normalization(self):
        self.assertScore(r"\dfrac{1}{2}", r"\frac{1}{2}")
        self.assertScore(r"1 + \sqrt{2}", "√2 + 1")
        self.assertScore("x + 1", r"f(x) = x + 1")
        self.assertScore(r"\dfrac{1}{3}", r"x_0 = \frac{1}{3}")
        self.assertScore(r"\dfrac{1}{6}", r"The residue is \frac{1}{6}.")

    def test_false_entity_and_numeric_cases(self):
        self.assertScore("Louise Glover", "The girl was Sophie Sumner.", expected=False)
        self.assertScore("7 April 2018", "March 31, 2018", expected=False)
        self.assertScore("Grant Park", "The answer is not Grant Park.", expected=False)

    def test_eval_signature_returns_binary_float(self):
        self.assertEqual(eval("question", "Oscar the Grouch", "Oscar the Grouch lives there."), 1.0)
        self.assertEqual(eval("question", "Oscar the Grouch", "Big Bird lives there."), 0.0)


if __name__ == "__main__":
    unittest.main()
