import unittest

from agentflow.verl.dynamic_padding import make_response_padding_plan


class DynamicPaddingTest(unittest.TestCase):
    def test_width_is_current_batch_max_but_hard_cap_is_respected(self):
        plan = make_response_padding_plan([3, 7, 20], hard_cap=10)
        self.assertEqual(plan.raw_max, 20)
        self.assertEqual(plan.clipped_lengths, (3, 7, 10))
        self.assertEqual(plan.effective_width, 10)
        self.assertEqual(plan.fixed_elements, 30)
        self.assertEqual(plan.dynamic_elements, 30)

    def test_short_batch_saves_padding_without_changing_lengths(self):
        plan = make_response_padding_plan([3, 7, 5], hard_cap=10)
        self.assertEqual(plan.effective_width, 7)
        self.assertEqual(plan.fixed_elements, 30)
        self.assertEqual(plan.dynamic_elements, 21)
        self.assertEqual(plan.padding_elements_saved, 9)
        self.assertGreater(plan.padding_ratio, 0.0)

    def test_empty_and_invalid_inputs(self):
        empty = make_response_padding_plan([], hard_cap=10)
        self.assertEqual(empty.effective_width, 0)
        with self.assertRaises(ValueError):
            make_response_padding_plan([-1], hard_cap=10)
        with self.assertRaises(ValueError):
            make_response_padding_plan([1], hard_cap=0)


if __name__ == "__main__":
    unittest.main()
