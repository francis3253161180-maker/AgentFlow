import unittest
from fractions import Fraction

from agentflow.models.game24_atomic import (
    AtomicAction,
    AtomicState,
    parse_atomic_action,
)


class Game24AtomicTest(unittest.TestCase):
    def test_exact_three_step_success_and_provenance(self):
        state = AtomicState((1, 4, 6, 6))
        state.apply(AtomicAction(left_id="n0", operator="+", right_id="n1"))
        state.apply(AtomicAction(left_id="n2", operator="*", right_id="n4"))
        state.apply(AtomicAction(left_id="n5", operator="-", right_id="n3"))
        node = next(iter(state.active.values()))
        self.assertEqual(node.value, Fraction(24))
        self.assertEqual(node.provenance, (0, 1, 2, 3))
        self.assertEqual(state.terminal_reward(), 1)

    def test_state_transition_and_snapshot(self):
        state = AtomicState((2, 3, 4, 8))
        node = state.apply(AtomicAction(left_id="n0", operator="+", right_id="n1"))
        self.assertEqual(node.value, Fraction(5))
        self.assertEqual(node.provenance, (0, 1))
        self.assertEqual({x["node_id"] for x in state.snapshot()}, {"n2", "n3", "n4"})

    def test_invalid_ids_distinct_operands_and_division_by_zero(self):
        for action, reason in [
            (AtomicAction(left_id="n0", operator="+", right_id="n0"), "distinct"),
            (AtomicAction(left_id="n0", operator="+", right_id="nx"), "active"),
        ]:
            with self.subTest(reason=reason):
                with self.assertRaisesRegex(ValueError, reason):
                    AtomicState((1, 2, 3, 4)).apply(action)
        with self.assertRaisesRegex(ValueError, "division by zero"):
            AtomicState((1, 0, 3, 4)).apply(
                AtomicAction(left_id="n0", operator="/", right_id="n1")
            )

    def test_strict_schema_and_no_free_form_repair(self):
        action = parse_atomic_action('{"left_id":"n0","operator":"*","right_id":"n1"}')
        self.assertEqual(action.operator, "*")
        for raw in (
            'prefix {"left_id":"n0","operator":"+","right_id":"n1"}',
            '{"left_id":"n0","operator":"+","right_id":"n1","extra":1}',
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_atomic_action(raw)

    def test_terminal_reward_requires_three_actions_and_all_inputs(self):
        state = AtomicState((1, 2, 3, 4))
        self.assertEqual(state.terminal_reward(), 0)
        state.apply(AtomicAction(left_id="n0", operator="+", right_id="n1"))
        state.apply(AtomicAction(left_id="n2", operator="+", right_id="n3"))
        state.apply(AtomicAction(left_id="n4", operator="*", right_id="n5"))
        self.assertEqual(state.terminal_reward(), 0)


if __name__ == "__main__":
    unittest.main()
