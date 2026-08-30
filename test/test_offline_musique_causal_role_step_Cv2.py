import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_offline_musique_causal_role_step_Cv2_20260830.py"
SPEC = importlib.util.spec_from_file_location("causal_role_step_cv2", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CausalRoleStepCv2Test(unittest.TestCase):
    def test_opportunity_excludes_consumed_and_respects_capacity(self):
        observation = [{"pid": "gold_a"}, {"pid": "gold_b"}, {"pid": "noise"}]
        self.assertEqual(MODULE.event_opportunity(observation, {"gold_a", "gold_b"}, {"gold_a"}, 1), {"gold_b"})
        self.assertEqual(MODULE.event_opportunity(observation, {"gold_a"}, set(), 0), set())

    def test_normalization_fails_closed_for_degenerate_group(self):
        self.assertEqual(MODULE.normalized([1.0]), ([0.0], False))
        self.assertEqual(MODULE.normalized([1.0, 1.0]), ([0.0, 0.0], False))

    def test_conditional_f2_penalizes_a_distractor_but_not_a_zero_true_positive(self):
        opportunity = {"gold"}
        self.assertEqual(MODULE.conditional_f2({"gold"}, opportunity), 1.0)
        self.assertLess(MODULE.conditional_f2({"gold", "noise"}, opportunity), 1.0)
        self.assertEqual(MODULE.conditional_f2({"noise"}, opportunity), 0.0)

    def test_valid_role_detectors_require_valid_schema(self):
        self.assertTrue(MODULE.is_valid_search({"mode": "DECISION", "semantic_output": {"action": "search"}, "validation_result": {"schema_valid": True}}))
        self.assertFalse(MODULE.is_valid_search({"mode": "DECISION", "semantic_output": {"action": "search"}, "validation_result": {"schema_valid": True, "format_failure": True}}))
        self.assertTrue(MODULE.is_final_answer({"mode": "DECISION", "semantic_output": {"action": "answer"}, "validation_result": {"schema_valid": True}}))


if __name__ == "__main__":
    unittest.main()
