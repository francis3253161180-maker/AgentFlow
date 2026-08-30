import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_offline_musique_transition_replay_20260830.py"
SPEC = importlib.util.spec_from_file_location("transition_replay", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TransitionReplayTest(unittest.TestCase):
    def test_non_degenerate_group_is_zero_mean(self):
        values = MODULE.normalize_group([0.0, 1.0, 2.0])
        self.assertAlmostEqual(sum(values) / len(values), 0.0, places=6)
        self.assertGreater(values[-1], 0)

    def test_degenerate_groups_have_zero_credit(self):
        self.assertEqual(MODULE.normalize_group([1.0]), [0.0])
        self.assertEqual(MODULE.normalize_group([1.0, 1.0]), [0.0, 0.0])

    def test_hop_parser(self):
        self.assertEqual(MODULE.hop_of("4hop3__x"), "4hop")
        with self.assertRaisesRegex(ValueError, "unrecognized"):
            MODULE.hop_of("unknown")


if __name__ == "__main__":
    unittest.main()
