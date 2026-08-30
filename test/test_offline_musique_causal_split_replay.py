import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_offline_musique_causal_split_replay_20260830.py"
SPEC = importlib.util.spec_from_file_location("causal_split_replay", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CausalSplitReplayTest(unittest.TestCase):
    def test_normalization_is_local_and_zero_mean(self):
        values = MODULE.normalize_group([0.0, 0.5, 1.0])
        self.assertAlmostEqual(sum(values) / len(values), 0.0, places=6)
        self.assertGreater(values[-1], values[0])

    def test_degenerate_credit_is_zero(self):
        self.assertEqual(MODULE.normalize_group([0.25]), [0.0])
        self.assertEqual(MODULE.normalize_group([0.25, 0.25]), [0.0, 0.0])

    def test_hop_parser_fails_closed(self):
        self.assertEqual(MODULE.hop_of("3hop1__example"), "3hop")
        with self.assertRaisesRegex(ValueError, "unrecognized"):
            MODULE.hop_of("hopless")


if __name__ == "__main__":
    unittest.main()
