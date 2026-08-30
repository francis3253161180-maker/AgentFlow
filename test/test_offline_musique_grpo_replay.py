import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

from agentflow.verl.trainer import materialize_offline_replay
from agentflow.verl.unified_smoke_capture import _field_digest


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_offline_musique_grpo_replay_20260830.py"
SPEC = importlib.util.spec_from_file_location("offline_musique_replay", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class OfflineMusiqueGRPOReplayTest(unittest.TestCase):
    def test_left_padding(self):
        values, mask = MODULE.left_pad([4, 5], 4, 0)
        self.assertEqual(values, [0, 0, 4, 5])
        self.assertEqual(mask, [0, 0, 1, 1])

    def test_right_padding(self):
        values, mask = MODULE.right_pad([4, 5], 4, 0)
        self.assertEqual(values, [4, 5, 0, 0])
        self.assertEqual(mask, [1, 1, 0, 0])

    def test_padding_fails_closed_instead_of_truncating(self):
        with self.assertRaisesRegex(ValueError, "exceeds"):
            MODULE.left_pad([1, 2, 3], 2, 0)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            MODULE.right_pad([1, 2, 3], 2, 0)

    def test_materialization_does_not_mutate_authenticated_metadata(self):
        tensors = {"input_ids": torch.tensor([[1, 2]], dtype=torch.long)}
        non_tensor = {"rollout_id_list": np.array(["r0"], dtype=object)}
        meta_info = {"temperature": 0.7, "multi_turn": None}
        pack = {
            "tensor_fields": tensors,
            "non_tensor_batch": non_tensor,
            "meta_info": meta_info,
            "metadata": {"temperature": 0.7},
        }
        before = _field_digest(tensors, non_tensor, meta_info)
        replay = materialize_offline_replay(pack, multi_turn=False)
        after = _field_digest(tensors, non_tensor, meta_info)
        self.assertEqual(before, after)
        self.assertIsNone(pack["meta_info"]["multi_turn"])
        self.assertFalse(replay.meta_info["multi_turn"])


if __name__ == "__main__":
    unittest.main()
