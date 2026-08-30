import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from tensordict import TensorDict
from torch import nn

from agentflow.verl import unified_smoke_capture as capture


class UnifiedSmokeCaptureTest(unittest.TestCase):
    def setUp(self):
        capture._PRE_HASH = None
        capture._POST_HASH = None
        capture._REPLAY_CAPTURED = False

    def test_lora_hash_changes_only_for_trainable_lora(self):
        module = nn.Module()
        module.lora_A = nn.Linear(2, 2, bias=False)
        module.base = nn.Linear(2, 2, bias=False)
        module.base.weight.requires_grad = False
        before = capture._lora_hash(module)
        with torch.no_grad():
            module.lora_A.weight.add_(1.0)
        after = capture._lora_hash(module)
        self.assertEqual(before["tensor_count"], 1)
        self.assertNotEqual(before["hash"], after["hash"])

    def test_runtime_pack_contains_update_tensors_and_roundtrips(self):
        module = nn.Module()
        module.lora_A = nn.Linear(2, 2, bias=False)
        batch = TensorDict(
            {
                "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "responses": torch.tensor([[2, 3]], dtype=torch.long),
                "response_mask": torch.tensor([[1, 1]], dtype=torch.long),
                "old_log_probs": torch.tensor([[-0.2, -0.3]], dtype=torch.float32),
                "advantages": torch.tensor([[0.5, -0.5]], dtype=torch.float32),
                "token_level_rewards": torch.tensor([[0.0, 1.0]], dtype=torch.float32),
            },
            batch_size=[1],
        )
        data = SimpleNamespace(
            batch=batch,
            non_tensor_batch={"uid": np.array(["g0"], dtype=object)},
            meta_info={"temperature": 0.7, "seed": 20260828},
        )
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            checksum = directory / "checksum.json"
            pack = directory / "pack.pt"
            with patch.dict(
                os.environ,
                {
                    "AGENTFLOW_LORA_CHECKSUM_ENABLED": "1",
                    "AGENTFLOW_LORA_CHECKSUM_PATH": str(checksum),
                    "AGENTFLOW_REPLAY_CAPTURE_ENABLED": "1",
                    "AGENTFLOW_REPLAY_PACK_PATH": str(pack),
                    "AGENTFLOW_UNIFIED_MODEL_PATH": "local-model",
                },
                clear=False,
            ):
                capture.capture_lora_pre(module)
                with torch.no_grad():
                    module.lora_A.weight.add_(0.1)
                capture.capture_lora_post(module, 0.5)
                capture.capture_replay_pre_update(data)
            self.assertTrue(json.loads(checksum.read_text())["hash_changed"])
            self.assertTrue(pack.exists())
            # Importing the validator is intentionally avoided here; the
            # validator test is run as a fresh process by the smoke script.
            loaded = torch.load(pack, map_location="cpu", weights_only=False)
            self.assertTrue(loaded["field_inventory"]["tensor_fields"])
            self.assertIn("input_ids", loaded["field_inventory"]["tensor_fields"])
            self.assertEqual(loaded["metadata"]["model_path"], "local-model")

    def test_zero_gradient_still_captures_exact_post_state(self):
        module = nn.Module()
        module.lora_A = nn.Linear(2, 2, bias=False)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            checksum = directory / "checksum.json"
            snapshot = directory / "post.pt"
            with patch.dict(
                os.environ,
                {
                    "AGENTFLOW_LORA_CHECKSUM_ENABLED": "1",
                    "AGENTFLOW_LORA_CHECKSUM_PATH": str(checksum),
                    "AGENTFLOW_LORA_POST_SNAPSHOT_PATH": str(snapshot),
                },
                clear=False,
            ):
                capture.capture_lora_pre(module)
                capture.capture_lora_post(module, 0.0)

            result = json.loads(checksum.read_text())
            self.assertEqual(result["grad_norm"], 0.0)
            self.assertFalse(result["hash_changed"])
            self.assertEqual(result["changed_tensor_count"], 0)
            saved = torch.load(snapshot, map_location="cpu", weights_only=False)
            self.assertEqual(saved["kind"], "agentflow_post_optimizer_lora_snapshot")
            self.assertEqual(saved["lora_hash"], result["post"]["hash"])
            self.assertEqual(set(saved["lora_state"]), {"lora_A.weight"})

    def test_nonfinite_gradient_fails_closed(self):
        module = nn.Module()
        module.lora_A = nn.Linear(2, 2, bias=False)
        with patch.dict(os.environ, {"AGENTFLOW_LORA_CHECKSUM_ENABLED": "1"}, clear=False):
            capture.capture_lora_pre(module)
            with self.assertRaisesRegex(RuntimeError, "non-finite actor gradient norm"):
                capture.capture_lora_post(module, float("nan"))

    @unittest.skipUnless(
        Path(
            os.getenv(
                "AGENTFLOW_BEHAVIOR_SNAPSHOT_TEST_PATH",
                "/root/autodl-tmp/tmp/random30_fresh_rollout_replay_20260828/"
                "random30-fresh-rollout-replay-20260828_20260828_115632_behavior_snapshot.pt",
            )
        ).exists(),
        "saved 392-tensor behavior snapshot is not available",
    )
    def test_saved_392_lora_tensors_restore_and_hash_exactly(self):
        source = Path(
            os.getenv(
                "AGENTFLOW_BEHAVIOR_SNAPSHOT_TEST_PATH",
                "/root/autodl-tmp/tmp/random30_fresh_rollout_replay_20260828/"
                "random30-fresh-rollout-replay-20260828_20260828_115632_behavior_snapshot.pt",
            )
        )
        payload = torch.load(source, map_location="cpu", weights_only=False)
        state = payload["lora_state"]
        self.assertEqual(len(state), 392)

        # Recreate only the PEFT parameter tree on CPU.  This exercises the
        # same pre-FSDP restore function without loading the 7B base model.
        module = nn.Module()
        for name, tensor in state.items():
            parent = module
            parts = name.split(".")
            for part in parts[:-1]:
                child = parent._modules.get(part)
                if child is None:
                    child = nn.Module()
                    parent.add_module(part, child)
                parent = child
            parent.register_parameter(parts[-1], nn.Parameter(torch.zeros_like(tensor)))

        with patch.dict(os.environ, {"AGENTFLOW_BEHAVIOR_SNAPSHOT_SOURCE_PATH": str(source)}, clear=False):
            result = capture.restore_behavior_snapshot(module)
        self.assertEqual(result["status"], "restored")
        self.assertEqual(result["tensor_count"], 392)
        self.assertEqual(result["lora_hash"], payload["lora_hash"])
        self.assertEqual(capture._lora_hash(module)["hash"], payload["lora_hash"])


if __name__ == "__main__":
    unittest.main()
