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


if __name__ == "__main__":
    unittest.main()

