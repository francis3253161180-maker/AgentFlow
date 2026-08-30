import importlib.util
import sys
import unittest
from pathlib import Path

import torch


SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_offline_musique_zero_lora_proof_20260830.py"
SPEC = importlib.util.spec_from_file_location("zero_lora_proof", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ZeroLoRAProofTest(unittest.TestCase):
    def test_descriptor_is_deterministic(self):
        tensor = torch.tensor([[1.0, 2.0]])
        self.assertEqual(MODULE.tensor_descriptor("x", tensor), MODULE.tensor_descriptor("x", tensor.clone()))


if __name__ == "__main__":
    unittest.main()
