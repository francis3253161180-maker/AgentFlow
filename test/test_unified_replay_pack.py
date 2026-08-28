import json
import importlib.util
import tempfile
import unittest
from pathlib import Path


_SCRIPT = Path(__file__).parents[1] / "scripts" / "export_unified_replay_pack_20260828.py"
_SPEC = importlib.util.spec_from_file_location("unified_replay_pack", _SCRIPT)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
dry_run = _MODULE.dry_run


class ReplayPackDryRunTest(unittest.TestCase):
    def test_dry_run_accepts_immutable_pack_shape(self):
        pack = {
            "kind": "agentflow_unified_pre_update_replay_pack",
            "source": {"pre_update_step": "1"},
            "records": [
                {
                    "prompt": "p",
                    "groundtruth": "g",
                    "reward": 0.0,
                    "trajectory": {"events": []},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pack.json"
            path.write_text(json.dumps(pack), encoding="utf-8")
            self.assertEqual(dry_run(path)["status"], "ok")


if __name__ == "__main__":
    unittest.main()
