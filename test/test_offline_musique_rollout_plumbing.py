import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from test.test_offline_musique import tiny_corpus


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_offline_musique_protocol_smoke_20260830.py"
SPEC = importlib.util.spec_from_file_location("offline_musique_runner", SCRIPT)
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


class FakeLLM:
    def generate(self, **_kwargs):
        generated = SimpleNamespace(
            text="ok",
            token_ids=[11, 12],
            logprobs=[
                {11: SimpleNamespace(logprob=-0.25)},
                {12: SimpleNamespace(logprob=-0.75)},
            ],
            cumulative_logprob=-1.0,
            finish_reason="stop",
        )
        return [SimpleNamespace(prompt_token_ids=[1, 2], outputs=[generated])]


class OfflineMusiqueRolloutPlumbingTest(unittest.TestCase):
    def test_generation_persists_selected_token_ids_and_old_logprobs(self):
        texts, metadata = RUNNER.generate_batch(FakeLLM(), [[1, 2]], object(), object())
        self.assertEqual(texts, ["ok"])
        self.assertEqual(metadata[0]["response_token_ids"], [11, 12])
        self.assertEqual(metadata[0]["response_token_logprobs"], [-0.25, -0.75])
        self.assertEqual(metadata[0]["selected_vs_cumulative_logprob_delta"], 0.0)

    def test_frozen_qid_file_is_exact_and_ordered(self):
        corpus, _ = tiny_corpus()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qids.json"
            path.write_text(json.dumps({"eval": {"qids": ["q1"]}}), encoding="utf-8")
            self.assertEqual(RUNNER.resolve_qids(corpus, 1, 1, path, "eval"), ["q1"])

    def test_frozen_qid_file_fails_closed_on_wrong_size(self):
        corpus, _ = tiny_corpus()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qids.json"
            path.write_text(json.dumps({"eval": {"qids": ["q1"]}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly 2 unique"):
                RUNNER.resolve_qids(corpus, 2, 1, path, "eval")


if __name__ == "__main__":
    unittest.main()
