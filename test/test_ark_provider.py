import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch


class ArkProviderTest(unittest.TestCase):
    def test_external_guard_blocks_doubao_without_network(self):
        from agentflow.engine import factory

        with patch.dict(os.environ, {"AGENTFLOW_DISABLE_EXTERNAL_LLM": "1"}):
            with self.assertRaises(RuntimeError):
                factory.create_llm_engine("doubao-seed-2-0-lite-260428")

    def test_ark_request_uses_exact_model_and_json_hint(self):
        from agentflow.engine.ark import ChatArk

        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"expression":"1"}'))]
        )
        calls = []

        def create(**kwargs):
            calls.append(kwargs)
            return response

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        with patch.dict(
            os.environ,
            {"ARK_API_KEY": "test-only", "ARK_BASE_URL": "http://example.invalid/v1"},
            clear=False,
        ), patch("agentflow.engine.ark.OpenAI", return_value=client) as openai:
            engine = ChatArk("doubao-seed-2-0-lite-260428", temperature=0.0, max_tokens=8)
            self.assertEqual(engine("short prompt", response_format=object), '{"expression":"1"}')
            self.assertEqual(openai.call_args.kwargs["base_url"], "http://example.invalid/v1")
            self.assertEqual(engine.api_model, "doubao-seed-2-0-lite-260428")
            self.assertNotIn("short prompt", engine._cache_key("system", "short prompt"))
            self.assertEqual(calls[0]["model"], "doubao-seed-2-0-lite-260428")
            self.assertEqual(calls[0]["temperature"], 0.0)
            self.assertEqual(calls[0]["response_format"], {"type": "json_object"})


if __name__ == "__main__":
    unittest.main()
