import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agentflow.engine.vllm import ChatVLLM
from agentflow.models.executor import Executor
from agentflow.models.formatters import MemoryVerification, NextStep, ToolCommand
from agentflow.models.planner import Planner
from agentflow.models.verifier import Verifier
from agentflow.runner import filter_trainable_triplets
from agentflow.types import Triplet

sys.path.insert(0, str(Path(__file__).parents[1] / "train"))
from train.rollout import Rollout


def _fake_client(content="ok"):
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )
    return client


class FollowupGithubAuditFixTest(unittest.TestCase):
    def test_vllm_constructor_defaults_are_sent_and_call_override_wins(self):
        client = _fake_client()
        with patch("agentflow.engine.vllm.OpenAI", return_value=client):
            fixed = ChatVLLM(
                model_string="qwen-base",
                use_cache=False,
                temperature=0.0,
                top_p=0.8,
                frequency_penalty=0.25,
                max_tokens=123,
            )
            fixed.generate("fixed")
            kwargs = client.chat.completions.create.call_args.kwargs
            self.assertEqual(kwargs["temperature"], 0.0)
            self.assertEqual(kwargs["top_p"], 0.8)
            self.assertEqual(kwargs["frequency_penalty"], 0.25)
            self.assertEqual(kwargs["max_tokens"], 123)

            client.chat.completions.create.reset_mock()
            fixed.generate(
                "planner override",
                temperature=0.7,
                top_p=0.95,
                frequency_penalty=0.0,
                max_tokens=456,
            )
            kwargs = client.chat.completions.create.call_args.kwargs
            self.assertEqual(kwargs["temperature"], 0.7)
            self.assertEqual(kwargs["top_p"], 0.95)
            self.assertEqual(kwargs["frequency_penalty"], 0.0)
            self.assertEqual(kwargs["max_tokens"], 456)

    def test_vllm_guided_json_is_preserved_with_defaults(self):
        client = _fake_client('{"expression":"1+2+3"}')
        with patch("agentflow.engine.vllm.OpenAI", return_value=client):
            engine = ChatVLLM("qwen-base", use_cache=False, temperature=0.0)
            engine.generate("structured", response_format={"type": "object"})
        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["temperature"], 0.0)
        self.assertEqual(request["extra_body"], {"guided_json": {"type": "object"}})

    def test_agent_exception_propagates_before_reward_evaluation(self):
        owner = object.__new__(Rollout)
        failing_agent = SimpleNamespace(
            llm_engine="qwen-actor",
            solve=MagicMock(side_effect=RuntimeError("vllm unavailable")),
        )
        task = {"question": "q", "result": "a"}
        with self.assertRaisesRegex(RuntimeError, "vllm unavailable"):
            asyncio.run(owner._solve_and_evaluate(failing_agent, task, 0))

    def test_non_unified_filter_discards_empty_token_spans_but_keeps_valid_path(self):
        valid = Triplet(prompt={"token_ids": [1]}, response={"token_ids": [2]})
        empty_fixed = Triplet(prompt={"token_ids": []}, response={"token_ids": []})
        kept, stats = filter_trainable_triplets([empty_fixed, valid], unified_local=False)
        self.assertEqual(kept, [valid])
        self.assertEqual(stats["excluded_invalid"], 1)

    def _planner_for_prompt(self, multimodal):
        planner = object.__new__(Planner)
        planner.is_multimodal = multimodal
        planner.available_tools = ["tool"]
        planner.toolbox_metadata = {"tool": "metadata"}
        planner.max_tokens = 32
        planner.llm_engine = MagicMock(
            return_value=NextStep(justification="j", context="c", sub_goal="s", tool_name="tool")
        )
        return planner

    def test_planner_structured_prompt_names_all_schema_fields(self):
        memory = MagicMock()
        memory.get_actions.return_value = "previous"
        for multimodal in (False, True):
            planner = self._planner_for_prompt(multimodal)
            planner.generate_next_step("question", "", "analysis", memory, 0, 2)
            prompt = planner.llm_engine.call_args.args[0]
            for field in ('"justification"', '"context"', '"sub_goal"', '"tool_name"'):
                self.assertIn(field, prompt)
            self.assertNotIn("Generated Command:", prompt)

    def _verifier_for_prompt(self, multimodal):
        verifier = object.__new__(Verifier)
        verifier.is_multimodal = multimodal
        verifier.available_tools = ["tool"]
        verifier.toolbox_metadata = {}
        verifier.llm_engine_fixed = MagicMock(return_value=MemoryVerification(analysis="a", stop_signal=True))
        return verifier

    def test_verifier_structured_prompt_removes_legacy_conclusion_protocol(self):
        memory = MagicMock()
        memory.get_actions.return_value = "memory"
        for multimodal in (False, True):
            verifier = self._verifier_for_prompt(multimodal)
            verifier.verificate_context("q", "", "analysis", memory)
            prompt = verifier.llm_engine_fixed.call_args.args[0][0]
            self.assertIn('"analysis"', prompt)
            self.assertIn('"stop_signal"', prompt)
            self.assertNotIn("Conclusion: STOP", prompt)
            self.assertNotIn("Conclusion: CONTINUE", prompt)

    def test_executor_structured_prompt_names_tool_command_schema(self):
        executor = object.__new__(Executor)
        executor.llm_generate_tool_command = MagicMock(
            return_value=ToolCommand(analysis="a", explanation="e", command="execution = tool.execute()")
        )
        executor.generate_tool_command("q", "", "context", "goal", "tool", {})
        prompt = executor.llm_generate_tool_command.call_args.args[0]
        for field in ('"analysis"', '"explanation"', '"command"'):
            self.assertIn(field, prompt)
        self.assertNotIn("Generated Command:", prompt)
        self.assertNotIn("```python", prompt)


if __name__ == "__main__":
    unittest.main()
