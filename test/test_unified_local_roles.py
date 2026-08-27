import os
import unittest
from unittest.mock import patch


class _FakeEngine:
    def __init__(self, model_string, **kwargs):
        self.model_string = model_string
        self.kwargs = kwargs


class UnifiedLocalRoleWiringTest(unittest.TestCase):
    def test_planner_and_verifier_pass_local_endpoint_to_fixed_role(self):
        from agentflow.models.planner import Planner
        from agentflow.models.verifier import Verifier

        with patch("agentflow.models.planner.create_llm_engine", side_effect=_FakeEngine) as planner_factory:
            Planner("vllm-models/Qwen2.5-7B-Instruct", "vllm-models/Qwen2.5-7B-Instruct", fixed_base_url="http://127.0.0.1:1/v1", base_url="http://127.0.0.1:1/v1")
            self.assertEqual(planner_factory.call_args_list[0].kwargs["base_url"], "http://127.0.0.1:1/v1")
            self.assertEqual(planner_factory.call_args_list[1].kwargs["base_url"], "http://127.0.0.1:1/v1")

        with patch("agentflow.models.verifier.create_llm_engine", side_effect=_FakeEngine) as verifier_factory:
            Verifier("vllm-models/Qwen2.5-7B-Instruct", "vllm-models/Qwen2.5-7B-Instruct", fixed_base_url="http://127.0.0.1:1/v1", base_url="http://127.0.0.1:1/v1")
            self.assertEqual(verifier_factory.call_args_list[0].kwargs["base_url"], "http://127.0.0.1:1/v1")
            self.assertEqual(verifier_factory.call_args_list[1].kwargs["base_url"], "http://127.0.0.1:1/v1")

    def test_self_tool_passes_local_endpoint(self):
        from agentflow.tools.base_generator.tool import Base_Generator_Tool

        with patch("agentflow.tools.base_generator.tool.create_llm_engine", side_effect=_FakeEngine) as factory:
            Base_Generator_Tool("vllm-models/Qwen2.5-7B-Instruct", base_url="http://127.0.0.1:1/v1")
            self.assertEqual(factory.call_args.kwargs["base_url"], "http://127.0.0.1:1/v1")

    def test_unified_role_contract_is_trainable_main_and_frozen_fixed(self):
        from agentflow.solver import construct_solver

        with patch("agentflow.solver.Initializer") as initializer, patch("agentflow.solver.Planner"), patch("agentflow.solver.Verifier"), patch("agentflow.solver.Executor"):
            initializer.return_value.toolbox_metadata = {}
            initializer.return_value.available_tools = []
            initializer.return_value.tool_instances_cache = {}
            construct_solver(
                llm_engine_name="vllm-models/Qwen2.5-7B-Instruct",
                enabled_tools=[],
                tool_engine=[],
                model_engine=["trainable", "frozen", "frozen", "frozen"],
                base_url="http://127.0.0.1:1/v1",
            )

    def test_external_guard_is_opt_in(self):
        from agentflow.engine import factory

        with patch.dict(os.environ, {"AGENTFLOW_DISABLE_EXTERNAL_LLM": "1"}):
            with self.assertRaises(RuntimeError):
                factory.create_llm_engine("deepseek-v4-flash")


if __name__ == "__main__":
    unittest.main()
