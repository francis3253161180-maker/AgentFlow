import unittest


class HierarchicalPlanStateTest(unittest.TestCase):
    def test_high_level_plan_rejects_empty_steps(self):
        from agentflow.models.formatters import HighLevelPlan

        with self.assertRaises(Exception):
            HighLevelPlan(steps=[])

    def _plan(self):
        from agentflow.models.plan_state import normalize_plan

        return normalize_plan({
            "steps": [
                {
                    "step_id": "identify",
                    "objective": "Identify the required entity",
                    "success_criteria": "Memory names the entity with supporting evidence",
                    "depends_on": [],
                },
                {
                    "step_id": "derive",
                    "objective": "Derive the requested value",
                    "success_criteria": "Memory supports the value using the identified entity",
                    "depends_on": ["identify"],
                },
            ]
        }, 3)

    def test_failed_step_remains_current_then_completed_step_advances(self):
        from agentflow.models.plan_state import activate_next_step, apply_step_verification

        plan, transitions = self._plan(), []
        active = activate_next_step(plan, transitions)
        self.assertEqual(active["step_id"], "identify")
        apply_step_verification(plan, "identify", {
            "completed": False, "missing_evidence": ["entity"], "verified_evidence": [],
            "contradiction": False, "invalidated_step_ids": [], "rationale": "not in memory",
        }, transitions)
        self.assertEqual(activate_next_step(plan, transitions)["step_id"], "identify")

        apply_step_verification(plan, "identify", {
            "completed": True, "missing_evidence": [], "verified_evidence": ["entity evidence"],
            "contradiction": False, "invalidated_step_ids": [], "rationale": "supported",
        }, transitions)
        self.assertEqual(activate_next_step(plan, transitions)["step_id"], "derive")
        self.assertEqual(plan["steps"][0]["status"], "completed")

    def test_completed_step_reopens_only_on_explicit_contradiction(self):
        from agentflow.models.plan_state import activate_next_step, apply_step_verification

        plan, transitions = self._plan(), []
        activate_next_step(plan, transitions)
        apply_step_verification(plan, "identify", {
            "completed": True, "missing_evidence": [], "verified_evidence": ["entity evidence"],
            "contradiction": False, "invalidated_step_ids": [], "rationale": "supported",
        }, transitions)
        active = activate_next_step(plan, transitions)
        apply_step_verification(plan, active["step_id"], {
            "completed": False, "missing_evidence": ["value"], "verified_evidence": [],
            "contradiction": True, "invalidated_step_ids": ["identify"], "rationale": "conflicting entity evidence",
        }, transitions)
        self.assertEqual(plan["steps"][0]["status"], "pending")
        self.assertTrue(any(item["event"] == "reopen" for item in transitions))

    def test_planner_prompt_scopes_main_policy_to_current_step(self):
        from unittest.mock import MagicMock
        from agentflow.models.formatters import NextStep
        from agentflow.models.memory import Memory
        from agentflow.models.planner import Planner

        planner = object.__new__(Planner)
        planner.is_multimodal = False
        planner.available_tools = ["Wikipedia_RAG_Search_Tool", "Web_RAG_Search_Tool"]
        planner.toolbox_metadata = {}
        planner.max_tokens = 32
        planner.llm_engine = MagicMock(return_value=NextStep(
            justification="j", context="c", sub_goal="s", tool_name="Wikipedia_RAG_Search_Tool",
        ))
        planner.generate_next_step(
            "question", "", "analysis", Memory(), 1, 3,
            hierarchical_plan={"steps": [{"step_id": "identify", "status": "in_progress"}]},
            current_step={"step_id": "identify", "objective": "Identify one entity"},
        )
        prompt = planner.llm_engine.call_args.args[0]
        self.assertIn("Hierarchical Plan State", prompt)
        self.assertIn("Current unresolved step", prompt)
        self.assertIn("Unresolved evidence gaps", prompt)
        self.assertIn("one atomic sub-goal", prompt)
        self.assertIn("do not redo a completed step", prompt)

    def test_high_level_plan_text_is_parsed_before_state_machine(self):
        from unittest.mock import MagicMock
        from agentflow.models.planner import Planner

        planner = object.__new__(Planner)
        planner.max_tokens = 64
        planner.available_tools = ["Wikipedia_RAG_Search_Tool"]
        planner.llm_engine_fixed = MagicMock(return_value=(
            '{"steps":[{"step_id":"discover","objective":"Find one fact",'
            '"success_criteria":"Memory contains cited fact"}]}'
        ))
        plan = planner.generate_high_level_plan("q", "", "a", 3, {})
        self.assertEqual(plan.steps[0].step_id, "discover")


if __name__ == "__main__":
    unittest.main()
