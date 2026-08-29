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
        self.assertEqual(planner.llm_engine.call_args.kwargs["response_format"].__name__, "NextStep")
        self.assertIn("Hierarchical Plan State", prompt)
        self.assertIn("Current-Step State Contract", prompt)
        self.assertIn("stable_step_id", prompt)
        self.assertNotIn("unresolved_evidence_gaps", prompt)
        self.assertNotIn('"target_gap"', prompt)
        self.assertIn("one atomic sub-goal", prompt)
        self.assertIn("redo a completed step", prompt)

    def test_current_step_contract_and_stagnation_guard_are_generic(self):
        from agentflow.models.current_step_state import (
            build_current_step_contract,
            should_revise_stagnant_action,
            executable_signature,
            stable_step_id,
        )

        step = {
            "step_id": "discover", "objective": "Find a relation",
            "success_criteria": "Memory supports the relation",
            "verified_evidence": [], "missing_evidence": ["the entity relation"],
        }
        contract = build_current_step_contract(
            step, {"Action Step 1": {"result": "https://example.test/a"}}, [], {"completed": False},
        )
        stable_id = stable_step_id(step)
        self.assertEqual(contract["stable_step_id"], stable_id)
        self.assertEqual(contract["missing_evidence_diagnostics"], ["the entity relation"])
        signature = executable_signature(
            "Wikipedia_RAG_Search_Tool", 'execution = tool.execute(query="entity relation")', "", "",
        )
        repeated, _ = should_revise_stagnant_action(
            tool_name="Wikipedia_RAG_Search_Tool", stable_step_id=stable_id,
            executable_signature=signature,
            prior_attempts=[{
                "tool_name": "Wikipedia_RAG_Search_Tool", "stable_step_id": stable_id,
                "executable_signature": signature, "made_progress": False,
            }],
        )
        self.assertTrue(repeated)
        changed, _ = should_revise_stagnant_action(
            tool_name="Wikipedia_RAG_Search_Tool", stable_step_id=stable_id,
            executable_signature=executable_signature(
                "Wikipedia_RAG_Search_Tool", 'execution = tool.execute(query="relation founding source")', "", "",
            ),
            prior_attempts=[{
                "tool_name": "Wikipedia_RAG_Search_Tool", "stable_step_id": stable_id,
                "executable_signature": signature, "made_progress": False,
            }],
        )
        self.assertFalse(changed)
        cosmetic, _ = should_revise_stagnant_action(
            tool_name="Wikipedia_RAG_Search_Tool", stable_step_id=stable_id,
            executable_signature=executable_signature(
                "Wikipedia_RAG_Search_Tool",
                'execution = tool.execute(query="entity relation in 1948 and 1949")', "", "",
            ),
            prior_attempts=[{
                "tool_name": "Wikipedia_RAG_Search_Tool", "stable_step_id": stable_id,
                "executable_signature": executable_signature(
                    "Wikipedia_RAG_Search_Tool",
                    'execution = tool.execute(query="entity relation 1948 1949")', "", "",
                ),
                "made_progress": False,
            }],
        )
        self.assertTrue(cosmetic)
        new_season, _ = should_revise_stagnant_action(
            tool_name="Wikipedia_RAG_Search_Tool", stable_step_id=stable_id,
            executable_signature=executable_signature(
                "Wikipedia_RAG_Search_Tool",
                'execution = tool.execute(query="entity relation 1950 1951")', "", "",
            ),
            prior_attempts=[{
                "tool_name": "Wikipedia_RAG_Search_Tool", "stable_step_id": stable_id,
                "executable_signature": signature, "made_progress": False,
            }],
        )
        self.assertFalse(new_season)

    def test_missing_evidence_wording_churn_is_not_progress(self):
        from agentflow.models.current_step_state import assess_step_progress

        before = {"status": "in_progress", "verified_evidence": [], "missing_evidence": ["name the league"]}
        after = {"status": "in_progress", "verified_evidence": [], "missing_evidence": ["specific competition name"]}
        progress = assess_step_progress(before, after)
        self.assertTrue(progress["missing_evidence_changed"])
        self.assertFalse(progress["made_progress"])

    def test_high_level_plan_text_is_parsed_before_state_machine(self):
        from unittest.mock import MagicMock
        from agentflow.models.planner import Planner

        planner = object.__new__(Planner)
        planner.max_tokens = 64
        planner.available_tools = ["Wikipedia_RAG_Search_Tool"]
        planner.llm_engine_fixed = MagicMock(side_effect=[
            '{"steps":[{"step_id":"discover","objective":"Find one fact",'
            '"success_criteria":"Memory contains cited fact"}]}',
            '{"sufficient":true,"independently_necessary_requirements":["one fact"],'
            '"requirement_coverage":[{"requirement":"one fact","covered_step_ids":["discover"]}],'
            '"covered_step_ids":["discover"],"missing_requirements":[],"composite_step_ids":[],"rationale":"atomic"}',
        ])
        data = {}
        plan = planner.generate_high_level_plan("q", "", "a", 3, data)
        self.assertEqual(plan.steps[0].step_id, "discover")
        self.assertTrue(data["high_level_plan_coverage_valid"])
        self.assertEqual(planner.llm_engine_fixed.call_count, 2)

    def test_incomplete_coverage_gets_one_fixed_role_plan_revision(self):
        from unittest.mock import MagicMock
        from agentflow.models.planner import Planner

        planner = object.__new__(Planner)
        planner.max_tokens = 64
        planner.available_tools = ["Wikipedia_RAG_Search_Tool"]
        planner.llm_engine_fixed = MagicMock(side_effect=[
            '{"steps":[{"step_id":"first","objective":"Find one fact",'
            '"success_criteria":"Memory contains fact"}]}',
            '{"sufficient":false,"independently_necessary_requirements":["fact one","fact two"],'
            '"requirement_coverage":[{"requirement":"fact one","covered_step_ids":["first"]},{"requirement":"fact two","covered_step_ids":[]}],'
            '"covered_step_ids":["first"],"missing_requirements":["fact two"],"composite_step_ids":[],"rationale":"missing"}',
            '{"steps":[{"step_id":"first","objective":"Find fact one","success_criteria":"fact one",'
            '"depends_on":[]},{"step_id":"second","objective":"Find fact two","success_criteria":"fact two",'
            '"depends_on":["first"]}]}',
            '{"sufficient":true,"independently_necessary_requirements":["fact one","fact two"],'
            '"requirement_coverage":[{"requirement":"fact one","covered_step_ids":["first"]},{"requirement":"fact two","covered_step_ids":["second"]}],'
            '"covered_step_ids":["first","second"],"missing_requirements":[],"composite_step_ids":[],"rationale":"complete"}',
        ])
        data = {}
        plan = planner.generate_high_level_plan("q", "", "a", 3, data)
        self.assertEqual([step.step_id for step in plan.steps], ["first", "second"])
        self.assertIsNotNone(data["high_level_plan_revised"])
        self.assertTrue(data["high_level_plan_coverage_valid"])
        self.assertEqual(planner.llm_engine_fixed.call_count, 4)

    def test_coverage_verdict_cannot_contradict_requirement_mapping(self):
        from agentflow.models.formatters import PlanCoverage, RequirementCoverage
        from agentflow.models.planner import Planner

        plan = {"steps": [{"step_id": "one"}]}
        contradictory = PlanCoverage(
            sufficient=True,
            independently_necessary_requirements=["fact one", "fact two"],
            requirement_coverage=[
                RequirementCoverage(requirement="fact one", covered_step_ids=["one"]),
                RequirementCoverage(requirement="fact two", covered_step_ids=[]),
            ],
            covered_step_ids=["one"],
            missing_requirements=[], composite_step_ids=[], rationale="claims sufficient",
        )
        self.assertFalse(Planner._coverage_is_sufficient(plan, contradictory))


if __name__ == "__main__":
    unittest.main()
