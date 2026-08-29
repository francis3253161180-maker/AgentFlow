import argparse
import copy
import time
import json
import os
from typing import Optional

from agentflow.models.initializer import Initializer
from agentflow.models.planner import Planner
from agentflow.models.verifier import Verifier
from agentflow.models.memory import Memory
from agentflow.models.current_step_state import (
    assess_step_progress,
    build_current_step_contract,
    executable_signature,
    should_revise_stagnant_action,
    stable_step_id,
)
from agentflow.models.plan_state import (
    activate_next_step,
    all_steps_completed,
    apply_step_verification,
    attach_requirement_coverage,
    normalize_plan,
    snapshot as plan_snapshot,
    unresolved_steps,
    validate_grounded_step_completion,
)
from agentflow.models.executor import Executor
from agentflow.models.utils import make_json_serializable_truncated

class Solver:
    def __init__(
        self,
        planner,
        verifier,
        memory,
        executor,
        output_types: str = "base,final,direct",
        max_steps: int = 10,
        max_time: int = 300,
        max_tokens: int = 4000,
        root_cache_dir: str = "cache",
        verbose: bool = True,
        temperature: float = .0
    ):
        self.planner = planner
        self.verifier = verifier
        self.memory = memory
        self.executor = executor
        self.max_steps = max_steps
        self.max_time = max_time
        self.max_tokens = max_tokens
        self.root_cache_dir = root_cache_dir

        self.output_types = output_types.lower().split(',')
        self.temperature  = temperature
        assert all(output_type in ["base", "final", "direct"] for output_type in self.output_types), "Invalid output type. Supported types are 'base', 'final', 'direct'."
        self.verbose = verbose
    def solve(self, question: str, image_path: Optional[str] = None):
        """
        Solve a single problem from the benchmark dataset.
        
        Args:
            index (int): Index of the problem to solve
        """
        # A Solver instance can serve multiple rollout tasks in one worker.
        # Do not leak files/actions from the previous query into this one.
        self.memory.reset()

        # Update cache directory for the executor
        self.executor.set_query_cache_dir(self.root_cache_dir)

        # Initialize json_data with basic problem information
        json_data = {
            "query": question,
            "image": image_path
        }
        if self.verbose:
            print(f"\n==> 🔍 Received Query: {question}")
            if image_path:
                print(f"\n==> 🖼️ Received Image: {image_path}")

        # Generate base response if requested
        if 'base' in self.output_types:
            base_response = self.planner.generate_base_response(question, image_path, self.max_tokens)
            json_data["base_response"] = base_response
            if self.verbose:
                print(f"\n==> 📝 Base Response from LLM:\n\n{base_response}")

        # If only base response is needed, save and return
        if set(self.output_types) == {'base'}:
            return json_data
    
        # Continue with query analysis and tool execution if final or direct responses are needed
        if {'final', 'direct'} & set(self.output_types):
            if self.verbose:
                print(f"\n==> 🐙 Reasoning Steps from AgentFlow (Deep Thinking...)")

            # [1] Analyze query
            query_start_time = time.time()
            query_analysis = self.planner.analyze_query(question, image_path)
            json_data["query_analysis"] = query_analysis
            if self.verbose:
                print(f"\n==> 🔍 Step 0: Query Analysis\n")
                print(f"{query_analysis}")
                print(f"[Time]: {round(time.time() - query_start_time, 2)}s")

            hierarchical_planning = os.getenv("AGENTFLOW_HIERARCHICAL_PLANNING", "0").lower() in {
                "1", "true", "yes", "on"
            }
            plan_state = None
            plan_transitions = []
            termination_reason = None
            if hierarchical_planning:
                high_level_plan = self.planner.generate_high_level_plan(
                    question, image_path, query_analysis, self.max_steps, json_data,
                )
                plan_state = normalize_plan(high_level_plan, self.max_steps)
                activate_next_step(plan_state, plan_transitions)
                json_data["hierarchical_planning"] = True
                coverage_state = getattr(self.planner, "last_high_level_plan_coverage", None)
                if isinstance(coverage_state, dict):
                    requirement_mapping = attach_requirement_coverage(
                        plan_state, coverage_state.get("coverage", {}),
                    )
                    json_data["requirement_to_step_mapping"] = copy.deepcopy(requirement_mapping)
                json_data["high_level_plan"] = plan_snapshot(plan_state)
                json_data["plan_transitions"] = copy.deepcopy(plan_transitions)
                if isinstance(coverage_state, dict) and not bool(coverage_state.get("valid", False)):
                    # Do not execute a known incomplete/composite evidence
                    # plan and then allow final synthesis to fill its gaps.
                    # The fixed planner already received exactly one revision.
                    termination_reason = "high_level_plan_coverage_invalid"
                if self.verbose:
                    print(f"\n==> 🧭 High-Level Plan\n{plan_state}")

            # Main execution loop
            step_count = 0
            action_times = []
            # Preserve a defined serializable value even if no plan step can
            # become active; normal execution overwrites it after each action.
            memory_actions = self.memory.get_actions()
            previous_verifier_assessment = None
            attempts_by_step: dict[str, list[dict]] = {}
            while (
                termination_reason is None
                and step_count < self.max_steps
                and (time.time() - query_start_time) < self.max_time
            ):
                current_plan_step = None
                if hierarchical_planning:
                    current_plan_step = activate_next_step(plan_state, plan_transitions)
                    if current_plan_step is None:
                        termination_reason = (
                            "all_plan_steps_completed" if all_steps_completed(plan_state)
                            else "no_dependency_satisfied_plan_step"
                        )
                        break
                    json_data[f"plan_before_step_{step_count + 1}"] = plan_snapshot(plan_state)
                step_count += 1
                step_start_time = time.time()

                # [2] Generate next step
                local_start_time = time.time()
                current_step_contract = None
                prior_attempts: list[dict] = []
                if hierarchical_planning:
                    prior_attempts = attempts_by_step.setdefault(current_plan_step["step_id"], [])
                    current_step_contract = build_current_step_contract(
                        current_plan_step, self.memory.get_actions(), prior_attempts,
                        previous_verifier_assessment, plan_state,
                    )
                    json_data[f"current_step_state_{step_count}"] = copy.deepcopy(current_step_contract)
                next_step = self.planner.generate_next_step(
                    question, 
                    image_path, 
                    query_analysis, 
                    self.memory, 
                    step_count,
                    self.max_steps,
                    json_data,
                    previous_verifier_assessment=previous_verifier_assessment,
                    hierarchical_plan=plan_state,
                    current_step=current_plan_step,
                    current_step_contract=current_step_contract,
                )
                context, sub_goal, tool_name = self.planner.extract_context_subgoal_and_tool(next_step)
                active_step_id = stable_step_id(current_plan_step) if hierarchical_planning else ""
                command = ""
                if hierarchical_planning:
                    json_data[f"action_stable_step_id_{step_count}"] = active_step_id
                if self.verbose:
                    print(f"\n==> 🎯 Step {step_count}: Action Prediction ({tool_name})\n")
                    print(f"[Context]: {context}\n[Stable Step ID]: {active_step_id}\n[Sub Goal]: {sub_goal}\n[Tool]: {tool_name}")
                    print(f"[Time]: {round(time.time() - local_start_time, 2)}s")

                if tool_name is None or tool_name not in self.planner.available_tools:
                    print(f"\n==> 🚫 Error: Tool '{tool_name}' is not available or not found.")
                    command = "No command was generated because the tool was not found."
                    result = "No result was generated because the tool was not found."

                else:
                    # [3] Generate the tool command
                    def generate_command(record_label: str):
                        tool_command = self.executor.generate_tool_command(
                            question, image_path, context, sub_goal, tool_name,
                            self.planner.toolbox_metadata[tool_name], step_count, json_data,
                            record_label=record_label,
                        )
                        return self.executor.extract_explanation_and_command(tool_command)

                    local_start_time = time.time()
                    analysis, explanation, command = generate_command("initial")
                    command_signature = executable_signature(tool_name, command, context, sub_goal)
                    if hierarchical_planning:
                        rejected, rejection_reason = should_revise_stagnant_action(
                            tool_name=tool_name, stable_step_id=active_step_id,
                            executable_signature=command_signature, prior_attempts=prior_attempts,
                        )
                        if rejected:
                            json_data[f"action_revision_{step_count}"] = {
                                "rejected": True, "reason": rejection_reason,
                                "stable_step_id": active_step_id,
                                "original": {
                                    "tool_name": tool_name, "sub_goal": sub_goal, "context": context,
                                    "command": command, "executable_signature": command_signature,
                                },
                            }
                            next_step = self.planner.generate_next_step(
                                question, image_path, query_analysis, self.memory, step_count,
                                self.max_steps, json_data,
                                previous_verifier_assessment=previous_verifier_assessment,
                                hierarchical_plan=plan_state, current_step=current_plan_step,
                                current_step_contract=current_step_contract,
                                action_rejection=rejection_reason, attempt_label="revision",
                            )
                            context, sub_goal, tool_name = self.planner.extract_context_subgoal_and_tool(next_step)
                            analysis, explanation, command = generate_command("revision")
                            command_signature = executable_signature(tool_name, command, context, sub_goal)
                            json_data[f"action_revision_{step_count}"]["revised"] = {
                                "tool_name": tool_name, "sub_goal": sub_goal, "context": context,
                                "command": command, "executable_signature": command_signature,
                            }
                            still_stagnant, _ = should_revise_stagnant_action(
                                tool_name=tool_name, stable_step_id=active_step_id,
                                executable_signature=command_signature, prior_attempts=prior_attempts,
                            )
                            if still_stagnant:
                                json_data[f"planner_action_stagnant_{step_count}"] = {
                                    "reason": "revised action retained the same executable intent after no progress",
                                    "stable_step_id": active_step_id,
                                    "tool_name": tool_name, "sub_goal": sub_goal, "context": context,
                                    "command": command, "executable_signature": command_signature,
                                }
                                termination_reason = "planner_action_stagnant"
                                break
                    if self.verbose:
                        print(f"\n==> 📝 Step {step_count}: Command Generation ({tool_name})\n")
                        print(f"[Analysis]: {analysis}\n[Explanation]: {explanation}\n[Command]: {command}")
                        print(f"[Time]: {round(time.time() - local_start_time, 2)}s")
                    
                    # [4] Execute the tool command
                    local_start_time = time.time()
                    result = self.executor.execute_tool_command(tool_name, command)
                    result = make_json_serializable_truncated(result) # Convert to JSON serializable format
                    json_data[f"tool_result_{step_count}"] = result

                    if self.verbose:
                        print(f"\n==> 🛠️ Step {step_count}: Command Execution ({tool_name})\n")
                        print(f"[Result]:\n{json.dumps(result, indent=4)}")
                        print(f"[Time]: {round(time.time() - local_start_time, 2)}s")
                
                # Track execution time for the current step
                execution_time_step = round(time.time() - step_start_time, 2)
                action_times.append(execution_time_step)

                # Update memory
                self.memory.add_action(step_count, tool_name, sub_goal, command, result)
                memory_actions = self.memory.get_actions()

                # [5] Verify either the active evidence step or legacy global memory.
                local_start_time = time.time()
                if hierarchical_planning:
                    before_current_step = copy.deepcopy(current_plan_step)
                    raw_step_verification = self.verifier.verificate_step(
                        question, image_path, query_analysis, self.memory,
                        current_plan_step, plan_state, step_count, json_data,
                    )
                    step_verification = self.verifier.extract_step_verification(raw_step_verification)
                    verification_payload = (
                        step_verification.model_dump()
                        if hasattr(step_verification, "model_dump") else step_verification.dict()
                    )
                    completion_gate = validate_grounded_step_completion(
                        verification_payload, current_plan_step["step_id"], plan_state, memory_actions,
                    )
                    if bool(verification_payload.get("completed")) and not completion_gate["accepted"]:
                        verification_payload["completed"] = False
                        verification_payload["completion_rejected"] = completion_gate
                        missing = list(verification_payload.get("missing_evidence", []))
                        missing.append("traceable provenance is required for every mapped active-step requirement")
                        verification_payload["missing_evidence"] = list(dict.fromkeys(missing))
                    json_data[f"step_completion_grounding_{step_count}"] = completion_gate
                    apply_step_verification(
                        plan_state, current_plan_step["step_id"], verification_payload, plan_transitions,
                    )
                    after_current_step = next(
                        step for step in plan_state["steps"] if step["step_id"] == current_plan_step["step_id"]
                    )
                    progress = assess_step_progress(before_current_step, after_current_step)
                    attempt = {
                        "attempt_index": len(prior_attempts) + 1,
                        "stable_step_id": active_step_id,
                        "tool_name": tool_name,
                        "sub_goal": sub_goal,
                        "context": context,
                        "command": command,
                        "executable_signature": executable_signature(tool_name, command, context, sub_goal),
                        "verified_evidence_before": progress["verified_evidence_before"],
                        "verified_evidence_after": progress["verified_evidence_after"],
                        "new_evidence_obtained": bool(progress["evidence_added"]),
                        "made_progress": progress["made_progress"],
                        "verifier_rationale": verification_payload.get("rationale", ""),
                    }
                    prior_attempts.append(attempt)
                    previous_verifier_assessment = verification_payload
                    json_data[f"step_verification_{step_count}"] = verification_payload
                    json_data[f"current_step_progress_{step_count}"] = {
                        "active_step_id": current_plan_step["step_id"],
                        "stable_step_id": active_step_id,
                        "attempt": attempt,
                        "progress": progress,
                    }
                    json_data[f"plan_after_step_{step_count}"] = plan_snapshot(plan_state)
                    json_data["plan_transitions"] = copy.deepcopy(plan_transitions)
                    if self.verbose:
                        print(f"\n==> 🤖 Step {step_count}: Plan-Step Verification\n")
                        print(
                            f"[Current Step]: {current_plan_step['step_id']} "
                            f"[Completed]: {step_verification.completed} "
                            f"[Missing]: {step_verification.missing_evidence}"
                        )
                        print(f"[Time]: {round(time.time() - local_start_time, 2)}s")
                    if all_steps_completed(plan_state):
                        termination_reason = "all_plan_steps_completed"
                        break
                else:
                    stop_verification = self.verifier.verificate_context(
                        question,
                        image_path,
                        query_analysis,
                        self.memory,
                        step_count,
                        json_data
                    )
                    context_verification, conclusion = self.verifier.extract_conclusion(stop_verification)
                    previous_verifier_assessment = stop_verification
                    if self.verbose:
                        conclusion_emoji = "✅" if conclusion == 'STOP' else "🛑"
                        print(f"\n==> 🤖 Step {step_count}: Context Verification\n")
                        print(f"[Analysis]: {context_verification}\n[Conclusion]: {conclusion} {conclusion_emoji}")
                        print(f"[Time]: {round(time.time() - local_start_time, 2)}s")
                    if conclusion == 'STOP':
                        termination_reason = "verifier_stop"
                        break

            if termination_reason is None:
                if step_count >= self.max_steps:
                    termination_reason = "max_steps_with_unresolved_plan" if hierarchical_planning else "max_steps"
                elif (time.time() - query_start_time) >= self.max_time:
                    termination_reason = "max_time"
                else:
                    termination_reason = "loop_exit_unknown"

            # Add memory and statistics to json_data
            json_data.update({
                "memory": memory_actions,
                "step_count": step_count,
                "execution_time": round(time.time() - query_start_time, 2),
                "termination_reason": termination_reason,
            })
            if hierarchical_planning:
                json_data["high_level_plan"] = plan_snapshot(plan_state)
                json_data["plan_transitions"] = copy.deepcopy(plan_transitions)

            plan_complete = not hierarchical_planning or all_steps_completed(plan_state)
            if hierarchical_planning and not plan_complete:
                unresolved = unresolved_steps(plan_state)
                unresolved_text = "; ".join(
                    f"{step['step_id']}: {', '.join(step.get('missing_evidence') or [step['success_criteria']])}"
                    for step in unresolved
                )
                grounded_failure = f"Insufficient verified evidence; unresolved plan steps: {unresolved_text}"
                if 'final' in self.output_types:
                    json_data["final_output"] = grounded_failure
                if 'direct' in self.output_types:
                    json_data["direct_output"] = grounded_failure
                if self.verbose:
                    print(f"\n==> ⚠️ Final generation withheld: {grounded_failure}")

            # Generate final output only after all required plan steps complete.
            if plan_complete and 'final' in self.output_types:
                final_output = self.planner.generate_final_output(question, image_path, self.memory)
                json_data["final_output"] = final_output
                print(f"\n==> 🐙 Detailed Solution:\n\n{final_output}")

            if plan_complete and 'direct' in self.output_types:
                direct_output = self.planner.generate_direct_output(question, image_path, self.memory)
                json_data["direct_output"] = direct_output
                print(f"\n==> 🐙 Final Answer:\n\n{direct_output}")

            print(f"\n[Total Time]: {round(time.time() - query_start_time, 2)}s")
            print(f"\n==> ✅ Query Solved!")

        return json_data

def construct_solver(llm_engine_name : str = "gpt-4o",
                     enabled_tools : list[str] = ["all"],
                     tool_engine: list[str] = ["Default"],
                     model_engine: list[str] = ["trainable", "gpt-4o", "gpt-4o", "gpt-4o"],  # [planner_main, planner_fixed, verifier, executor]
                     output_types : str = "final,direct",
                     max_steps : int = 10,
                     max_time : int = 300,
                     max_tokens : int = 4000,
                     root_cache_dir : str = "solver_cache",
                     verbose : bool = True,
                     vllm_config_path : str = None,
                     base_url : str = None,
                     temperature: float = 0.0
                     ):

    # Parse model_engine configuration
    # Format: [planner_main, planner_fixed, verifier, executor]
    # Both markers refer to the same local base endpoint.  The distinction is
    # intentional: only planner_main is allowed to use the actor LoRA;
    # frozen roles are base-only.  The async OpenAI-compatible vLLM route in
    # the pinned VERL/vLLM stack is given stable request-level aliases below;
    # the frozen path is represented explicitly and never creates another model.
    unified_local_roles = os.getenv("AGENTFLOW_UNIFIED_LOCAL_ROLES", "0").lower() in {
        "1", "true", "yes", "on"
    }
    unified_fixed_role_engine = os.getenv("AGENTFLOW_UNIFIED_FIXED_ROLE_ENGINE", "").strip()
    if unified_local_roles:
        # These are logical OpenAI-compatible model ids.  PatchedvLLMServer
        # maps qwen-base to no adapter and qwen-actor to the latest synced
        # TensorLoRARequest.  They must remain distinct at request time.
        planner_main_engine = "vllm-qwen-actor"
        planner_fixed_engine = unified_fixed_role_engine or "vllm-qwen-base"
        verifier_engine = unified_fixed_role_engine or "vllm-qwen-base"
        executor_engine = unified_fixed_role_engine or "vllm-qwen-base"
    else:
        def resolve_role_engine(spec: str) -> str:
            return llm_engine_name if spec in {"trainable", "frozen"} else spec

        planner_main_engine = resolve_role_engine(model_engine[0])
        planner_fixed_engine = resolve_role_engine(model_engine[1])
        verifier_engine = resolve_role_engine(model_engine[2])
        executor_engine = resolve_role_engine(model_engine[3])

    if unified_local_roles:
        if model_engine != ["trainable", "frozen", "frozen", "frozen"]:
            raise ValueError(
                "Unified local role mode requires MODEL_ENGINE=['trainable','frozen','frozen','frozen']"
            )
        if any(engine not in {"self", "frozen"} for engine in tool_engine):
            raise ValueError("Unified local role mode requires every TOOL_ENGINE entry to be local frozen base")
        if not base_url or not llm_engine_name.startswith("vllm-"):
            raise ValueError("Unified local role mode requires a local vLLM model and base_url")
        if unified_fixed_role_engine and not unified_fixed_role_engine.startswith("doubao-"):
            raise ValueError("AGENTFLOW_UNIFIED_FIXED_ROLE_ENGINE must be a doubao-* model")
        print(
            "UNIFIED_LOCAL_ROLES enabled "
            f"model={llm_engine_name.removeprefix('vllm-')} "
            "planner_main=trainable_actor_lora "
            f"planner_fixed={planner_fixed_engine} verifier={verifier_engine} "
            f"executor={executor_engine} fixed_roles_frozen=1"
        )

    fixed_role_external = planner_fixed_engine.startswith("doubao-")
    fixed_role_temperature = float(
        os.getenv("AGENTFLOW_UNIFIED_FIXED_ROLE_TEMPERATURE", "0.0" if fixed_role_external else str(temperature))
    )
    fixed_base_url = None if fixed_role_external else base_url
    initializer_model = planner_fixed_engine if fixed_role_external else (
        "vllm-qwen-base" if unified_local_roles else llm_engine_name
    )
    initializer_base_url = None if fixed_role_external else base_url

    # Instantiate Initializer
    initializer = Initializer(
        enabled_tools=enabled_tools,
        tool_engine=tool_engine,
        model_string=initializer_model,
        verbose=verbose,
        vllm_config_path=vllm_config_path,
        base_url=initializer_base_url,
        max_tokens=max_tokens,
    )

    # Instantiate Planner
    planner = Planner(
        llm_engine_name=planner_main_engine,
        llm_engine_fixed_name=planner_fixed_engine,
        toolbox_metadata=initializer.toolbox_metadata,
        available_tools=initializer.available_tools,
        verbose=verbose,
        base_url=base_url,
        fixed_base_url=fixed_base_url,
        fixed_temperature=fixed_role_temperature,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Instantiate Verifier
    verifier = Verifier(
        llm_engine_name=verifier_engine,
        llm_engine_fixed_name=planner_fixed_engine,
        toolbox_metadata=initializer.toolbox_metadata,
        available_tools=initializer.available_tools,
        verbose=verbose,
        base_url=None if fixed_role_external else (base_url if (unified_local_roles or verifier_engine == llm_engine_name) else None),
        fixed_base_url=fixed_base_url if (unified_local_roles or planner_fixed_engine == llm_engine_name) else None,
        max_tokens=max_tokens,
        temperature=fixed_role_temperature if fixed_role_external else temperature
    )

    # Instantiate Memory
    memory = Memory()

    # Instantiate Executor with tool instances cache
    executor = Executor(
        llm_engine_name=executor_engine,
        root_cache_dir=root_cache_dir,
        verbose=verbose,
        base_url=None if fixed_role_external else (base_url if (unified_local_roles or executor_engine == llm_engine_name) else None),
        temperature=fixed_role_temperature if fixed_role_external else temperature,
        tool_instances_cache=initializer.tool_instances_cache,  # Pass the cached tool instances
        max_tokens=max_tokens,
    )

    # Instantiate Solver
    solver = Solver(
        planner=planner,
        verifier=verifier,
        memory=memory,
        executor=executor,
        output_types=output_types,
        max_steps=max_steps,
        max_time=max_time,
        max_tokens=max_tokens,
        root_cache_dir=root_cache_dir,
        verbose=verbose,
        temperature=temperature
    )
    return solver

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the agentflow demo with specified parameters.")
    parser.add_argument("--llm_engine_name", default="gpt-4o", help="LLM engine name.")
    parser.add_argument(
        "--output_types",
        default="base,final,direct",
        help="Comma-separated list of required outputs (base,final,direct)"
    )
    parser.add_argument("--enabled_tools", default="Base_Generator_Tool", help="List of enabled tools.")
    parser.add_argument("--root_cache_dir", default="solver_cache", help="Path to solver cache directory.")
    parser.add_argument("--max_tokens", type=int, default=4000, help="Maximum tokens for LLM generation.")
    parser.add_argument("--max_steps", type=int, default=10, help="Maximum number of steps to execute.")
    parser.add_argument("--max_time", type=int, default=300, help="Maximum time allowed in seconds.")
    parser.add_argument("--verbose", type=bool, default=True, help="Enable verbose output.")
    return parser.parse_args()
    
def main(args):
    tool_engine=["gpt-4o-mini","gpt-4o-mini","Default","Default"]
    solver = construct_solver(
        llm_engine_name=args.llm_engine_name,
        enabled_tools=["Base_Generator_Tool","Python_Coder_Tool","Google_Search_Tool","Wikipedia_Search_Tool"],
        tool_engine=tool_engine,
        output_types=args.output_types,
        max_steps=args.max_steps,
        max_time=args.max_time,
        max_tokens=args.max_tokens,
        # base_url="http://localhost:8080/v1",
        verbose=args.verbose,
        temperature=0.7
    )

    # Solve the task or problem
    solver.solve("What is the capital of France?")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
