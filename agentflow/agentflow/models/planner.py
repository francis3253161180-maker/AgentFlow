import json
import os
import re
from typing import Any, Dict, List, Tuple

from PIL import Image

from agentflow.engine.factory import create_llm_engine
from agentflow.models.formatters import HighLevelPlan, NextStep, PlanCoverage, QueryAnalysis
from agentflow.models.memory import Memory
from agentflow.models.current_step_state import build_current_step_contract
from agentflow.models.role_boundaries import audit_fixed_role_output, structurally_safe_supervisor_output
from agentflow.models.structured_outputs import (
    Game24Answer,
    extract_game24_numbers,
    game24_prompt,
    parse_game24_answer,
    parse_strict_json,
    select_valid_candidate,
)


# This is deliberately a capability policy, not a task or dataset policy.  It
# is shown wherever a planner is asked to reason about tools or select one.
TOOL_SELECTION_GUIDANCE = """
Capability boundaries (apply only when the named tool is available):
- Wikipedia_RAG_Search_Tool is for stable encyclopedic discovery: entities,
  relations, and historical/background facts.  It returns raw public
  Wikipedia/MediaWiki evidence and URLs; it does not answer the task.
- Web_RAG_Search_Tool is for deterministic deep reading of an already known
  URL returned by a discovery tool when its short excerpt lacks a needed
  relation or detail.  It is not an open-ended search engine.
- Ground_Google_Search_Tool is for current/open-web, official/non-Wikipedia
  discovery, or a clearly insufficient Wikipedia result, and only when it is
  configured.  Never assume it is available.
- Python_Code_Generator_Tool is for arithmetic/derivation only after the
  necessary numeric operands are present in memory/evidence; it must not
  invent factual inputs.
- Pubmed_Search_Tool is for biomedical literature/knowledge lookup.
- Generalist_Solution_Generator_Tool is reasoning/synthesis/fallback only
  over already available evidence when a specialist tool is inapplicable.  It
  must not perform factual retrieval, web search, or arithmetic.

Observation-based policy, not a fixed tool order: stable factual sub-goals
usually warrant Wikipedia discovery; use Web_RAG only to deepen a returned URL;
use configured Google only for an open-web/insufficient-Wikipedia case; and use
Python only after evidence supplies operands.  Select one available tool that
fits the current sub-goal and evidence state.
""".strip()

STAGNATION_GUARD = """
Stagnation guard: if the verifier says current evidence is insufficient, do
not repeat the same retrieval tool with the same or near-identical query when
memory has not gained relevant evidence.  Reformulate a genuinely narrower
sub-goal, deep-read a URL already returned by discovery, or switch source/tool
only when the evidence and sub-goal justify it.  Do not switch tools merely to
create diversity.
""".strip()


def routing_state_snapshot(memory: Memory, previous_verifier_assessment: Any = None) -> str:
    """Expose compact, evidence-derived routing state to the next planner call.

    This is deliberately advisory: it makes existing Memory URLs, the last
    action signature, and the verifier's prior assessment visible without
    prescribing a tool or forbidding legitimate repeated retrieval.
    """
    raw_actions = memory.get_actions()
    actions = list(raw_actions.values()) if isinstance(raw_actions, dict) else []
    last_action = actions[-1] if actions else {}
    last_tool = last_action.get("tool_name", "none") if isinstance(last_action, dict) else "none"
    last_subgoal = last_action.get("sub_goal", "none") if isinstance(last_action, dict) else "none"
    urls = sorted(set(re.findall(r"https?://[^\s'\"<>)}\]]+", str(raw_actions))))
    compact_verifier = str(previous_verifier_assessment) if previous_verifier_assessment is not None else "none (first step)"
    compact_verifier = compact_verifier[:700]
    urls_text = ", ".join(urls[:8]) if urls else "none"
    return (
        "Routing State Snapshot (advisory; not a forced tool order):\n"
        f"- Previous verifier assessment: {compact_verifier}\n"
        f"- Previous action signature: tool={last_tool}; sub_goal={str(last_subgoal)[:500]}\n"
        f"- Known URLs already present in Memory: {urls_text}\n"
        "Repeated use of the same tool remains valid for a genuinely new entity or sub-goal. "
        "If the verifier identifies the same unresolved evidence gap and Memory already contains "
        "a relevant URL, consider deep-reading that URL as an available candidate rather than "
        "repeating unchanged discovery."
    )

QUERY_ANALYSIS_BOUNDARY = """
Role boundary: this fixed query-analysis role may decompose the request and
identify needed tools, but must not answer factual/entity questions from
parametric knowledge, perform hidden retrieval, or perform arithmetic as a
substitute for an available specialist tool.
""".strip()

FINAL_SYNTHESIS_BOUNDARY = """
Role boundary: synthesize only from the accumulated actions/results in memory.
Do not perform fresh factual lookup, retrieve new facts, or do independent
nontrivial calculation. If memory lacks the needed evidence, state that
limitation rather than filling it in from model knowledge.
""".strip()


class Planner:
    def __init__(self, llm_engine_name: str, llm_engine_fixed_name: str = "gpt-4o",
                 toolbox_metadata: dict = None, available_tools: List = None,
                 verbose: bool = False, base_url: str = None, is_multimodal: bool = False,
                 check_model: bool = True, temperature : float = .0,
                 max_tokens: int = 2048, fixed_base_url: str = None,
                 fixed_temperature: float = None,
                 llm_engine_supervisor_name: str | None = None,
                 supervisor_base_url: str = None,
                 supervisor_temperature: float | None = None):
        self.llm_engine_name = llm_engine_name
        self.llm_engine_fixed_name = llm_engine_fixed_name
        self.is_multimodal = is_multimodal
        # self.llm_engine_mm = create_llm_engine(model_string=llm_engine_name, is_multimodal=False, base_url=base_url, temperature = temperature)
        self.llm_engine_fixed = create_llm_engine(
            model_string=llm_engine_fixed_name,
            is_multimodal=False,
            base_url=fixed_base_url,
            max_tokens=max_tokens,
            temperature=temperature if fixed_temperature is None else fixed_temperature,
        )
        self.llm_engine_supervisor_name = llm_engine_supervisor_name or llm_engine_fixed_name
        # Preserve the old single fixed-role engine by default.  A distinct
        # supervisor is opt-in so query analysis/final synthesis never move
        # providers merely because high-level evidence planning does.
        self.llm_engine_supervisor = self.llm_engine_fixed
        if self.llm_engine_supervisor_name != llm_engine_fixed_name:
            self.llm_engine_supervisor = create_llm_engine(
                model_string=self.llm_engine_supervisor_name,
                is_multimodal=False,
                base_url=supervisor_base_url,
                max_tokens=max_tokens,
                temperature=temperature if supervisor_temperature is None else supervisor_temperature,
            )
        self.supervisor_call_count = 0
        self.supervisor_boundary_audits: list[dict[str, Any]] = []
        self.llm_engine = create_llm_engine(
            model_string=llm_engine_name,
            is_multimodal=False,
            base_url=base_url,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self.toolbox_metadata = toolbox_metadata if toolbox_metadata is not None else {}
        self.available_tools = available_tools if available_tools is not None else []
        self.max_tokens = max_tokens

        self.verbose = verbose

    def _call_supervisor(self, prompt: str, response_format: Any) -> Any:
        self.supervisor_call_count = int(getattr(self, "supervisor_call_count", 0)) + 1
        engine = getattr(self, "llm_engine_supervisor", self.llm_engine_fixed)
        return engine(prompt, response_format=response_format, max_tokens=self.max_tokens)

    @staticmethod
    def _supervisor_audit(raw: Any, *, recorded_evidence: Any = None) -> dict[str, Any]:
        return audit_fixed_role_output(raw, recorded_evidence=recorded_evidence)

    def _structured_game24_output(self, question: str, memory: Memory) -> str:
        """Return one strictly validated Game24 JSON object, or a failure object.

        Existing marked candidates are checked locally first.  At most one
        constrained retry is sent to the local fixed Qwen role; free-form
        output is never promoted to an answer.
        """
        numbers = extract_game24_numbers(question)
        if numbers is None:
            raise ValueError("structured Game24 harness requires four input numbers")
        memory_text = str(memory.get_actions())
        candidate, local_result = select_valid_candidate(memory_text, numbers)
        if candidate is not None:
            answer = Game24Answer(expression=candidate)
            print("STRUCTURED_HARNESS route=deterministic status=validated", flush=True)
            return answer.model_dump_json() if hasattr(answer, "model_dump_json") else answer.json()

        prompt = game24_prompt(question, memory_text[-12000:])
        for attempt in range(2):
            raw = self.llm_engine_fixed([prompt], response_format=Game24Answer)
            answer, result = parse_game24_answer(raw, numbers)
            if answer is not None:
                print(
                    f"STRUCTURED_HARNESS route=guided_json status=validated attempt={attempt + 1}",
                    flush=True,
                )
                return answer.model_dump_json() if hasattr(answer, "model_dump_json") else answer.json()
            print(
                f"STRUCTURED_HARNESS schema_or_semantic_failure attempt={attempt + 1} reason={result.get('reason')}",
                flush=True,
            )
            if attempt == 0:
                prompt = game24_prompt(
                    question,
                    memory_text[-12000:],
                    feedback=(
                        f"reason={result.get('reason')}; expected_numbers={list(numbers)}; "
                        f"used_numbers={result.get('used_numbers', [])}"
                    ),
                )
        print(
            f"STRUCTURED_HARNESS status=failed reason={local_result.get('reason', 'no_valid_candidate')}",
            flush=True,
        )
        return '{"expression":""}'
    def get_image_info(self, image_path: str) -> Dict[str, Any]:
        image_info = {}
        if image_path and os.path.isfile(image_path):
            image_info["image_path"] = image_path
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                image_info.update({
                    "width": width,
                    "height": height
                })
            except Exception as e:
                print(f"Error processing image file: {str(e)}")
        return image_info

    def generate_base_response(self, question: str, image: str, max_tokens: int = 2048) -> str:
        image_info = self.get_image_info(image)

        input_data = [question]
        if image_info and "image_path" in image_info:
            try:
                with open(image_info["image_path"], 'rb') as file:
                    image_bytes = file.read()
                input_data.append(image_bytes)
            except Exception as e:
                print(f"Error reading image file: {str(e)}")


        print("Input data of `generate_base_response()`: ", input_data)
        self.base_response = self.llm_engine(input_data, max_tokens=max_tokens)
        # self.base_response = self.llm_engine_fixed(input_data, max_tokens=max_tokens)

        return self.base_response

    def analyze_query(self, question: str, image: str) -> str:
        image_info = self.get_image_info(image)

        if self.is_multimodal:
            query_prompt = f"""
Task: Analyze the given query with accompanying inputs and determine the skills and tools needed to address it effectively.

Available tools: {self.available_tools}

Metadata for the tools: {self.toolbox_metadata}

{TOOL_SELECTION_GUIDANCE}

{QUERY_ANALYSIS_BOUNDARY}

Image: {image_info}

Query: {question}

Instructions:
1. Carefully read and understand the query and any accompanying inputs.
2. Identify the main objectives or tasks within the query.
3. List the specific skills that would be necessary to address the query comprehensively.
4. Examine the available tools in the toolbox and determine which ones might relevant and useful for addressing the query. Make sure to consider the user metadata for each tool, including limitations and potential applications (if available).
5. Provide a brief explanation for each skill and tool you've identified, describing how it would contribute to answering the query.

Your response should include:
1. A concise summary of the query's main points and objectives, as well as content in any accompanying inputs.
2. A list of required skills, with a brief explanation for each.
3. A list of relevant tools from the toolbox, with a brief explanation of how each tool would be utilized and its potential limitations.
4. Any additional considerations that might be important for addressing the query effectively.

Please present your analysis in a clear, structured format.
                        """
        else: 
            query_prompt = f"""
Task: Analyze the given query to determine necessary skills and tools.

Inputs:
- Query: {question}
- Available tools: {self.available_tools}
- Metadata for tools: {self.toolbox_metadata}

{TOOL_SELECTION_GUIDANCE}

{QUERY_ANALYSIS_BOUNDARY}

Instructions:
1. Identify the main objectives in the query.
2. List the necessary skills and tools.
3. For each skill and tool, explain how it helps address the query.
4. Note any additional considerations.

Format your response with a summary of the query, lists of skills and tools with explanations, and a section for additional considerations.

Be biref and precise with insight. 
"""


        input_data = [query_prompt]
        if image_info:
            try:
                with open(image_info["image_path"], 'rb') as file:
                    image_bytes = file.read()
                input_data.append(image_bytes)
            except Exception as e:
                print(f"Error reading image file: {str(e)}")

        print("Input data of `analyze_query()`: ", input_data)

        # self.query_analysis = self.llm_engine_mm(input_data, response_format=QueryAnalysis)
        # self.query_analysis = self.llm_engine(input_data, response_format=QueryAnalysis)
        self.query_analysis = self.llm_engine_fixed(input_data, response_format=QueryAnalysis)

        return str(self.query_analysis).strip()

    def extract_context_subgoal_and_tool(self, response: Any) -> Tuple[str, str, str]:

        def normalize_tool_name(tool_name: str) -> str:
            """
            Normalizes a tool name robustly using regular expressions.
            It handles any combination of spaces and underscores as separators.
            """
            def to_canonical(name: str) -> str:
                # Split the name by any sequence of one or more spaces or underscores
                parts = re.split('[ _]+', name)
                # Join the parts with a single underscore and convert to lowercase
                return "_".join(part.lower() for part in parts)

            normalized_input = to_canonical(tool_name)
            
            for tool in self.available_tools:
                if to_canonical(tool) == normalized_input:
                    return tool
                    
            return f"No matched tool given: {tool_name}"

        try:
            if isinstance(response, str):
                # Attempt to parse the response as JSON
                try:
                    response_dict = json.loads(response)
                    response = NextStep(**response_dict)
                except Exception as e:
                    print(f"Failed to parse response as JSON: {str(e)}")
            if isinstance(response, NextStep):
                print("arielg 1")
                context = response.context.strip()
                sub_goal = response.sub_goal.strip()
                tool_name = response.tool_name.strip()
            else:
                print("arielg 2")
                text = response.replace("**", "")

                # Pattern to match the exact format
                pattern = r"Context:\s*(.*?)Sub-Goal:\s*(.*?)Tool Name:\s*(.*?)\s*(?:```)?\s*(?=\n\n|\Z)"

                # Find all matches
                matches = re.findall(pattern, text, re.DOTALL)

                # Return the last match (most recent/relevant)
                context, sub_goal, tool_name = matches[-1]
                context = context.strip()
                sub_goal = sub_goal.strip()
            tool_name = normalize_tool_name(tool_name)
        except Exception as e:
            print(f"Error extracting context, sub-goal, and tool name: {str(e)}")
            return None, None, None

        return context, sub_goal, tool_name

    def generate_next_step(self, question: str, image: str, query_analysis: str, memory: Memory, step_count: int, max_step_count: int, json_data: Any = None, previous_verifier_assessment: Any = None, hierarchical_plan: Any = None, current_step: Any = None, current_step_contract: Any = None, action_rejection: str | None = None, attempt_label: str = "initial") -> Any:
        def compact(value: Any, limit: int) -> str:
            text = str(value)
            if len(text) <= limit:
                return text
            head = max(1, limit * 2 // 3)
            return text[:head] + "\n...[context truncated]...\n" + text[-(limit - head):]

        # Keep the trainable local-model decision prompt within the small
        # context window used by the single-GPU smoke configuration.
        compact_question = compact(question, 1200)
        compact_analysis = compact(query_analysis, 1200)
        compact_tools = compact(self.available_tools, 500)
        compact_metadata = compact(self.toolbox_metadata, 1400)
        compact_memory = compact(memory.get_actions(), 600)
        compact_routing_state = routing_state_snapshot(memory, previous_verifier_assessment)
        hierarchical_context = ""
        if hierarchical_plan is not None and current_step is not None:
            contract = current_step_contract or build_current_step_contract(
                current_step, memory.get_actions(), [], previous_verifier_assessment,
            )
            hierarchical_context = f"""
Hierarchical Plan State:
- Full plan: {compact(hierarchical_plan, 1600)}
- Current-Step State Contract: {compact(json.dumps(contract, ensure_ascii=False), 2200)}
- The active plan step's `stable_step_id` is attached by the system; do not
  copy, invent, or select a target identifier. Formulate one atomic sub-goal
  for this one active goal only; do not redo a completed step or combine
  dependent goals into one action.
- Prior attempts record whether the verifier observed evidence progress. A
  repeat is valid only when it has a genuinely changed objective or target.
- `web_rag_deep_read_candidates` are URLs already obtained by discovery. Use
  Web_RAG only when one plausibly bears on an unresolved active requirement and
  its bounded discovery excerpt is insufficient; otherwise a materially new
  Wikipedia query remains valid. This is an evidence affordance, not a forced
  Wikipedia-to-Web sequence.
- An action executing successfully is not evidence that the step is complete;
  the verifier decides completion.
"""
            if action_rejection:
                hierarchical_context += f"""
Action revision required (this is the single allowed revision for this action):
- {action_rejection}
Use the same Current-Step State Contract and emit a changed, valid action.
"""

        if self.is_multimodal:
            prompt_generate_next_step = f"""
Task: Determine the optimal next step to address the given query based on the provided analysis, available tools, and previous steps taken.

Context:
Query: {compact_question}
Image: {image}
Query Analysis: {compact_analysis}

Available Tools:
{compact_tools}

Tool Metadata:
{compact_metadata}

Previous Steps and Their Results:
{compact_memory}

{compact_routing_state}

{hierarchical_context}

{TOOL_SELECTION_GUIDANCE}

{STAGNATION_GUARD}

Current Step: {step_count} in {max_step_count} steps
Remaining Steps: {max_step_count - step_count}

Instructions:
1. Analyze the context thoroughly, including the query, its analysis, any image, available tools and their metadata, and previous steps taken.

2. Determine the most appropriate next step by considering:
- Key objectives from the query analysis
- Capabilities of available tools
- Logical progression of problem-solving
- Outcomes from previous steps
- Current step count and remaining steps

3. Select ONE tool best suited for the next step, keeping in mind the limited number of remaining steps.

4. Formulate a specific, achievable sub-goal for the selected tool that maximizes progress towards answering the query.

Response Format (JSON object only; match the NextStep schema exactly):
{{"justification": "<why this one tool is appropriate>", "context": "<all information required by the tool>", "sub_goal": "<one achievable objective for the active step>", "tool_name": "<exact tool name>"}}

Your response MUST be this JSON object and nothing else. Do not use the legacy section format below.
Rules:
- Select only ONE tool.
- Include all relevant query, analysis, prior-step, file, variable, and tool metadata in the `context` field.
- Use the exact JSON field names above.
                        """
        else:
            prompt_generate_next_step = f"""
Task: Determine the optimal next step to address the query using available tools and previous steps.

Context:
- **Query:** {compact_question}
- **Query Analysis:** {compact_analysis}
- **Available Tools:** {compact_tools}
- **Toolbox Metadata:** {compact_metadata}
- **Previous Steps:** {compact_memory}
- **Routing State:** {compact_routing_state}
{hierarchical_context}

{TOOL_SELECTION_GUIDANCE}

{STAGNATION_GUARD}

Instructions:
1. Analyze the query, previous steps, and available tools.
2. Select the **single best tool** for the next step.
3. Formulate a specific, achievable **sub-goal** for that tool.
4. Provide all necessary **context** (data, file names, variables) for the tool to function.

Response Format (JSON object only; match the NextStep schema exactly):
{{"justification": "<why this one tool is appropriate>", "context": "<all information required by the tool>", "sub_goal": "<one achievable objective for the active step>", "tool_name": "<exact tool name>"}}

Use the exact JSON field names above. Do not emit markdown, section headings, or any text outside the JSON object.

Rules:
- Select only ONE tool.
- The sub-goal must be directly achievable by the selected tool.
- Put all required tool context in the `context` field.
                    """
            
        next_step = self.llm_engine(
            prompt_generate_next_step,
            response_format=NextStep,
            max_tokens=self.max_tokens,
        )
        if json_data is not None:
            prefix = f"action_predictor_{step_count}"
            if attempt_label != "initial":
                prefix += f"_{attempt_label}"
            json_data[f"{prefix}_routing_state"] = compact_routing_state
            json_data[f"{prefix}_prompt"] = prompt_generate_next_step
            json_data[f"{prefix}_response"] = str(next_step)
        return next_step

    def extract_target_gap(self, response: Any) -> str:
        """Read the structured current-step target without changing legacy extraction."""
        if isinstance(response, NextStep):
            return response.target_gap.strip()
        if isinstance(response, str):
            try:
                return NextStep(**json.loads(response)).target_gap.strip()
            except Exception:
                return ""
        if isinstance(response, dict):
            try:
                return NextStep(**response).target_gap.strip()
            except Exception:
                return ""
        return ""

    def generate_high_level_plan(
        self, question: str, image: str, query_analysis: str, max_step_count: int, json_data: Any = None,
    ) -> Any:
        """Ask the fixed planner role for small, dependency-aware evidence goals."""
        self.supervisor_boundary_audits = []
        prompt = f"""
Task: Create a concise high-level evidence plan for answering the query. This plan does not answer the query and does not select executable commands.

Query: {question}
Initial analysis: {query_analysis}
Available tools: {self.available_tools}

Create at most {max_step_count} ordered, atomic evidence goals. Each step must establish exactly one fact, relation, entity, or derivation input with a concrete success criterion. Split dependent facts into separate steps and use dependencies only when a later step needs a prior verified fact. Do not create composite steps, tool scripts, or a fixed tool sequence. Do not name tools, URLs, queries, commands, search strategies, factual answers, or calculations.

Response Format (JSON object only; match the HighLevelPlan schema exactly):
{{"steps":[{{"step_id":"step_1","objective":"<one atomic evidence goal>","success_criteria":"<what evidence proves it>","depends_on":[],"status":"pending","verified_evidence":[],"missing_evidence":[]}}]}}
"""
        # The OpenAI-compatible vLLM engine returns text even when guided_json
        # is requested. Parse that text before handing it to the state machine.
        raw_plan = self._call_supervisor(prompt, HighLevelPlan)
        plan_audit = self._supervisor_audit(raw_plan)
        self.supervisor_boundary_audits.append({"role": "high_level_evidence_planner", "audit": plan_audit})
        if not structurally_safe_supervisor_output(plan_audit):
            raise ValueError("supervisor high-level plan violated capability boundary")
        plan = parse_strict_json(raw_plan, HighLevelPlan)
        initial_plan = self._model_payload(plan)
        initial_coverage = self._audit_high_level_plan_coverage(question, initial_plan)
        final_plan = plan
        final_coverage = initial_coverage
        revised_plan_payload = None
        if not self._coverage_is_sufficient(initial_plan, initial_coverage):
            revision_prompt = f"""
Task: Revise an evidence plan. Do not answer the query and do not choose executable commands.

Query: {question}
Original plan: {json.dumps(initial_plan, ensure_ascii=False)}
Coverage audit: {json.dumps(self._model_payload(initial_coverage), ensure_ascii=False)}

Produce one corrected plan with at most {max_step_count} ordered atomic evidence steps.
Every independently necessary fact, relation, entity, or derivation input must
have its own step and concrete success criterion. Make dependencies explicit
when a later input needs an earlier verified input. Do not combine dependent
requirements inside a step. A final synthesis is not an evidence step. Do not name tools, URLs, queries, commands, search strategies, factual answers, or calculations.

Response Format (JSON object only; match the HighLevelPlan schema exactly):
{{"steps":[{{"step_id":"step_1","objective":"<one atomic evidence goal>","success_criteria":"<what evidence proves it>","depends_on":[],"status":"pending","verified_evidence":[],"missing_evidence":[]}}]}}
"""
            raw_revised_plan = self._call_supervisor(revision_prompt, HighLevelPlan)
            revised_plan_audit = self._supervisor_audit(raw_revised_plan)
            self.supervisor_boundary_audits.append({"role": "high_level_evidence_planner_revision", "audit": revised_plan_audit})
            if not structurally_safe_supervisor_output(revised_plan_audit):
                raise ValueError("supervisor revised plan violated capability boundary")
            final_plan = parse_strict_json(raw_revised_plan, HighLevelPlan)
            revised_plan_payload = self._model_payload(final_plan)
            final_coverage = self._audit_high_level_plan_coverage(question, revised_plan_payload)

        coverage_payload = self._model_payload(final_coverage)
        coverage_valid = self._coverage_is_sufficient(self._model_payload(final_plan), final_coverage)
        self.last_high_level_plan_coverage = {
            "valid": coverage_valid,
            "coverage": coverage_payload,
        }
        if json_data is not None:
            json_data["high_level_plan_prompt"] = prompt
            json_data["high_level_plan_response"] = str(raw_plan)
            json_data["high_level_plan_role_boundary_audit"] = plan_audit
            json_data["high_level_plan_original"] = initial_plan
            json_data["high_level_plan_coverage_initial"] = self._model_payload(initial_coverage)
            json_data["high_level_plan_revised"] = revised_plan_payload
            if revised_plan_payload is not None:
                json_data["high_level_plan_revision_role_boundary_audit"] = revised_plan_audit
            json_data["high_level_plan_coverage_final"] = coverage_payload
            json_data["high_level_plan_coverage_valid"] = coverage_valid
            json_data["high_level_plan_validated"] = self._model_payload(final_plan)
            json_data["supervisor_role_boundary_audits"] = list(self.supervisor_boundary_audits)
        return final_plan

    @staticmethod
    def _model_payload(value: Any) -> dict[str, Any]:
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if hasattr(value, "dict"):
            return value.dict()
        return value if isinstance(value, dict) else {}

    def _audit_high_level_plan_coverage(self, question: str, plan: dict[str, Any]) -> PlanCoverage:
        """Use the existing frozen planner role as a bounded plan-quality audit."""
        prompt = f"""
Task: Audit an evidence plan. Do not answer the query, retrieve facts, choose
tools, or infer missing factual content from model knowledge.

Query: {question}
Plan: {json.dumps(plan, ensure_ascii=False)}

Identify the independently necessary evidence requirements needed to answer
the query. Check whether every requirement is covered by an atomic plan step,
whether dependencies are explicit, and whether any step combines independent
requirements. Treat final answer synthesis as not being an evidence
requirement. Report only this plan audit. Do not name tools, URLs, queries, commands, search strategies, factual answers, or calculations.

Response Format (JSON object only; match the PlanCoverage schema exactly):
{{"sufficient":true,"independently_necessary_requirements":["<requirement>"],"requirement_coverage":[{{"requirement":"<same requirement>","covered_step_ids":["<one atomic step id>"]}}],"covered_step_ids":["<all covered step ids>"],"missing_requirements":[],"composite_step_ids":[],"rationale":"<brief evidence-plan rationale>"}}
"""
        raw_coverage = self._call_supervisor(prompt, PlanCoverage)
        coverage_audit = self._supervisor_audit(raw_coverage)
        self.supervisor_boundary_audits.append({"role": "coverage_auditor", "audit": coverage_audit})
        if not structurally_safe_supervisor_output(coverage_audit):
            return PlanCoverage(
                sufficient=False,
                missing_requirements=["coverage auditor violated capability boundary"],
                rationale="coverage_boundary_violation",
            )
        try:
            return parse_strict_json(raw_coverage, PlanCoverage)
        except ValueError as exc:
            # A malformed coverage audit is not permission to execute an
            # incomplete plan.  It triggers the one permitted plan revision
            # and, if still malformed, the solver's safe coverage gate.
            return PlanCoverage(
                sufficient=False,
                missing_requirements=["coverage audit could not be parsed"],
                rationale=f"coverage_parse_error: {exc}",
            )

    @staticmethod
    def _coverage_is_sufficient(plan: dict[str, Any], coverage: PlanCoverage) -> bool:
        def normalized(value: Any) -> str:
            return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(value).lower())).strip()

        step_ids = {str(step.get("step_id")) for step in plan.get("steps", []) if isinstance(step, dict)}
        requirements = {normalized(value) for value in coverage.independently_necessary_requirements if normalized(value)}
        coverage_map = {
            normalized(item.requirement): {str(step_id) for step_id in item.covered_step_ids}
            for item in coverage.requirement_coverage if normalized(item.requirement)
        }
        covered = {step_id for step_ids_for_requirement in coverage_map.values() for step_id in step_ids_for_requirement}
        composite = {str(step_id) for step_id in coverage.composite_step_ids}
        step_use_counts: dict[str, int] = {}
        for step_ids_for_requirement in coverage_map.values():
            for step_id in step_ids_for_requirement:
                step_use_counts[step_id] = step_use_counts.get(step_id, 0) + 1
        # A positive verdict must be internally consistent with the submitted
        # plan.  Each independent requirement needs a mapped atomic step; an
        # audit cannot silently map multiple independent requirements onto one
        # step while claiming that the plan is non-composite.
        return bool(
            coverage.sufficient
            and requirements
            and requirements == set(coverage_map)
            and all(coverage_map[requirement] for requirement in requirements)
            and covered
            and covered.issubset(step_ids)
            and set(coverage.covered_step_ids) == covered
            and all(count == 1 for count in step_use_counts.values())
            and not coverage.missing_requirements
            and not composite
        )


    def generate_final_output(self, question: str, image: str, memory: Memory) -> str:
        if os.getenv("AGENTFLOW_STRUCTURED_OUTPUT_HARNESS", "0").lower() in {"1", "true", "yes", "on"} and extract_game24_numbers(question):
            return self._structured_game24_output(question, memory)
        image_info = self.get_image_info(image)
        if self.is_multimodal:
            prompt_generate_final_output = f"""
Task: Generate the final output based on the query, image, and tools used in the process.

Context:
Query: {question}
Image: {image_info}
Actions Taken:
{memory.get_actions()}

Instructions:
1. Review the query, image, and all actions taken during the process.
2. Consider the results obtained from each tool execution.
3. Incorporate the relevant information from the memory to generate the step-by-step final output.
4. The final output should be consistent and coherent using the results from the tools.

{FINAL_SYNTHESIS_BOUNDARY}

Output Structure:
Your response should be well-organized and include the following sections:

1. Summary:
   - Provide a brief overview of the query and the main findings.

2. Detailed Analysis:
   - Break down the process of answering the query step-by-step.
   - For each step, mention the tool used, its purpose, and the key results obtained.
   - Explain how each step contributed to addressing the query.

3. Key Findings:
   - List the most important discoveries or insights gained from the analysis.
   - Highlight any unexpected or particularly interesting results.

4. Answer to the Query:
   - Directly address the original question with a clear and concise answer.
   - If the query has multiple parts, ensure each part is answered separately.

5. Additional Insights (if applicable):
   - Provide any relevant information or insights that go beyond the direct answer to the query.
   - Discuss any limitations or areas of uncertainty in the analysis.

6. Conclusion:
   - Summarize the main points and reinforce the answer to the query.
   - If appropriate, suggest potential next steps or areas for further investigation.
"""
        else:
                prompt_generate_final_output = f"""
Task: Generate the final output based on the query and the results from all tools used.

Context:
- **Query:** {question}
- **Actions Taken:** {memory.get_actions()}

Instructions:
1. Review the query and the results from all tool executions.
2. Incorporate the relevant information to create a coherent, step-by-step final output.

{FINAL_SYNTHESIS_BOUNDARY}
"""

        input_data = [prompt_generate_final_output]
        if image_info:
            try:
                with open(image_info["image_path"], 'rb') as file:
                    image_bytes = file.read()
                input_data.append(image_bytes)
            except Exception as e:
                print(f"Error reading image file: {str(e)}")

        # final_output = self.llm_engine_mm(input_data)
        # final_output = self.llm_engine(input_data)
        final_output = self.llm_engine_fixed(input_data)

        return final_output


    def generate_direct_output(self, question: str, image: str, memory: Memory) -> str:
        if os.getenv("AGENTFLOW_STRUCTURED_OUTPUT_HARNESS", "0").lower() in {"1", "true", "yes", "on"} and extract_game24_numbers(question):
            return self._structured_game24_output(question, memory)
        image_info = self.get_image_info(image)
        if self.is_multimodal:
            prompt_generate_final_output = f"""
Context:
Query: {question}
Image: {image_info}
Initial Analysis:
{self.query_analysis}
Actions Taken:
{memory.get_actions()}

Please generate the concise output based on the query, image information, initial analysis, and actions taken. Break down the process into clear, logical, and conherent steps. Conclude with a precise and direct answer to the query.

{FINAL_SYNTHESIS_BOUNDARY}

Answer:
"""
        else:
            prompt_generate_final_output = f"""
Task: Generate a concise final answer to the query based on all provided context.

Context:
- **Query:** {question}
- **Initial Analysis:** {self.query_analysis}
- **Actions Taken:** {memory.get_actions()}

Instructions:
1. Review the query and the results from all actions.
2. Synthesize the key findings into a clear, step-by-step summary of the process.
3. Provide a direct, precise answer to the original query.

{FINAL_SYNTHESIS_BOUNDARY}

Output Structure:
1.  **Process Summary:** A clear, step-by-step breakdown of how the query was addressed, including the purpose and key results of each action.
2.  **Answer:** A direct and concise final answer to the query.
"""

        input_data = [prompt_generate_final_output]
        if image_info:
            try:
                with open(image_info["image_path"], 'rb') as file:
                    image_bytes = file.read()
                input_data.append(image_bytes)
            except Exception as e:
                print(f"Error reading image file: {str(e)}")

        # final_output = self.llm_engine(input_data)
        final_output = self.llm_engine_fixed(input_data)
        # final_output = self.llm_engine_mm(input_data)

        return final_output
