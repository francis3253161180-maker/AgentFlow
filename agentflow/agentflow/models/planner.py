import json
import os
import re
from typing import Any, Dict, List, Tuple

from PIL import Image

from agentflow.engine.factory import create_llm_engine
from agentflow.models.formatters import NextStep, QueryAnalysis
from agentflow.models.memory import Memory
from agentflow.models.structured_outputs import (
    Game24Answer,
    extract_game24_numbers,
    game24_prompt,
    parse_game24_answer,
    select_valid_candidate,
)


# This is deliberately a capability policy, not a task or dataset policy.  It
# is shown wherever a planner is asked to reason about tools or select one.
TOOL_SELECTION_GUIDANCE = """
Tool responsibility priority (apply only when the named tool is available):
- For factual, entity, or relation lookup, prefer a Wikipedia/knowledge-search
  or web-search/retrieval tool over a language-model answer generator.
- For arithmetic or calculation with supplied inputs, prefer
  Python_Code_Generator_Tool.
- For biomedical literature/knowledge lookup, prefer Pubmed_Search_Tool.
- Generalist_Solution_Generator_Tool is for reasoning, synthesis, or fallback
  only when no available specialized tool fits.  Do not use it as a factual
  search/retrieval or calculator substitute when an applicable specialized
  tool is available.
This is a relevance priority, not a fixed tool order: select one tool that is
both available and appropriate to the current sub-goal.
""".strip()

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
                 fixed_temperature: float = None):
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

    def generate_next_step(self, question: str, image: str, query_analysis: str, memory: Memory, step_count: int, max_step_count: int, json_data: Any = None) -> Any:
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

{TOOL_SELECTION_GUIDANCE}

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
{{"justification": "<why this one tool is appropriate>", "context": "<all information required by the tool>", "sub_goal": "<one achievable objective>", "tool_name": "<exact tool name>"}}

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

{TOOL_SELECTION_GUIDANCE}

Instructions:
1. Analyze the query, previous steps, and available tools.
2. Select the **single best tool** for the next step.
3. Formulate a specific, achievable **sub-goal** for that tool.
4. Provide all necessary **context** (data, file names, variables) for the tool to function.

Response Format (JSON object only; match the NextStep schema exactly):
{{"justification": "<why this one tool is appropriate>", "context": "<all information required by the tool>", "sub_goal": "<one achievable objective>", "tool_name": "<exact tool name>"}}

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
            json_data[f"action_predictor_{step_count}_prompt"] = prompt_generate_next_step
            json_data[f"action_predictor_{step_count}_response"] = str(next_step)
        return next_step


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
