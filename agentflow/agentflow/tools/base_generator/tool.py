import os
from agentflow.tools.base import BaseTool
from agentflow.engine.factory import create_llm_engine

# Tool name mapping - this defines the external name for this tool
TOOL_NAME = "Generalist_Solution_Generator_Tool"

LIMITATION = f"""
The {TOOL_NAME} may provide hallucinated or incorrect responses.
"""

BEST_PRACTICE = f"""
For optimal results with the {TOOL_NAME}:
1. Use it only for reasoning, synthesis, open-ended transformation, or fallback
   when no available specialized tool fits the current sub-goal.
2. Do not use it as a factual/entity/relation search tool when an available
   Wikipedia, knowledge-search, or web-search tool is applicable.
3. Do not use it as a calculator when an available Python_Code_Generator_Tool
   can execute the calculation from supplied inputs.
4. Do not use it for biomedical literature lookup when an available
   Pubmed_Search_Tool is applicable.
5. Provide clear, specific context and verify important claims with an
   appropriate retrieval or calculation tool when one is available.
"""

class Base_Generator_Tool(BaseTool):
    require_llm_engine = True

    def __init__(self, model_string="gpt-4o-mini", base_url=None, max_tokens=2048):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Fallback reasoning and synthesis tool for open-ended tasks when no "
                "available specialized tool fits. It is not a factual search/retrieval "
                "tool or arithmetic calculator substitute."
            ),
            tool_version="1.0.0",
            input_types={
                "query": "str - The query that includes query from the user to guide the agent to generate response.",
                # "query": "str - The query that includes query from the user to guide the agent to generate response (Examples: 'Describe this image in detail').",
                # "image": "str - The path to the image file if applicable (default: None).",
            },
            output_type="str - The generated response to the original query",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(query="Summarize the following text in a few lines")',
                    "description": "Generate a short summary given the query from the user."
                },
                # {
                #     "command": 'execution = tool.execute(query="Explain the mood of this scene.", image="path/to/image1.png")',
                #     "description": "Generate a caption focusing on the mood using a specific query and image."
                # },
                # {
                    # "command": 'execution = tool.execute(query="Give your best coordinate estimate for the pacemaker in the image and return (x1, y1, x2, y2)", image="path/to/image2.png")',
                    # "description": "Generate bounding box coordinates given the image and query from the user. The format should be (x1, y1, x2, y2)."
                # },
                # {
                #     "command": 'execution = tool.execute(query="Is the number of tiny objects that are behind the small metal jet less than the number of tiny things left of the tiny sedan?", image="path/to/image2.png")',
                #     "description": "Answer a question step by step given the image."
                # }
            ],

            user_metadata = {
                "limitation": LIMITATION,
                "best_practice": BEST_PRACTICE
            }

        )
        self.model_string = model_string  
        self.base_url = base_url
        self.max_tokens = max_tokens
        print(f"Initializing Generalist Tool with model: {self.model_string}")
        # multimodal = True if image else False
        multimodal = False
        # llm_engine = create_llm_engine(model_string=self.model_string, is_multimodal=multimodal, base_url=self.base_url)
        
        # NOTE: deterministic mode
        self.llm_engine = create_llm_engine(
            model_string=self.model_string, 
            is_multimodal=multimodal, 
            base_url=self.base_url,
            max_tokens=self.max_tokens,
            temperature=0.0, 
            top_p=1.0, 
            frequency_penalty=0.0, 
            presence_penalty=0.0
            )


    def execute(self, query, image=None):
        
        try:
            input_data = [query]
            response = self.llm_engine(input_data[0])
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def get_metadata(self):
        metadata = super().get_metadata()
        return metadata

if __name__ == "__main__":
    # Test command:
    """
    Run the following commands in the terminal to test the script:
    
    cd agentflow/tools/base_generator
    python tool.py
    """

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Script directory: {script_dir}")

    # Example usage of the Generalist_Tool
    tool = Base_Generator_Tool()

    tool = Base_Generator_Tool(model_string="gpt-4o-mini") # NOTE: strong LLM for tool
    # tool = Base_Generator_Tool(model_string="gemini-1.5-flash") # NOTE: weak 8B model for tool
    # tool = Base_Generator_Tool(model_string="dashscope") # NOTE: weak Qwen2.5-7B model for tool


    # Get tool metadata
    metadata = tool.get_metadata()
    print(metadata)

    query = "What is the capital of France?"

    # Execute the tool with default query
    try:
        execution = tool.execute(query=query)

        print("Generated Response:")
        print(execution)
    except Exception as e: 
        print(f"Execution failed: {e}")

    print("Done!")
