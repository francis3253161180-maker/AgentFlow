import requests
from urllib.parse import quote

from agentflow.tools.base import BaseTool

# Tool name mapping - this defines the external name for this tool
TOOL_NAME = "Wikipedia_RAG_Search_Tool"
MEDIAWIKI_API_URL = "https://en.wikipedia.org/w/api.php"
MEDIAWIKI_HEADERS = {
    "User-Agent": "AgentFlowResearchSmoke/1.0 (public evidence retrieval)"
}

LIMITATION = f"""
{TOOL_NAME} has the following limitations:
1. It is designed specifically for retrieving grounded information from Wikipedia pages only.
2. It returns a bounded, deterministic prefix of MediaWiki/Wikipedia search results;
   it does not claim semantic reranking.
3. The returned information accuracy depends on Wikipedia content quality and the
   public endpoint's availability.
"""

BEST_PRACTICE = f"""
For optimal results with {TOOL_NAME}:
1. Use specific, targeted queries rather than broad or ambiguous questions.
2. The tool preserves the public search-result order and returns raw retrieved
   excerpts; treat them as evidence rather than an answer generated from model knowledge.
3. If initial results are insufficient, issue a more specific follow-up query.
4. Use this tool as part of a multi-step research process rather than a single source of truth.
"""

class Wikipedia_Search_Tool(BaseTool):
    """Public Wikipedia retrieval with no LLM, embeddings, or private API key.

    The executor supplies a planner-selected search query.  This tool returns raw
    evidence only: it neither answers the benchmark question nor semantically
    reranks/summarizes pages.  Public ``wikipedia`` search order is the explicit,
    deterministic ranking policy.
    """

    require_llm_engine = False

    def __init__(self, model_string=None, base_url=None, max_tokens=2048):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description="A factual retrieval tool that searches public Wikipedia/MediaWiki and returns raw pages in public search order with titles, URLs, and excerpts. It does not generate an answer or semantic reranking.",
            tool_version="1.0.0",
            input_types={
                "query": "str - The search query for Wikipedia."
            },
            output_type="dict - A dictionary containing search results, all matching pages with their content, URLs, and metadata.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(query="What is the exact mass in kg of the moon")',
                    "description": "Search Wikipedia and get the information about the mass of the moon."
                },
                {
                    "command": 'execution = tool.execute(query="Funtion of human kidney")',
                    "description": "Search Wikipedia and get the information about the function of human kidney."
                },
                {
                    "command": 'execution = tool.execute(query="When was the first moon landing?")',
                    "description": "Search Wikipedia and get the information about the first moon landing."
                }
            ],
            user_metadata = {
                "limitation": LIMITATION,
                "best_practice": BEST_PRACTICE
            }
        )
        # Accept the standard tool constructor arguments so unified-local mode
        # can instantiate every enabled tool uniformly.  They are deliberately
        # unused: factual retrieval is raw public Wikipedia/MediaWiki evidence.
        self.model_string = "raw-wikipedia"
        self.base_url = base_url
        self.max_tokens = max_tokens

    def _get_wikipedia_url(self, query):
        """
        Get the Wikipedia URL for a given query.
        """
        return f"https://en.wikipedia.org/wiki/{quote(query.replace(' ', '_'))}"

    def search_wikipedia(self, query, max_length=600, max_pages=2):
        """
        Searches Wikipedia based on the given query and returns multiple pages with their text and URLs.

        Parameters:
            query (str): The search query for Wikipedia.

        Returns:
            tuple: (search_results, pages_data)
                - search_results: List of search result titles
                - pages_data: List of dictionaries containing page info (title, text, url, error)
        """
        try:
            search_response = requests.get(
                MEDIAWIKI_API_URL,
                params={
                    "action": "query", "list": "search", "srsearch": query,
                    "srlimit": max_pages, "format": "json",
                },
                headers=MEDIAWIKI_HEADERS,
                timeout=15,
            )
            search_response.raise_for_status()
            matches = search_response.json().get("query", {}).get("search", [])
            titles = [match.get("title") for match in matches if match.get("title")]
            if not titles:
                return [{"title": None, "url": None, "abstract": None, "error": f"No results found for query: {query}"}]

            pages_data = []
            for title in titles:
                try:
                    page_response = requests.get(
                        MEDIAWIKI_API_URL,
                        params={
                            "action": "query", "prop": "extracts|info", "inprop": "url",
                            "exintro": 1, "explaintext": 1, "exchars": max_length,
                            "titles": title, "format": "json",
                        },
                        headers=MEDIAWIKI_HEADERS,
                        timeout=15,
                    )
                    page_response.raise_for_status()
                    pages = page_response.json().get("query", {}).get("pages", {})
                    page = next(iter(pages.values()), {})
                    text = page.get("extract") or ""
                    pages_data.append({
                        "title": page.get("title", title),
                        "url": page.get("fullurl", self._get_wikipedia_url(title)),
                        "abstract": text,
                    })
                except requests.RequestException as exc:
                    pages_data.append({
                        "title": title,
                        "url": self._get_wikipedia_url(title),
                        "abstract": None,
                        "error": f"Error retrieving Wikipedia page: {exc}",
                    })
            return pages_data
        except (requests.RequestException, ValueError) as exc:
            return [{"title": None, "url": None, "abstract": None, "error": f"Error searching Wikipedia: {exc}"}]

    def execute(self, query):
        """
        Searches Wikipedia based on the provided query and returns all matching pages.

        Parameters:
            query (str): The search query for Wikipedia.

        Returns:
            dict: A dictionary containing the search results and all matching pages with their content.
        """
        search_results = self.search_wikipedia(query)
        return {
            "query": query,
            "relevant_pages (public search order; raw evidence only)": search_results,
            "search_telemetry": {
                "provider": "public_wikipedia",
                "ranking": "public_search_order",
                "search_internal_llm_calls": 0,
                "openai_calls": 0,
                "doubao_calls": 0,
            },
        }

    def get_metadata(self):
        """
        Returns the metadata for the Wikipedia_Search_Tool.

        Returns:
            dict: A dictionary containing the tool's metadata.
        """
        metadata = super().get_metadata()
        return metadata


if __name__ == "__main__":
    # Test command:
    """
    Run the following commands in the terminal to test the script:

    cd agentflow/tools/wikipedia_search
    python tool.py
    """

    # Example usage of the Wikipedia_Search_Tool
    tool = Wikipedia_Search_Tool()

    # Get tool metadata
    metadata = tool.get_metadata()
    # print(metadata)

    # Sample query for searching Wikipedia
    # query = "Python programming language"
    # query = "what is the main function of the human kidney"
    # query = "What is the mass of the moon"
    # query = "mass of the moon"
    # query = "mass of the moon in kg"
    # query = "What is the mass of the moon (in kg)?"
    # query = "What is the capital of France"
    # query = "Who is Yann LeCun"
    # query = "What is the exact mass in kg of the moon?"
    query = "When was the first moon landing?"

    import json

    # Execute the tool with the sample query
    try:
        # Test with default parameters (all pages)
        execution = tool.execute(query=query)
        print("Execution Result (all pages):")
        print(json.dumps(execution, indent=4))

        # Save the execution result to a JSON file
        os.makedirs("logs", exist_ok=True)
        with open(f"logs/{query}.json", "w") as f:
            json.dump(execution, f, indent=4)
        
    except ValueError as e:
        print(f"Execution failed: {e}")

    print("Done!")
