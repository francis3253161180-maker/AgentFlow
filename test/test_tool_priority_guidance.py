import unittest
from unittest.mock import patch


class _FakeEngine:
    def __init__(self, model_string, **kwargs):
        self.model_string = model_string
        self.kwargs = kwargs


class ToolPriorityGuidanceTest(unittest.TestCase):
    def test_guidance_is_generic_and_names_specialist_responsibilities(self):
        from agentflow.models.executor import EXECUTOR_ROLE_BOUNDARY
        from agentflow.models.planner import (
            FINAL_SYNTHESIS_BOUNDARY,
            QUERY_ANALYSIS_BOUNDARY,
            STAGNATION_GUARD,
            TOOL_SELECTION_GUIDANCE,
        )
        from agentflow.models.verifier import VERIFIER_ROLE_BOUNDARY

        self.assertIn("Wikipedia_RAG_Search_Tool", TOOL_SELECTION_GUIDANCE)
        self.assertIn("Web_RAG_Search_Tool", TOOL_SELECTION_GUIDANCE)
        self.assertIn("already known", TOOL_SELECTION_GUIDANCE)
        self.assertIn("Ground_Google_Search_Tool", TOOL_SELECTION_GUIDANCE)
        self.assertIn("Python_Code_Generator_Tool", TOOL_SELECTION_GUIDANCE)
        self.assertIn("Pubmed_Search_Tool", TOOL_SELECTION_GUIDANCE)
        self.assertIn("not a fixed tool order", TOOL_SELECTION_GUIDANCE)
        self.assertIn("same or near-identical query", STAGNATION_GUARD)
        self.assertIn("merely to\ncreate diversity", STAGNATION_GUARD)
        self.assertIn("must not answer factual/entity questions", QUERY_ANALYSIS_BOUNDARY)
        self.assertIn("synthesize only from the accumulated actions/results", FINAL_SYNTHESIS_BOUNDARY)
        self.assertIn("translate only the planner-selected tool", EXECUTOR_ROLE_BOUNDARY)
        self.assertIn("judge only the evidence already present in memory", VERIFIER_ROLE_BOUNDARY)

    def test_generalist_metadata_marks_it_as_fallback_not_lookup_or_calculator(self):
        from agentflow.tools.base_generator.tool import Base_Generator_Tool

        with patch("agentflow.tools.base_generator.tool.create_llm_engine", side_effect=_FakeEngine):
            metadata = Base_Generator_Tool("vllm-qwen-base", base_url="http://127.0.0.1:1/v1").get_metadata()

        description = metadata["tool_description"].lower()
        practices = metadata["user_metadata"]["best_practice"].lower()
        self.assertIn("fallback", description)
        self.assertIn("not a factual search", description)
        self.assertIn("do not use it as a calculator", practices)

    def test_python_tool_accepts_unified_local_endpoint(self):
        from agentflow.tools.python_coder.tool import Python_Coder_Tool

        with patch("agentflow.tools.python_coder.tool.create_llm_engine", side_effect=_FakeEngine) as factory:
            Python_Coder_Tool(
                "vllm-qwen-base",
                base_url="http://127.0.0.1:1/v1",
                max_tokens=321,
            )

        self.assertEqual(factory.call_args.kwargs["base_url"], "http://127.0.0.1:1/v1")
        self.assertEqual(factory.call_args.kwargs["max_tokens"], 321)

    def test_wikipedia_retrieval_is_raw_and_does_not_require_openai_or_an_llm(self):
        from agentflow.tools.wikipedia_search.tool import Wikipedia_Search_Tool

        class _Response:
            def __init__(self, payload):
                self.payload = payload
                self.status_code = 200
                self.headers = {}

            def raise_for_status(self):
                return None

            def json(self):
                return self.payload

        tool = Wikipedia_Search_Tool("vllm-qwen-base", base_url="http://127.0.0.1:1/v1")
        self.assertFalse(tool.require_llm_engine)
        self.assertEqual(tool.model_string, "raw-wikipedia")
        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "agentflow.tools.wikipedia_search.tool.requests.get",
                side_effect=[
                    _Response({"query": {"search": [{"title": "Example"}]}}),
                    _Response({"query": {"pages": {"1": {
                        "title": "Example",
                        "fullurl": "https://en.wikipedia.org/wiki/Example",
                        "extract": "Evidence text from the public encyclopedia.",
                    }}}}),
                ],
            ):
                    result = tool.execute("Ernst Mach")

        pages = result["relevant_pages (public search order; raw evidence only)"]
        self.assertEqual(pages[0]["title"], "Example")
        self.assertEqual(pages[0]["url"], "https://en.wikipedia.org/wiki/Example")
        self.assertEqual(result["search_telemetry"]["search_internal_llm_calls"], 0)
        self.assertEqual(result["search_telemetry"]["openai_calls"], 0)
        self.assertEqual(result["search_telemetry"]["doubao_calls"], 0)

    def test_wikipedia_caches_network_response_and_retries_429(self):
        from agentflow.tools.wikipedia_search.tool import Wikipedia_Search_Tool

        class _Response:
            def __init__(self, payload, status_code=200, retry_after=None):
                self.payload = payload
                self.status_code = status_code
                self.headers = {"Retry-After": retry_after} if retry_after is not None else {}

            def raise_for_status(self):
                if self.status_code == 429:
                    import requests
                    raise requests.HTTPError("429")

            def json(self):
                return self.payload

        tool = Wikipedia_Search_Tool()
        with patch("agentflow.tools.wikipedia_search.tool.time.sleep") as sleep, patch(
            "agentflow.tools.wikipedia_search.tool.requests.get",
            side_effect=[
                _Response({}, status_code=429, retry_after="0"),
                _Response({"query": {"search": [{"title": "Example"}]}}),
                _Response({"query": {"pages": {"1": {
                    "title": "Example", "fullurl": "https://en.wikipedia.org/wiki/Example",
                    "extract": "Evidence text.",
                }}}}),
            ],
        ) as get:
            first = tool.execute("Example")
            second = tool.execute("Example")

        self.assertEqual(first["search_telemetry"]["http_429"], 1)
        self.assertEqual(first["search_telemetry"]["retries"], 1)
        self.assertEqual(second["search_telemetry"]["cache_hits"], 2)
        self.assertEqual(get.call_count, 3)
        sleep.assert_called_once_with(0.0)

    def test_web_rag_is_known_url_raw_evidence_without_openai(self):
        from agentflow.tools.web_search.tool import Web_Search_Tool

        class _Response:
            status_code = 200
            headers = {}
            content = b"<html><body>Barcelona won the Spanish league title. A season has 38 games.</body></html>"

            def raise_for_status(self):
                return None

        with patch.dict("os.environ", {}, clear=True), patch(
            "agentflow.tools.web_search.tool.requests.get", return_value=_Response()
        ) as get:
            tool = Web_Search_Tool("vllm-qwen-base", base_url="http://127.0.0.1:1/v1")
            first = tool.execute("league games", "https://example.org/facts")
            second = tool.execute("league games", "https://example.org/facts")

        self.assertFalse(tool.require_llm_engine)
        self.assertEqual(tool.model_string, "raw-web-lexical")
        self.assertTrue(first["evidence_chunks"])
        self.assertEqual(first["web_search_telemetry"]["openai_calls"], 0)
        self.assertEqual(first["web_search_telemetry"]["doubao_calls"], 0)
        self.assertEqual(second["web_search_telemetry"]["cache_hits"], 1)
        self.assertEqual(get.call_count, 1)


if __name__ == "__main__":
    unittest.main()
