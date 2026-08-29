import unittest
from tempfile import TemporaryDirectory
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

    def test_routing_state_snapshot_exposes_only_generic_prior_state(self):
        from agentflow.models.memory import Memory
        from agentflow.models.planner import routing_state_snapshot

        memory = Memory()
        memory.add_action(
            1,
            "Wikipedia_RAG_Search_Tool",
            "Find a league format",
            "execution = tool.execute(...) ",
            {"url": "https://example.org/league"},
        )
        snapshot = routing_state_snapshot(
            memory,
            {"analysis": "The exact format is still missing.", "stop_signal": False},
        )

        self.assertIn("Previous verifier assessment", snapshot)
        self.assertIn("Wikipedia_RAG_Search_Tool", snapshot)
        self.assertIn("https://example.org/league", snapshot)
        self.assertIn("not a forced tool order", snapshot)
        self.assertIn("genuinely new entity or sub-goal", snapshot)

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
        tool.min_request_interval_seconds = 0.0
        self.assertFalse(tool.require_llm_engine)
        self.assertEqual(tool.model_string, "raw-wikipedia")
        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "agentflow.tools.wikipedia_search.tool.requests.get",
                side_effect=[
                    _Response({"query": {"search": [{"title": "Example", "snippet": "<span>Matched</span> evidence"}]}}),
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
        self.assertEqual(pages[0]["search_snippet"], "Matched evidence")
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
        tool.min_request_interval_seconds = 0.0
        with TemporaryDirectory() as tmpdir:
            tool.throttle_lock_path = f"{tmpdir}/mediawiki.lock"
            tool.shared_cache_dir = f"{tmpdir}/raw-cache"
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
        sleep.assert_not_called()

    def test_wikipedia_shared_raw_cache_uses_full_request_semantics(self):
        from agentflow.tools.wikipedia_search.tool import Wikipedia_Search_Tool

        class _Response:
            status_code = 200
            headers = {}

            def __init__(self, payload):
                self.payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self.payload

        with TemporaryDirectory() as tmpdir:
            first = Wikipedia_Search_Tool()
            second = Wikipedia_Search_Tool()
            for tool in (first, second):
                tool.min_request_interval_seconds = 0.0
                tool.throttle_lock_path = f"{tmpdir}/throttle.lock"
                tool.shared_cache_dir = f"{tmpdir}/raw-cache"
            with patch("agentflow.tools.wikipedia_search.tool.requests.get", side_effect=[
                _Response({"query": {"search": [{"title": "Example", "snippet": "<b>Matched</b>"}]}}),
                _Response({"query": {"pages": {"1": {
                    "title": "Example", "fullurl": "https://en.wikipedia.org/wiki/Example",
                    "extract": "Evidence text.",
                }}}}),
            ]) as get:
                first_result = first.execute("same semantic request")
                second_result = second.execute("same semantic request")

        self.assertEqual(get.call_count, 2)
        self.assertEqual(first_result["search_telemetry"]["http_requests"], 2)
        self.assertEqual(second_result["search_telemetry"]["http_requests"], 0)
        self.assertEqual(second_result["search_telemetry"]["shared_cache_hits"], 2)
        self.assertEqual(first_result["relevant_pages (public search order; raw evidence only)"][0]["search_snippet"], "Matched")

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

    def test_web_rag_bm25_prefers_rare_relation_and_exact_numeric_tokens(self):
        """Ranking must not let a long generic chunk drown a rare dated fact."""
        from agentflow.tools.web_search.tool import Web_Search_Tool

        tool = Web_Search_Tool()
        generic = " ".join(["league Barcelona title season"] * 30)
        dated_relation = (
            "A historical record confirms Barcelona won back to back league titles "
            "in 1948 and 1949."
        )
        ranked = tool._rank(
            "Barcelona league titles 1948 1949",
            [generic, dated_relation],
        )
        self.assertEqual(ranked[0]["chunk_index"], 1)
        self.assertEqual(ranked[0]["matched_numeric_tokens"], ["1948", "1949"])
        self.assertGreater(ranked[0]["query_term_coverage"], ranked[1]["query_term_coverage"])
        self.assertIn("bm25_score", ranked[0])


if __name__ == "__main__":
    unittest.main()
