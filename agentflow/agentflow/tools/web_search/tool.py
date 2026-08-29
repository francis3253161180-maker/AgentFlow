"""Deterministic evidence extraction from one already known public URL."""

from __future__ import annotations

import re
import time
from typing import Any
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from agentflow.tools.base import BaseTool


TOOL_NAME = "Web_RAG_Search_Tool"
WEB_HEADERS = {
    "User-Agent": "AgentFlowResearchSmoke/1.0 (deterministic evidence extraction)",
    "Accept": "text/html,application/xhtml+xml",
}

LIMITATION = f"""
{TOOL_NAME} accepts one already known http(s) URL and extracts bounded raw text.
It is not a discovery/search engine, cannot access authenticated or JavaScript-only
content reliably, and uses deterministic lexical ranking rather than an LLM.
"""

BEST_PRACTICE = f"""
For optimal results with {TOOL_NAME}:
1. Use only a URL returned by a discovery tool or otherwise present in memory.
2. Use it when that URL's short discovery excerpt is insufficient for a specific
   relation/detail; do not use it for open-ended discovery.
3. Treat the returned chunks as evidence, not an answer to the original task.
4. Use a calculation tool only after retrieved evidence supplies the operands.
"""


class Web_Search_Tool(BaseTool):
    """Raw URL deep-reader with no OpenAI, embeddings, or answer generation."""

    require_llm_engine = False

    def __init__(self, model_string=None, base_url=None, max_tokens=2048):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Deep-read/extract raw evidence from one already known URL. It is not "
                "an open-web search engine and does not answer the original task."
            ),
            tool_version="1.0.0",
            input_types={
                "query": "str - Specific detail to locate in the already known page.",
                "url": "str - An already known http(s) URL returned by discovery/evidence.",
            },
            output_type="dict - URL, deterministic lexical-ranked raw excerpts, and telemetry.",
            demo_commands=[{
                "command": 'execution = tool.execute(query="mass in kg", url="https://en.wikipedia.org/wiki/Moon")',
                "description": "Extract evidence from an already discovered Moon URL.",
            }],
            user_metadata={"limitation": LIMITATION, "best_practice": BEST_PRACTICE},
        )
        self.model_string = "raw-web-lexical"
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.chunk_size_words = 100
        self.chunk_overlap_words = 20
        self.top_k = 2
        self.max_http_retries = 2
        self.max_backoff_seconds = 2.0
        # Successful page text only; scoped to one tool instance and never
        # carries planner/verifier state, answers, or rewards across rollouts.
        self._content_cache: dict[str, str] = {}

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _request_content(self, url: str, telemetry: dict[str, Any]) -> str:
        cached = self._content_cache.get(url)
        if cached is not None:
            telemetry["cache_hits"] += 1
            return cached
        for attempt in range(self.max_http_retries + 1):
            response = requests.get(url, headers=WEB_HEADERS, timeout=15)
            if response.status_code == 429:
                telemetry["http_429"] += 1
                if attempt >= self.max_http_retries:
                    response.raise_for_status()
                retry_after = response.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after is not None else 0.25 * (2 ** attempt)
                except (TypeError, ValueError):
                    delay = 0.25 * (2 ** attempt)
                telemetry["retries"] += 1
                time.sleep(min(self.max_backoff_seconds, max(0.0, delay)))
                continue
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            content = soup.get_text(separator=" ", strip=True)[:200000]
            self._content_cache[url] = content
            return content
        raise RuntimeError("bounded URL retry loop exited unexpectedly")

    def _chunks(self, content: str) -> list[str]:
        words = content.split()
        if not words:
            return []
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + self.chunk_size_words, len(words))
            chunks.append(" ".join(words[start:end]))
            if end == len(words):
                break
            start = end - self.chunk_overlap_words
        return chunks

    def _rank(self, query: str, chunks: list[str]) -> list[tuple[int, int]]:
        query_terms = set(self._tokens(query))
        ranked = []
        for index, chunk in enumerate(chunks):
            terms = self._tokens(chunk)
            score = sum(terms.count(term) for term in query_terms)
            ranked.append((score, index))
        return sorted(ranked, key=lambda item: (-item[0], item[1]))

    def execute(self, query: str, url: str) -> dict[str, Any]:
        telemetry = {
            "provider": "public_url", "ranking": "deterministic_lexical",
            "cache_hits": 0, "retries": 0, "http_429": 0,
            "search_internal_llm_calls": 0, "openai_calls": 0, "doubao_calls": 0,
        }
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return {
                "query": query, "url": url,
                "error": "Web_RAG requires one valid already known http(s) URL.",
                "web_search_telemetry": telemetry,
            }
        try:
            content = self._request_content(url, telemetry)
            chunks = self._chunks(content)
            if not chunks:
                return {
                    "query": query, "url": url, "evidence_chunks": [],
                    "error": "No text could be extracted.",
                    "web_search_telemetry": telemetry,
                }
            ranked = self._rank(query, chunks)[: self.top_k]
            return {
                "query": query,
                "url": url,
                "evidence_chunks": [
                    {"chunk_index": index, "lexical_score": score, "excerpt": chunks[index]}
                    for score, index in ranked
                ],
                "web_search_telemetry": telemetry,
            }
        except requests.RequestException as exc:
            return {
                "query": query, "url": url, "error": f"Error fetching known URL: {exc}",
                "web_search_telemetry": telemetry,
            }
        except Exception as exc:
            return {
                "query": query, "url": url, "error": f"Error extracting known URL evidence: {exc}",
                "web_search_telemetry": telemetry,
            }
