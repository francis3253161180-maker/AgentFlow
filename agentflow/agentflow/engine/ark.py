"""Minimal OpenAI-compatible Ark/Doubao engine for frozen AgentFlow roles.

The provider is selected explicitly with a ``doubao-*`` model string.  The
API key and endpoint are resolved only from the process environment; neither
is persisted or printed.  AgentFlow's existing prompt and Pydantic validation
remain authoritative for structured responses.
"""

import hashlib
import os
from typing import List, Union

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError("Install the openai package to use the Ark/Doubao engine.") from exc

import platformdirs
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .base import CachedEngine, EngineLM


class ChatArk(EngineLM, CachedEngine):
    """OpenAI-compatible client for the explicitly configured Doubao model."""

    DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
    DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
    DEFAULT_MODEL = "doubao-seed-2-0-lite-260428"

    def __init__(
        self,
        model_string: str | None = None,
        use_cache: bool = False,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        is_multimodal: bool = False,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        **kwargs: object,
    ):
        self.model_string = model_string or os.getenv("ARK_MODEL") or self.DEFAULT_MODEL
        self.api_model = self.model_string
        self.use_cache = use_cache
        self.system_prompt = system_prompt
        self.is_multimodal = is_multimodal
        self.is_chat_model = True
        self.is_reasoning_model = False
        self.temperature = float(kwargs.get("temperature", 0.0))
        self.top_p = float(kwargs.get("top_p", 1.0))
        self.max_tokens = int(kwargs.get("max_tokens", 2048))

        if self.use_cache:
            root = platformdirs.user_cache_dir("agentflow")
            cache_path = os.path.join(root, f"cache_ark_{self.api_model}.db")
            super().__init__(cache_path=cache_path)

        resolved_api_key = api_key or os.getenv("ARK_API_KEY")
        if not resolved_api_key:
            raise ValueError("Please set ARK_API_KEY before selecting a Doubao engine.")
        resolved_base_url = base_url or os.getenv("ARK_BASE_URL") or self.DEFAULT_BASE_URL
        client_kwargs = {"api_key": resolved_api_key, "base_url": resolved_base_url}
        if timeout is not None:
            client_kwargs["timeout"] = timeout
        self.client = OpenAI(**client_kwargs)

    @staticmethod
    def _cache_key(system_prompt: str, prompt: str) -> str:
        # Cache keys must not expose question/answer text or credentials.
        return hashlib.sha256((system_prompt + "\n" + prompt).encode("utf-8")).hexdigest()

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
    def generate(self, content: Union[str, List[Union[str, bytes]]], system_prompt=None, **kwargs):
        if isinstance(content, list):
            if any(isinstance(item, bytes) for item in content):
                raise NotImplementedError("Ark frozen-role integration is text-only.")
            content = "\n".join(content)
        if not isinstance(content, str):
            raise TypeError("Ark engine expects text content.")
        return self._generate_text(content, system_prompt=system_prompt, **kwargs)

    def _generate_text(
        self,
        prompt: str,
        system_prompt=None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        response_format=None,
        **_: object,
    ):
        sys_prompt = system_prompt or self.system_prompt
        temperature = self.temperature if temperature is None else temperature
        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        top_p = self.top_p if top_p is None else top_p
        cache_key = self._cache_key(sys_prompt, prompt)
        if self.use_cache:
            cached = self._check_cache(cache_key)
            if cached is not None:
                return cached

        request = {
            "model": self.api_model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        # Ark deployments may not expose OpenAI beta.parse.  json_object is
        # the least provider-specific structured hint; downstream AgentFlow
        # code still performs strict Pydantic/schema and Game24 validation.
        if response_format is not None:
            request["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(**request)
        text = response.choices[0].message.content
        if self.use_cache:
            self._save_cache(cache_key, text)
        return text

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)
