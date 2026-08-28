"""Stable identifiers for AgentFlow rollout groups and replay metadata."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def canonical_prompt(sample: dict[str, Any]) -> str:
    """Return a stable prompt representation independent of batch UUIDs."""
    for key in ("question", "prompt", "input"):
        if sample.get(key) is not None:
            return str(sample[key]).strip()
    return json.dumps(_jsonable(sample), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_prompt_id(sample: dict[str, Any]) -> str:
    """Derive a repeatable prompt id without using a per-batch data UUID.

    Dataset/source identity is included when present, while the canonical
    prompt remains part of the key to avoid collisions between reused row ids.
    ``data_id`` and ``rollout_id`` are intentionally never used here.
    """
    extra = sample.get("extra_info") or {}
    dataset = (
        extra.get("dataset")
        or extra.get("benchmark")
        or sample.get("dataset")
        or sample.get("source")
        or "unknown-dataset"
    )
    source_id = (
        extra.get("question_id")
        or extra.get("problem_id")
        or extra.get("source_id")
        or extra.get("benchmark_pid")
        or extra.get("idx")
        or sample.get("id")
    )
    identity = {"dataset": str(dataset), "source_id": str(source_id) if source_id is not None else None,
                "prompt": canonical_prompt(sample)}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return f"prompt-{digest[:32]}"
