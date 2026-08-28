"""Stable logical routes for one local vLLM base model.

The pinned VERL/vLLM stack registers synchronized LoRA adapters with the
engine using an ephemeral numeric id.  AgentFlow roles should not depend on
that id: fixed roles always use ``qwen-base`` and only the main planner uses
``qwen-actor``.  The small JSON registry is runtime state and must live
outside the repository (the caller supplies its path).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

BASE_ROLE = "qwen-base"
ACTOR_ROLE = "qwen-actor"
ROUTE_SCHEMA_VERSION = 1


def route_state_path() -> str | None:
    """Return the explicitly configured runtime registry path."""

    value = os.environ.get("AGENTFLOW_ROLE_ROUTING_STATE", "").strip()
    return value or None


def resolve_role(role: str, actor_route: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Resolve a logical role without ever silently upgrading base to LoRA."""

    if role == BASE_ROLE:
        return {"role": BASE_ROLE, "model": BASE_ROLE, "adapter": None}
    if role != ACTOR_ROLE:
        raise ValueError(f"unsupported unified role: {role!r}")
    if not actor_route:
        raise LookupError("qwen-actor is unavailable before the first LoRA sync")
    return {
        "role": ACTOR_ROLE,
        "model": ACTOR_ROLE,
        "adapter": {
            "lora_name": str(actor_route["lora_name"]),
            "lora_int_id": int(actor_route["lora_int_id"]),
            "version": str(actor_route["version"]),
        },
    }


def write_actor_route(path: str, *, lora_name: str, lora_int_id: int, version: str, base_model: str) -> str:
    """Atomically publish the latest adapter route and return its content hash."""

    payload = {
        "schema_version": ROUTE_SCHEMA_VERSION,
        "base_role": BASE_ROLE,
        "actor_role": ACTOR_ROLE,
        "base_model": base_model,
        "actor": {
            "lora_name": str(lora_name),
            "lora_int_id": int(lora_int_id),
            "version": str(version),
        },
    }
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return hashlib.sha256(encoded).hexdigest()


def read_actor_route(path: str | None) -> dict[str, Any] | None:
    """Read and validate runtime route state; malformed state is unavailable."""

    if not path:
        return None
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        actor = payload["actor"]
        if payload.get("schema_version") != ROUTE_SCHEMA_VERSION:
            return None
        return {
            "lora_name": str(actor["lora_name"]),
            "lora_int_id": int(actor["lora_int_id"]),
            "version": str(actor["version"]),
            "base_model": str(payload.get("base_model", "")),
        }
    except (OSError, ValueError, TypeError, KeyError):
        return None
