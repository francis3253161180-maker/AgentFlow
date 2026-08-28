"""Opt-in evidence capture for the unified-model infrastructure smoke.

This module is deliberately inert unless the two output environment variables
are set.  It is imported by the pinned VERL actor worker through the small
backport patch in ``patches/``.  The replay file is a torch serialization of
the actual DataProto presented to ``update_policy``; it is not reconstructed
from rollout JSON.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


_PRE_HASH: dict[str, Any] | None = None
_POST_HASH: dict[str, Any] | None = None
_REPLAY_CAPTURED = False


def _enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _cpu_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {str(key): _cpu_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_value(item) for item in value)
    return value


def _json_safe(value: Any, key: str = "") -> Any:
    lower_key = key.lower()
    if any(secret in lower_key for secret in ("api_key", "authorization", "password", "secret")):
        return "<redacted>"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_safe(item, key) for item in value.tolist()]
    if isinstance(value, torch.Tensor):
        return {"type": "torch.Tensor", "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {str(k): _json_safe(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, key) for item in value]
    return repr(value)


def _tensor_bytes(value: Any) -> tuple[Any, torch.Tensor]:
    """Return a local/full tensor suitable for deterministic hashing."""
    tensor = value
    if hasattr(tensor, "full_tensor"):
        tensor = tensor.full_tensor()
    elif hasattr(tensor, "to_local") and not isinstance(tensor, torch.Tensor):
        tensor = tensor.to_local()
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"unsupported parameter type: {type(value)!r}")
    tensor = tensor.detach().cpu().contiguous()
    raw = tensor.view(torch.uint8).numpy().tobytes()
    return raw, tensor


def _lora_hash(module: torch.nn.Module) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for name, parameter in sorted(module.named_parameters(), key=lambda item: item[0]):
        if not parameter.requires_grad or "lora_" not in name.lower():
            continue
        raw, tensor = _tensor_bytes(parameter)
        descriptor = {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "numel": int(tensor.numel()),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        entries.append(descriptor)
    aggregate = hashlib.sha256()
    for entry in entries:
        aggregate.update(json.dumps(entry, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return {
        "hash": aggregate.hexdigest(),
        "tensor_count": len(entries),
        "total_numel": sum(item["numel"] for item in entries),
        "tensors": entries,
    }


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _field_digest(tensors: dict[str, Any], non_tensor: dict[str, Any], meta_info: Any) -> str:
    digest = hashlib.sha256()
    for namespace, values in (("tensor", tensors), ("non_tensor", non_tensor), ("meta_info", meta_info)):
        if not isinstance(values, dict):
            values = {"value": values}
        for key in sorted(values):
            value = values[key]
            if namespace == "tensor":
                raw, tensor = _tensor_bytes(value)
                descriptor = {"key": key, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
                digest.update(namespace.encode("utf-8"))
                digest.update(json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8"))
                digest.update(raw)
            else:
                digest.update(namespace.encode("utf-8"))
                digest.update(key.encode("utf-8"))
                digest.update(json.dumps(_json_safe(value, key), ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def capture_lora_pre(module: torch.nn.Module) -> None:
    global _PRE_HASH
    if not _enabled("AGENTFLOW_LORA_CHECKSUM_ENABLED") or _PRE_HASH is not None:
        return
    _PRE_HASH = _lora_hash(module)
    print(
        f"UNIFIED_LORA_CHECKSUM stage=pre hash={_PRE_HASH['hash']} tensors={_PRE_HASH['tensor_count']} ",
        flush=True,
    )
    output = os.getenv("AGENTFLOW_LORA_CHECKSUM_PATH", "").strip()
    if output:
        _write_json_atomic(Path(output), {"status": "pre_captured", "pre": _PRE_HASH})


def capture_lora_post(module: torch.nn.Module, grad_norm: float) -> None:
    global _POST_HASH
    if not _enabled("AGENTFLOW_LORA_CHECKSUM_ENABLED") or _POST_HASH is not None:
        return
    if not float(grad_norm) > 0.0:
        return
    _POST_HASH = _lora_hash(module)
    print(
        f"UNIFIED_LORA_CHECKSUM stage=post hash={_POST_HASH['hash']} tensors={_POST_HASH['tensor_count']} ",
        flush=True,
    )
    output = os.getenv("AGENTFLOW_LORA_CHECKSUM_PATH", "").strip()
    if not output:
        return
    payload = {
        "status": "post_captured",
        "pre": _PRE_HASH,
        "post": _POST_HASH,
        "grad_norm": float(grad_norm),
        "hash_changed": bool(_PRE_HASH and _PRE_HASH["hash"] != _POST_HASH["hash"]),
        "changed_tensor_count": sum(
            before["sha256"] != after["sha256"]
            for before, after in zip(_PRE_HASH["tensors"], _POST_HASH["tensors"])
        )
        if _PRE_HASH and len(_PRE_HASH["tensors"]) == len(_POST_HASH["tensors"])
        else None,
    }
    _write_json_atomic(Path(output), payload)


def _batch_fields(data: Any) -> dict[str, Any]:
    if data.batch is None:
        return {}
    return {str(key): _cpu_value(value) for key, value in data.batch.items()}


def _route_snapshot() -> Any:
    route_path = os.getenv("AGENTFLOW_ROLE_ROUTING_STATE", "").strip()
    if not route_path or not Path(route_path).exists():
        return None
    try:
        return json.loads(Path(route_path).read_text(encoding="utf-8"))
    except Exception as exc:  # evidence capture must never break training
        return {"read_error": type(exc).__name__}


def capture_replay_pre_update(data: Any) -> None:
    global _REPLAY_CAPTURED
    if not _enabled("AGENTFLOW_REPLAY_CAPTURE_ENABLED") or _REPLAY_CAPTURED:
        return
    output = os.getenv("AGENTFLOW_REPLAY_PACK_PATH", "").strip()
    if not output:
        return
    _REPLAY_CAPTURED = True
    tensors = _batch_fields(data)
    non_tensor = _cpu_value(getattr(data, "non_tensor_batch", {}))
    meta_info = _json_safe(getattr(data, "meta_info", {}))
    captured_field_digest = _field_digest(tensors, non_tensor, meta_info)
    try:
        batch_size = len(data)
    except TypeError:
        batch_size = next(iter(tensors.values())).shape[0] if tensors else 0
    payload = {
        "schema_version": 2,
        "kind": "agentflow_unified_authentic_pre_update_replay_pack",
        "metadata": {
            "source_run_id": os.getenv("AGENTFLOW_UNIFIED_SMOKE_RUN_ID", ""),
            "model_path": os.getenv("AGENTFLOW_UNIFIED_MODEL_PATH", ""),
            "temperature": os.getenv("AGENTFLOW_UNIFIED_TEMPERATURE", ""),
            "rollout_n": os.getenv("AGENTFLOW_UNIFIED_ROLLOUT_N", ""),
            "seed": os.getenv("AGENTFLOW_UNIFIED_SEED", ""),
            "scorer": os.getenv("AGENTFLOW_UNIFIED_SCORER", "hybrid; external disabled"),
            "lora_pre_hash": _PRE_HASH["hash"] if _PRE_HASH else None,
            "role_route_state": _route_snapshot(),
        },
        "field_inventory": {
            "tensor_fields": sorted(tensors),
            "non_tensor_fields": sorted(non_tensor),
            "meta_info_fields": sorted(meta_info) if isinstance(meta_info, dict) else [],
        },
        "batch_size": int(batch_size),
        "captured_field_digest": captured_field_digest,
        "tensor_fields": tensors,
        "non_tensor_batch": non_tensor,
        "meta_info": meta_info,
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
