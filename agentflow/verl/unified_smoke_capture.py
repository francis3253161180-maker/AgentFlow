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
import random
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


def _lora_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Materialize the trainable LoRA tensors as an auditable CPU state."""
    state: dict[str, torch.Tensor] = {}
    for name, parameter in sorted(module.named_parameters(), key=lambda item: item[0]):
        if parameter.requires_grad and "lora_" in name.lower():
            _, tensor = _tensor_bytes(parameter)
            state[name] = tensor.clone()
    if not state:
        raise RuntimeError("LoRA state capture found no trainable LoRA tensors")
    return state


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


def _rng_snapshot() -> dict[str, Any]:
    """Capture worker-local RNG state without putting it in a text log."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "explicit_seeds": {
            key: os.getenv(key)
            for key in ("PYTHONHASHSEED", "AGENTFLOW_UNIFIED_SEED", "SEED")
            if os.getenv(key) is not None
        },
    }


def capture_behavior_snapshot(module: torch.nn.Module) -> dict[str, Any]:
    """Persist an exact, reloadable LoRA/RNG snapshot for a rollout-only run.

    This is opt-in and called from the actor worker after model construction and
    before the first rollout request.  Only trainable LoRA tensors are saved;
    the base model remains identified by the run metadata.
    """
    output = os.getenv("AGENTFLOW_BEHAVIOR_SNAPSHOT_PATH", "").strip()
    metadata_output = os.getenv("AGENTFLOW_BEHAVIOR_SNAPSHOT_METADATA_PATH", "").strip()
    if not output:
        return {"status": "disabled"}

    state = _lora_state(module)

    tensor_descriptors = []
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].contiguous()
        raw = tensor.view(torch.uint8).numpy().tobytes()
        descriptor = {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "numel": int(tensor.numel()),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        tensor_descriptors.append(descriptor)
        digest.update(json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    lora_hash = digest.hexdigest()
    payload = {
        "schema_version": 1,
        "kind": "agentflow_behavior_policy_snapshot",
        "lora_state": state,
        "rng_state": _rng_snapshot(),
        "lora_hash": lora_hash,
        "tensor_descriptors": tensor_descriptors,
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    result = {
        "status": "captured",
        "path": str(path),
        "lora_hash": lora_hash,
        "tensor_count": len(state),
        "total_numel": sum(int(item["numel"]) for item in tensor_descriptors),
    }
    if metadata_output:
        _write_json_atomic(Path(metadata_output), result)
    print(
        f"AGENTFLOW_BEHAVIOR_SNAPSHOT status=captured hash={lora_hash} tensors={len(state)}",
        flush=True,
    )
    return result


def restore_behavior_snapshot(module: torch.nn.Module) -> dict[str, Any]:
    """Restore an opt-in LoRA snapshot into an ordinary, pre-FSDP module."""
    source = os.getenv("AGENTFLOW_BEHAVIOR_SNAPSHOT_SOURCE_PATH", "").strip()
    if not source:
        return {"status": "disabled"}
    payload = torch.load(source, map_location="cpu", weights_only=False)
    state = payload.get("lora_state")
    expected = payload.get("lora_hash")
    if not isinstance(state, dict) or not expected:
        raise RuntimeError("invalid behavior snapshot payload")
    current = dict(module.state_dict())
    missing = []
    for name, tensor in state.items():
        if name not in current:
            missing.append(name)
            continue
        current[name] = tensor
    if missing:
        raise RuntimeError(f"behavior snapshot tensor names missing: {missing[:3]}")
    module.load_state_dict(current, strict=True)
    actual = _lora_hash(module)["hash"]
    if actual != expected:
        raise RuntimeError(f"behavior snapshot hash mismatch: expected={expected} actual={actual}")
    result = {
        "status": "restored",
        "source": source,
        "lora_hash": actual,
        "tensor_count": len(state),
    }
    print(
        f"AGENTFLOW_BEHAVIOR_SNAPSHOT status=restored hash={actual} tensors={len(state)}",
        flush=True,
    )
    return result


def verify_behavior_snapshot(module: torch.nn.Module) -> dict[str, Any]:
    """Verify a snapshot hash without copying CPU tensors into DTensors."""
    source = os.getenv("AGENTFLOW_BEHAVIOR_SNAPSHOT_SOURCE_PATH", "").strip()
    if not source:
        return {"status": "disabled"}
    payload = torch.load(source, map_location="cpu", weights_only=False)
    expected = payload.get("lora_hash")
    state = payload.get("lora_state")
    if not isinstance(state, dict) or not expected:
        raise RuntimeError("invalid behavior snapshot payload")
    actual = _lora_hash(module)["hash"]
    if actual != expected:
        raise RuntimeError(f"behavior snapshot hash mismatch: expected={expected} actual={actual}")
    result = {
        "status": "verified",
        "source": source,
        "lora_hash": actual,
        "tensor_count": len(state),
    }
    print(
        f"AGENTFLOW_BEHAVIOR_SNAPSHOT status=verified hash={actual} tensors={len(state)}",
        flush=True,
    )
    return result


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


def validate_replay_pack_for_update(
    pack: dict[str, Any],
    *,
    expected_model_path: str,
    expected_rollout_n: int,
    expected_temperature: float,
    expected_seed: str | int | None,
    current_lora_hash: str,
) -> dict[str, Any]:
    """Fail closed unless a replay pack is authentic for this update.

    Replay data is an input to an optimizer, so structural and behavior
    identity checks are intentionally stricter than the diagnostic capture
    path.  In particular, a missing identity/hash is not treated as a match.
    """
    if not isinstance(pack, dict) or pack.get("kind") != "agentflow_unified_authentic_pre_update_replay_pack":
        raise ValueError("unsupported replay pack kind")
    schema_version = int(pack.get("schema_version", 0))
    if schema_version < 2:
        raise ValueError("replay pack schema is too old")
    metadata = pack.get("metadata")
    tensors = pack.get("tensor_fields")
    non_tensor = pack.get("non_tensor_batch", {})
    meta_info = pack.get("meta_info", {})
    if not isinstance(metadata, dict) or not isinstance(tensors, dict) or not isinstance(non_tensor, dict):
        raise ValueError("replay pack has invalid field containers")

    required_tensors = {
        "prompts", "responses", "input_ids", "attention_mask", "position_ids",
        "response_mask", "old_log_probs", "token_level_scores", "token_level_rewards",
        "advantages", "returns", "is_drop_mask",
    }
    missing_tensors = sorted(required_tensors.difference(tensors))
    required_non_tensor = {"prompt_id_list", "data_id_list", "rollout_id_list", "rollout_reward_list"}
    missing_non_tensor = sorted(required_non_tensor.difference(non_tensor))
    if missing_tensors or missing_non_tensor:
        raise ValueError(
            f"replay pack missing fields: tensors={missing_tensors} non_tensor={missing_non_tensor}"
        )
    inventory = pack.get("field_inventory")
    if not isinstance(inventory, dict):
        raise ValueError("replay pack is missing field inventory")
    if set(inventory.get("tensor_fields", [])) != set(tensors):
        raise ValueError("replay pack tensor field inventory mismatch")
    if set(inventory.get("non_tensor_fields", [])) != set(non_tensor):
        raise ValueError("replay pack non-tensor field inventory mismatch")

    expected_digest = pack.get("captured_field_digest")
    actual_digest = _field_digest(tensors, non_tensor, meta_info)
    if not expected_digest or expected_digest != actual_digest:
        raise ValueError("replay pack field digest mismatch")

    identity = ("model_path", "temperature", "rollout_n", "seed")
    if any(key not in metadata or metadata[key] in (None, "") for key in identity):
        raise ValueError("replay pack is missing behavior identity metadata")
    if str(metadata["model_path"]) != str(expected_model_path):
        raise ValueError("replay pack model path mismatch")
    if abs(float(metadata["temperature"]) - float(expected_temperature)) > 1e-12:
        raise ValueError("replay pack temperature mismatch")
    if int(metadata["rollout_n"]) != int(expected_rollout_n):
        raise ValueError("replay pack rollout_n mismatch")
    if expected_seed is not None and str(metadata["seed"]) != str(expected_seed):
        raise ValueError("replay pack seed mismatch")

    expected_lora_hash = metadata.get("lora_pre_hash")
    if not expected_lora_hash:
        snapshot = metadata.get("behavior_snapshot")
        if isinstance(snapshot, dict):
            expected_lora_hash = snapshot.get("lora_hash") or snapshot.get("hash")
    if not expected_lora_hash or not current_lora_hash:
        raise ValueError("replay pack lacks verifiable LoRA identity")
    if str(expected_lora_hash) != str(current_lora_hash):
        raise ValueError("replay pack LoRA hash mismatch")

    drop_mask = tensors["is_drop_mask"]
    if isinstance(drop_mask, torch.Tensor) and bool(torch.any(drop_mask).item()):
        raise ValueError("replay pack contains truncated/dropped transitions")
    batch_size = int(pack.get("batch_size", -1))
    if batch_size < 1 or int(tensors["input_ids"].shape[0]) != batch_size:
        raise ValueError("replay pack batch size mismatch")
    return {
        "status": "validated",
        "schema_version": schema_version,
        "batch_size": batch_size,
        "field_digest": actual_digest,
        "lora_hash": str(current_lora_hash),
    }


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
    numeric_grad_norm = float(grad_norm)
    if not np.isfinite(numeric_grad_norm):
        raise RuntimeError(f"non-finite actor gradient norm: {numeric_grad_norm}")
    _POST_HASH = _lora_hash(module)
    print(
        f"UNIFIED_LORA_CHECKSUM stage=post hash={_POST_HASH['hash']} tensors={_POST_HASH['tensor_count']} ",
        flush=True,
    )
    payload = {
        "status": "post_captured",
        "pre": _PRE_HASH,
        "post": _POST_HASH,
        "grad_norm": numeric_grad_norm,
        "hash_changed": bool(_PRE_HASH and _PRE_HASH["hash"] != _POST_HASH["hash"]),
        "changed_tensor_count": sum(
            before["sha256"] != after["sha256"]
            for before, after in zip(_PRE_HASH["tensors"], _POST_HASH["tensors"])
        )
        if _PRE_HASH and len(_PRE_HASH["tensors"]) == len(_POST_HASH["tensors"])
        else None,
    }
    output = os.getenv("AGENTFLOW_LORA_CHECKSUM_PATH", "").strip()
    if output:
        _write_json_atomic(Path(output), payload)

    snapshot_output = os.getenv("AGENTFLOW_LORA_POST_SNAPSHOT_PATH", "").strip()
    if snapshot_output:
        snapshot = {
            "schema_version": 1,
            "kind": "agentflow_post_optimizer_lora_snapshot",
            "lora_state": _lora_state(module),
            "lora_hash": _POST_HASH["hash"],
            "tensor_descriptors": _POST_HASH["tensors"],
            "grad_norm": numeric_grad_norm,
        }
        path = Path(snapshot_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        torch.save(snapshot, temporary)
        os.replace(temporary, path)


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
    tensors = _batch_fields(data)
    # A replay pack is useful for proving the update path only when it carries
    # authentic algorithm signal.  Do not consume the one-shot capture slot on
    # an all-zero warm-up batch; the next real pre-update batch can then be
    # captured without fabricating rewards or advantages.
    signal = tensors.get("advantages")
    if isinstance(signal, torch.Tensor):
        has_signal = bool(torch.any(torch.abs(signal.float()) > 0).item())
    else:
        has_signal = False
        for key in ("token_level_rewards", "token_level_scores"):
            candidate = tensors.get(key)
            if isinstance(candidate, torch.Tensor) and bool(torch.any(torch.abs(candidate.float()) > 0).item()):
                has_signal = True
                break
    if not has_signal:
        print("UNIFIED_REPLAY_CAPTURE skipped=zero_reward_or_advantage", flush=True)
        return
    _REPLAY_CAPTURED = True
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
            "ppo_epochs": os.getenv("AGENTFLOW_UNIFIED_PPO_EPOCHS", ""),
            "max_response_length": os.getenv("AGENTFLOW_UNIFIED_MAX_RESPONSE_LENGTH", ""),
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


def write_replay_pack_from_dataproto(data: Any, output: str, metadata: dict[str, Any]) -> dict[str, Any]:
    """Write a replay pack from an already-collected DataProto.

    The caller must have built the batch from saved rollout triplets and may
    recompute old log-probabilities/advantages, but this helper never generates
    text and never calls an optimizer.
    """
    tensors = _batch_fields(data)
    non_tensor = _cpu_value(getattr(data, "non_tensor_batch", {}))
    meta_info = _json_safe(getattr(data, "meta_info", {}))
    required = {
        "prompts", "responses", "input_ids", "attention_mask", "position_ids",
        "response_mask", "old_log_probs", "token_level_scores", "token_level_rewards",
        "advantages", "returns", "is_drop_mask",
    }
    missing = sorted(required.difference(tensors))
    if missing:
        raise ValueError(f"replay pack missing required tensor fields: {missing}")
    payload = {
        "schema_version": 3,
        "kind": "agentflow_unified_authentic_pre_update_replay_pack",
        "metadata": _json_safe(metadata),
        "field_inventory": {
            "tensor_fields": sorted(tensors),
            "non_tensor_fields": sorted(non_tensor),
            "meta_info_fields": sorted(meta_info) if isinstance(meta_info, dict) else [],
        },
        "batch_size": int(len(data)),
        "captured_field_digest": _field_digest(tensors, non_tensor, meta_info),
        "tensor_fields": tensors,
        "non_tensor_batch": non_tensor,
        "meta_info": meta_info,
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    result = {
        "status": "written",
        "path": str(path),
        "batch_size": payload["batch_size"],
        "field_digest": payload["captured_field_digest"],
        "tensor_fields": payload["field_inventory"]["tensor_fields"],
    }
    print(f"AGENTFLOW_REPLAY_PACK status=written batch={payload['batch_size']} digest={payload['captured_field_digest']}", flush=True)
    return result
