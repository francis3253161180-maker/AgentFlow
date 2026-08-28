#!/usr/bin/env python3
"""Validate an authentic runtime Replay Pack without rollout or model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from agentflow.verl.unified_smoke_capture import _field_digest, _json_safe


REQUIRED_TENSORS = {
    "input_ids",
    "responses",
    "response_mask",
    "old_log_probs",
    "advantages",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _equal_value(left, right) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and tuple(left.shape) == tuple(right.shape) and torch.equal(left, right)
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_equal_value(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_equal_value(a, b) for a, b in zip(left, right))
    return left == right


def validate(path: Path) -> dict:
    pack = torch.load(path, map_location="cpu", weights_only=False)
    assert pack.get("kind") == "agentflow_unified_authentic_pre_update_replay_pack"
    tensors = pack.get("tensor_fields", {})
    non_tensor = pack.get("non_tensor_batch", {})
    meta_info = pack.get("meta_info", {})
    missing = sorted(REQUIRED_TENSORS - tensors.keys())
    assert not missing, f"missing required runtime tensors: {missing}"
    assert tensors["input_ids"].dtype == torch.int64, "input_ids must preserve token ids as int64"
    assert tensors["responses"].dtype == torch.int64, "responses must preserve token ids as int64"
    assert tensors["response_mask"].shape == tensors["old_log_probs"].shape
    assert tensors["response_mask"].shape == tensors["advantages"].shape
    captured_digest = pack.get("captured_field_digest")
    loaded_digest = _field_digest(tensors, non_tensor, meta_info)
    assert captured_digest == loaded_digest, "captured field digest mismatch"

    # A second save/load in this fresh validator process checks the actual
    # serialized representation field-by-field without invoking a model.
    with tempfile.TemporaryDirectory(prefix="agentflow-replay-roundtrip-") as directory:
        copy_path = Path(directory) / "roundtrip.pt"
        torch.save(pack, copy_path)
        roundtrip = torch.load(copy_path, map_location="cpu", weights_only=False)
    assert _equal_value(pack, roundtrip), "field-by-field roundtrip mismatch"

    mask = tensors["response_mask"].bool()
    advantages = tensors["advantages"].float()
    old_log_probs = tensors["old_log_probs"].float()
    reward_key = next((key for key in ("token_level_rewards", "token_level_scores", "rewards") if key in tensors), None)
    reward_summary = None
    if reward_key:
        rewards = tensors[reward_key].float()
        reward_summary = {
            "field": reward_key,
            "shape": list(rewards.shape),
            "mean": float(rewards.mean().item()),
            "min": float(rewards.min().item()),
            "max": float(rewards.max().item()),
        }
    uid_fields = [key for key in ("uid", "group_id", "request_id") if key in non_tensor]
    dry_run = {
        "status": "ok",
        "rollout_generation": False,
        "external_calls": False,
        "batch_size": int(tensors["input_ids"].shape[0]),
        "response_token_count": int(mask.sum().item()),
        "advantage_mean_on_response": float(advantages[mask].mean().item()) if mask.any() else 0.0,
        "old_log_prob_mean_on_response": float(old_log_probs[mask].mean().item()) if mask.any() else 0.0,
        "reward": reward_summary,
        "uid_fields": uid_fields,
    }
    return {
        "status": "ok",
        "pack_sha256": sha256_file(path),
        "token_ids_available": True,
        "field_inventory": pack.get("field_inventory", {}),
        "captured_field_digest": captured_digest,
        "roundtrip_field_equal": True,
        "offline_replay_dry_run": dry_run,
        "metadata": _json_safe(pack.get("metadata", {})),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.pack)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

