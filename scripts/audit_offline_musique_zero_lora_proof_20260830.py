#!/usr/bin/env python3
"""CPU-only proof that the restored initial LoRA equals its disabled adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def tensor_descriptor(name: str, tensor: torch.Tensor) -> dict:
    tensor = tensor.detach().cpu().contiguous()
    raw = tensor.view(torch.uint8).numpy().tobytes()
    return {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": int(tensor.numel()),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--replay-pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    snapshot = torch.load(args.snapshot, map_location="cpu", weights_only=False)
    state = snapshot.get("lora_state")
    if not isinstance(state, dict):
        raise ValueError("snapshot lacks lora_state")
    keys = sorted(state)
    a_keys = [key for key in keys if ".lora_A." in key]
    b_keys = [key for key in keys if ".lora_B." in key]
    other = sorted(set(keys).difference(a_keys, b_keys))
    expected_b = {key.replace(".lora_A.", ".lora_B.") for key in a_keys}
    missing_b = sorted(expected_b.difference(b_keys))
    extra_b = sorted(set(b_keys).difference(expected_b))
    b_nonzero = {key: int(torch.count_nonzero(state[key]).item()) for key in b_keys}
    b_max_abs = {
        key: float(state[key].detach().float().abs().max().item()) if state[key].numel() else 0.0
        for key in b_keys
    }

    digest = hashlib.sha256()
    descriptors = []
    for key in keys:
        descriptor = tensor_descriptor(key, state[key])
        descriptors.append(descriptor)
        digest.update(json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    actual_hash = digest.hexdigest()
    if actual_hash != snapshot.get("lora_hash"):
        raise ValueError("snapshot LoRA hash does not match tensor contents")

    pack = torch.load(args.replay_pack, map_location="cpu", weights_only=False)
    advantages = pack["tensor_fields"]["advantages"]
    response_mask = pack["tensor_fields"]["response_mask"].bool()
    if not bool(torch.all(advantages == 0).item()):
        raise ValueError("terminal replay advantages are not all zero")
    rewards = pack["non_tensor_batch"]["rollout_reward_list"]
    rollout_rewards = {}
    for qid, rollout_id, reward in zip(
        pack["non_tensor_batch"]["prompt_id_list"],
        pack["non_tensor_batch"]["rollout_id_list"],
        rewards,
    ):
        key = str(rollout_id)
        previous = rollout_rewards.setdefault(key, (str(qid), float(reward)))
        if previous != (str(qid), float(reward)):
            raise ValueError("one rollout has inconsistent qid or reward across transitions")
    grouped = {}
    for qid, reward in rollout_rewards.values():
        grouped.setdefault(qid, []).append(reward)
    if len(rollout_rewards) != 1024 or len(grouped) != 128 or any(len(values) != 8 for values in grouped.values()):
        raise ValueError("replay does not contain exact 128x8 groups")
    if any(any(value != 0 for value in values) for values in grouped.values()):
        raise ValueError("terminal reward groups are not all zero")

    result = {
        "schema_version": 1,
        "kind": "offline_musique_terminal_zero_signal_exact_lora_proof",
        "snapshot": {
            "path": str(args.snapshot),
            "lora_hash": actual_hash,
            "tensor_count": len(keys),
            "lora_A_count": len(a_keys),
            "lora_B_count": len(b_keys),
            "other_adapter_tensor_count": len(other),
            "missing_lora_B_pairs": missing_b,
            "extra_lora_B_pairs": extra_b,
            "lora_B_nonzero_tensor_count": sum(value != 0 for value in b_nonzero.values()),
            "lora_B_total_nonzero_elements": sum(b_nonzero.values()),
            "lora_B_max_abs": max(b_max_abs.values(), default=0.0),
            "effective_lora_delta_exactly_zero": (
                len(a_keys) == 196
                and len(b_keys) == 196
                and not other
                and not missing_b
                and not extra_b
                and not any(b_nonzero.values())
            ),
        },
        "official_kl_loss": {
            "reference_policy": "VERL FSDP LoRA path: same actor with adapter disabled",
            "source": "verl/workers/fsdp_workers.py compute_ref_log_prob; actor.disable_adapter()",
            "loss_type": "low_var_kl",
            "formula": "exp(ref_logp - logp) - (ref_logp - logp) - 1, with configured clamp",
            "beta": 0.001,
            "response_mask_only": True,
            "entropy_coefficient": 0.0,
            "at_actor_equals_ref": {"kl": 0.0, "d_kl_d_logp": 0.0},
        },
        "terminal_pack": {
            "path": str(args.replay_pack),
            "question_group_count": len(grouped),
            "rollouts_per_group": 8,
            "all_zero_terminal_group_count": len(grouped),
            "effective_terminal_group_count": 0,
            "response_masked_token_count": int(response_mask.sum().item()),
            "terminal_advantage_abs_max": float(advantages.abs().max().item()),
            "terminal_advantage_nonzero_token_count": int((advantages[response_mask] != 0).sum().item()),
        },
        "decision": {
            "terminal_only_A": "verified_no_op",
            "reason": "terminal task advantage is exactly zero and initial actor equals adapter-disabled reference exactly",
            "optimizer_run": False,
            "post_A": "skipped; definitionally identical to frozen PRE",
            "next_phase": "authorized transition-aware B from the same initial snapshot and persisted pack",
        },
    }
    if not result["snapshot"]["effective_lora_delta_exactly_zero"]:
        raise ValueError("exact zero-LoRA proof failed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
