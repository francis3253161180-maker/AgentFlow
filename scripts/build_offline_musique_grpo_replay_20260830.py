#!/usr/bin/env python3
"""Build an authentic terminal-only GRPO replay pack from persisted rollouts."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentflow.offline_musique import sha256_file, stable_json_hash
from agentflow.verl.advantage import compute_rollout_group_advantage
from agentflow.verl.unified_smoke_capture import write_replay_pack_from_dataproto
from verl import DataProto


def left_pad(values: list[int], width: int, pad_id: int) -> tuple[list[int], list[int]]:
    if len(values) > width:
        raise ValueError("prompt exceeds frozen replay width")
    padding = width - len(values)
    return [pad_id] * padding + values, [0] * padding + [1] * len(values)


def right_pad(values: list, width: int, pad_value) -> tuple[list, list[int]]:
    if len(values) > width:
        raise ValueError("response exceeds frozen replay width")
    padding = width - len(values)
    return values + [pad_value] * padding, [1] * len(values) + [0] * padding


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detail", type=Path, required=True)
    parser.add_argument("--runner-summary", type=Path, required=True)
    parser.add_argument("--transition-audit", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--behavior-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--prompt-width", type=int, default=1536)
    parser.add_argument("--response-width", type=int, default=256)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    detail = json.loads(args.detail.read_text(encoding="utf-8"))
    runner = json.loads(args.runner_summary.read_text(encoding="utf-8"))
    transition_audit = json.loads(args.transition_audit.read_text(encoding="utf-8"))
    snapshot = torch.load(args.behavior_snapshot, map_location="cpu", weights_only=False)
    snapshot_hash = snapshot.get("lora_hash")
    if not snapshot_hash or not isinstance(snapshot.get("lora_state"), dict):
        raise SystemExit("invalid behavior-policy snapshot")
    from safetensors.torch import load_file

    adapter_state = load_file(args.adapter / "adapter_model.safetensors", device="cpu")
    mapped_snapshot = {
        name.replace(".default.weight", ".weight"): tensor
        for name, tensor in snapshot["lora_state"].items()
    }
    if set(mapped_snapshot) != set(adapter_state) or any(
        not torch.equal(mapped_snapshot[name], adapter_state[name]) for name in adapter_state
    ):
        raise SystemExit("vLLM adapter tensors do not equal the behavior-policy snapshot")
    if detail["configuration"] != runner["configuration"]:
        raise SystemExit("runner/raw configuration mismatch")
    if detail["configuration"]["n"] != 8 or len(detail["qids"]) != 128:
        raise SystemExit("replay source must be fixed 128x8 train pack")
    if transition_audit["outcome"]["grounded_positive_count"] != sum(
        int(row["reward_detail"]["reward"]) for row in detail["trajectories"]
    ):
        raise SystemExit("transition audit and raw reward totals differ")

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    pad_id = int(tokenizer.pad_token_id)
    prompts: list[list[int]] = []
    prompt_masks: list[list[int]] = []
    responses: list[list[int]] = []
    response_masks: list[list[int]] = []
    old_log_probs: list[list[float]] = []
    prompt_ids: list[str] = []
    data_ids: list[str] = []
    rollout_ids: list[str] = []
    rollout_rewards: list[float] = []
    turn_indexes: list[int] = []
    modes: list[str] = []
    trajectory_ids: list[str] = []
    raw_prompt_lengths: list[int] = []
    raw_response_lengths: list[int] = []
    logprob_deltas: list[float] = []
    by_qid_rollouts: dict[str, dict[int, float]] = defaultdict(dict)
    mode_token_counts: dict[str, list[int]] = defaultdict(list)

    for trajectory in detail["trajectories"]:
        qid = trajectory["qid"]
        rollout_index = int(trajectory["rollout_index"])
        reward = float(trajectory["reward_detail"]["reward"])
        if rollout_index in by_qid_rollouts[qid]:
            raise ValueError(f"duplicate rollout slot: {qid}/{rollout_index}")
        by_qid_rollouts[qid][rollout_index] = reward
        for turn_index, transition in enumerate(trajectory["transitions"]):
            metadata = transition["token_logprob_metadata"]
            response_ids = [int(value) for value in metadata["response_token_ids"]]
            response_logprobs = [float(value) for value in metadata["response_token_logprobs"]]
            if not response_ids or len(response_ids) != len(response_logprobs):
                raise ValueError(f"invalid response/logprob row: {trajectory['trajectory_id']}/{turn_index}")
            if any(not math.isfinite(value) for value in response_logprobs):
                raise ValueError("non-finite old logprob")
            rendered = detail["audit_prompts"][transition["prompt_hash"]]
            prompt = tokenizer.encode(rendered, add_special_tokens=False)
            if len(prompt) != int(metadata["prompt_tokens"]):
                raise ValueError("re-tokenized prompt length differs from behavior prompt")
            padded_prompt, prompt_mask = left_pad(prompt, args.prompt_width, pad_id)
            padded_response, response_mask = right_pad(response_ids, args.response_width, pad_id)
            padded_logprobs, _ = right_pad(response_logprobs, args.response_width, 0.0)
            prompts.append(padded_prompt)
            prompt_masks.append(prompt_mask)
            responses.append(padded_response)
            response_masks.append(response_mask)
            old_log_probs.append(padded_logprobs)
            prompt_ids.append(qid)
            data_ids.append(qid)
            rollout_ids.append(trajectory["rollout_id"])
            rollout_rewards.append(reward)
            turn_indexes.append(turn_index)
            modes.append(transition["mode"])
            trajectory_ids.append(trajectory["trajectory_id"])
            raw_prompt_lengths.append(len(prompt))
            raw_response_lengths.append(len(response_ids))
            logprob_deltas.append(abs(float(metadata["selected_vs_cumulative_logprob_delta"])))
            mode_token_counts[transition["mode"]].append(len(response_ids))

    if set(by_qid_rollouts) != set(detail["qids"]):
        raise ValueError("replay qids differ from raw fixed qids")
    for qid, slots in by_qid_rollouts.items():
        if sorted(slots) != list(range(8)):
            raise ValueError(f"incomplete rollout group: {qid}")

    prompt_tensor = torch.tensor(prompts, dtype=torch.long)
    prompt_mask_tensor = torch.tensor(prompt_masks, dtype=torch.long)
    response_tensor = torch.tensor(responses, dtype=torch.long)
    response_mask_tensor = torch.tensor(response_masks, dtype=torch.long)
    input_ids = torch.cat((prompt_tensor, response_tensor), dim=-1)
    attention_mask = torch.cat((prompt_mask_tensor, response_mask_tensor), dim=-1)
    position_ids = torch.clamp(torch.cumsum(attention_mask, dim=-1) - 1, min=0)
    transition_count = len(prompts)
    token_level_scores = torch.zeros((transition_count, args.response_width), dtype=torch.bfloat16)
    batch = TensorDict(
        {
            "prompts": prompt_tensor,
            "responses": response_tensor,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "response_mask": response_mask_tensor,
            "old_log_probs": torch.tensor(old_log_probs, dtype=torch.float32),
            "token_level_scores": token_level_scores,
            "token_level_rewards": token_level_scores.clone(),
            "is_drop_mask": torch.zeros(transition_count, dtype=torch.bool),
        },
        batch_size=[transition_count],
    )
    data = DataProto(batch=batch)
    data.non_tensor_batch.update(
        {
            "prompt_id_list": np.asarray(prompt_ids, dtype=object),
            "data_id_list": np.asarray(data_ids, dtype=object),
            "rollout_id_list": np.asarray(rollout_ids, dtype=object),
            "rollout_reward_list": np.asarray(rollout_rewards, dtype=np.float32),
            "turn_index_list": np.asarray(turn_indexes, dtype=np.int64),
            "policy_mode_list": np.asarray(modes, dtype=object),
            "trajectory_id_list": np.asarray(trajectory_ids, dtype=object),
            "uid": np.asarray(prompt_ids, dtype=object),
        }
    )
    data.meta_info["temperature"] = float(detail["configuration"]["temperature"])
    data.meta_info["multi_turn"] = True
    compute_rollout_group_advantage(data, rollout_n=8, normalize_by_std=True)

    response_mask_bool = data.batch["response_mask"].bool()
    advantages = data.batch["advantages"][response_mask_bool].float()
    masked_old = data.batch["old_log_probs"][response_mask_bool].float()
    for row_index, _rollout_id in enumerate(rollout_ids):
        expected = rollout_rewards[row_index]
        group_values = list(by_qid_rollouts[data_ids[row_index]].values())
        mean = statistics.fmean(group_values)
        std = statistics.stdev(group_values) if len(group_values) > 1 else 1.0
        expected_advantage = (expected - mean) / (std + 1e-6)
        actual_values = data.batch["advantages"][row_index][response_mask_bool[row_index]].unique()
        if len(actual_values) != 1 or abs(float(actual_values.item()) - expected_advantage) > 1e-6:
            raise ValueError("rollout advantage was not broadcast exactly to every response token")

    metadata = {
        "source_run_id": "offline-musique-grpo-n8-20260830",
        "model_path": str(args.model.resolve()),
        "temperature": float(detail["configuration"]["temperature"]),
        "rollout_n": 8,
        "seed": str(detail["configuration"]["seed"]),
        "scorer": "outcome_v2_exact_set local deterministic terminal reward only",
        "transition_diagnostic_training_weight": 0,
        "ppo_epochs": 1,
        "lora_pre_hash": snapshot_hash,
        "behavior_snapshot_path": str(args.behavior_snapshot.resolve()),
        "behavior_snapshot_sha256": sha256_file(args.behavior_snapshot),
        "adapter_weights_sha256": sha256_file(args.adapter / "adapter_model.safetensors"),
        "adapter_tensors_equal_behavior_snapshot": True,
        "source_raw_sha256": sha256_file(args.detail),
        "ordered_qids_sha256": stable_json_hash(detail["qids"]),
        "optimizer_steps_before_pack": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_result = write_replay_pack_from_dataproto(data, str(args.output), metadata)
    histogram = Counter(int(sum(slots.values())) for slots in by_qid_rollouts.values())
    summary = {
        "schema_version": 1,
        "experiment": "offline_musique_terminal_only_grpo_replay",
        "training_weight": {"outcome_v2_exact_set": 1, "F1_F2_delta_diagnostics": 0},
        "question_count": len(by_qid_rollouts),
        "rollout_n": 8,
        "rollout_count": len(detail["trajectories"]),
        "transition_count": transition_count,
        "group_histogram": {f"{k}/8": histogram[k] for k in range(9)},
        "effective_mixed_group_count": sum(0 < sum(slots.values()) < 8 for slots in by_qid_rollouts.values()),
        "effective_group_rate": sum(0 < sum(slots.values()) < 8 for slots in by_qid_rollouts.values()) / len(by_qid_rollouts),
        "advantage": {
            "count": int(advantages.numel()),
            "min": float(advantages.min().item()),
            "max": float(advantages.max().item()),
            "mean": float(advantages.mean().item()),
            "std": float(advantages.std(unbiased=False).item()),
            "nonzero_token_count": int(torch.count_nonzero(advantages).item()),
            "broadcast_to_all_response_tokens_verified": True,
        },
        "response": {
            "prompt_width": args.prompt_width,
            "response_width": args.response_width,
            "prompt_length_min_max_mean": [min(raw_prompt_lengths), max(raw_prompt_lengths), statistics.fmean(raw_prompt_lengths)],
            "response_length_min_max_mean": [min(raw_response_lengths), max(raw_response_lengths), statistics.fmean(raw_response_lengths)],
            "response_token_count": int(response_mask_bool.sum().item()),
            "response_mask_matches_recorded_lengths": int(response_mask_bool.sum().item()) == sum(raw_response_lengths),
            "old_logprob_min_max_mean": [float(masked_old.min()), float(masked_old.max()), float(masked_old.mean())],
            "max_selected_vs_cumulative_logprob_delta": max(logprob_deltas),
            "mode_transition_counts": dict(sorted(Counter(modes).items())),
            "mode_response_token_counts": {key: sum(values) for key, values in sorted(mode_token_counts.items())},
        },
        "grouping_checks": {
            "exactly_128_qids": len(by_qid_rollouts) == 128,
            "exactly_8_rollouts_per_qid": all(sorted(slots) == list(range(8)) for slots in by_qid_rollouts.values()),
            "all_groups_preserved_without_resampling": True,
            "all_transitions_preserved_without_drop": True,
        },
        "artifacts": {
            "source_raw_path": str(args.detail.resolve()),
            "source_raw_sha256": sha256_file(args.detail),
            "transition_audit_path": str(args.transition_audit.resolve()),
            "transition_audit_sha256": sha256_file(args.transition_audit),
            "replay_pack_path": str(args.output.resolve()),
            "replay_pack_sha256": sha256_file(args.output),
            "field_digest": write_result["field_digest"],
            "behavior_snapshot_path": str(args.behavior_snapshot.resolve()),
            "behavior_snapshot_sha256": sha256_file(args.behavior_snapshot),
            "behavior_lora_hash": snapshot_hash,
            "adapter_weights_sha256": sha256_file(args.adapter / "adapter_model.safetensors"),
            "adapter_tensors_equal_behavior_snapshot": True,
        },
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
