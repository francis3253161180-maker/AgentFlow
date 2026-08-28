"""AgentFlow rollout-level GRPO advantage computation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch


def compute_rollout_group_advantage(
    data: Any,
    *,
    rollout_n: int,
    normalize_by_std: bool = True,
    epsilon: float = 1e-6,
) -> Any:
    """Compute GRPO over unique rollout outcomes, then broadcast to transitions.

    A rollout may produce multiple actor transitions.  Group statistics are
    therefore built from one reward per unique ``rollout_id`` and not from
    transition rows.  Every transition from that rollout receives the same
    scalar advantage over its response mask.
    """
    required = ("prompt_id_list", "data_id_list", "rollout_id_list", "rollout_reward_list")
    missing = [key for key in required if key not in data.non_tensor_batch]
    if missing:
        raise ValueError(f"rollout-level GRPO metadata missing: {missing}")

    prompt_ids = data.non_tensor_batch["prompt_id_list"]
    data_ids = data.non_tensor_batch["data_id_list"]
    rollout_ids = data.non_tensor_batch["rollout_id_list"]
    rewards = data.non_tensor_batch["rollout_reward_list"]
    if not (len(prompt_ids) == len(data_ids) == len(rollout_ids) == len(rewards) == len(data)):
        raise ValueError("rollout-level GRPO metadata length mismatch")

    rollout_records: dict[tuple[str, str, str], float] = {}
    group_rollouts: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    for prompt_id, data_id, rollout_id, reward in zip(prompt_ids, data_ids, rollout_ids, rewards):
        group_key = (str(prompt_id), str(data_id))
        rollout_key = str(rollout_id)
        reward_value = float(reward)
        record_key = (group_key[0], group_key[1], rollout_key)
        prior = rollout_records.get(record_key)
        if prior is not None and abs(prior - reward_value) > epsilon:
            raise ValueError(f"inconsistent reward for rollout {rollout_key}")
        rollout_records[record_key] = reward_value
        group_rollouts[group_key].add(rollout_key)

    incomplete = {key: len(values) for key, values in group_rollouts.items() if len(values) != rollout_n}
    if incomplete:
        raise ValueError(f"incomplete rollout groups cannot reach advantage/update: {incomplete}")

    group_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (prompt_id, data_id, _rollout_id), reward in rollout_records.items():
        group_values[(prompt_id, data_id)].append(reward)

    row_advantages = torch.zeros(len(data), dtype=torch.float32, device=data.batch["response_mask"].device)
    for index, (prompt_id, data_id, rollout_id) in enumerate(zip(prompt_ids, data_ids, rollout_ids)):
        values = torch.tensor(group_values[(str(prompt_id), str(data_id))], dtype=torch.float32, device=row_advantages.device)
        reward = torch.tensor(
            rollout_records[(str(prompt_id), str(data_id), str(rollout_id))],
            device=row_advantages.device,
        )
        mean = values.mean()
        std = values.std(unbiased=True) if len(values) > 1 else torch.tensor(1.0, device=values.device)
        row_advantages[index] = (reward - mean) / (std + epsilon) if normalize_by_std else reward - mean

    advantages = row_advantages.unsqueeze(-1) * data.batch["response_mask"]
    data.batch["advantages"] = advantages
    data.batch["returns"] = advantages
    return data
