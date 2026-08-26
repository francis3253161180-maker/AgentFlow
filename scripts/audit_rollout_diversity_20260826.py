#!/usr/bin/env python3
"""Audit training-only rollout group diversity for the 2026-08-26 run.

This script deliberately reads only ``rollout_root/train``.  Validation files
are never included in group statistics.  It does not call an LLM and does not
modify rollout data.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
METRIC_RE = re.compile(
    r"(?:^| - )(?P<key>[A-Za-z0-9_./-]+):(?P<value>[^ -]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def normalize_answer(value: Any) -> str:
    """Deterministic normalization used only for exact duplicate detection."""

    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^[`*_#\s]+|[`*_#\s]+$", "", text)
    return text


def json_number(value: str) -> float | int | None:
    value = value.strip()
    value = value.removeprefix("np.float64(").removesuffix(")")
    try:
        parsed = float(value)
    except ValueError:
        return None
    if parsed.is_integer():
        return int(parsed)
    return parsed


def parse_metric_line(line: str) -> dict[str, float | int]:
    line = ANSI_RE.sub("", line)
    metrics: dict[str, float | int] = {}
    for match in METRIC_RE.finditer(line):
        value = match.group("value")
        value = value.strip("()")
        parsed = json_number(value)
        if parsed is not None:
            metrics[match.group("key")] = parsed
    return metrics


def parse_training_metrics(path: Path) -> list[dict[str, float | int]]:
    metrics = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        clean = ANSI_RE.sub("", line)
        if " step:" not in clean or "training/global_step:" not in clean:
            continue
        parsed = parse_metric_line(clean)
        if "training/global_step" in parsed:
            metrics.append(parsed)
    return metrics


def extract_tool_signature(total_result: Any) -> str | None:
    """Return a small structural path signature; never include raw text."""

    if not isinstance(total_result, dict):
        return None
    result_keys = sorted(
        key for key in total_result if key.startswith("tool_result_")
    )
    commander_keys = sorted(
        key for key in total_result if key.startswith("tool_commander_")
    )
    step_count = total_result.get("step_count")
    if not result_keys and not commander_keys and step_count is None:
        return None
    return json.dumps(
        {
            "tool_result_keys": result_keys,
            "tool_commander_keys": commander_keys,
            "step_count": step_count,
        },
        sort_keys=True,
    )


def sample_std(values: list[float]) -> float:
    # This mirrors torch.std(torch.tensor([scores])) in the installed GRPO
    # implementation: the default is unbiased=True for groups with n > 1.
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values)


def theoretical_advantages(rewards: list[float]) -> list[float]:
    if len(rewards) <= 1:
        return [0.0 for _ in rewards]
    mean = statistics.mean(rewards)
    std = sample_std(rewards)
    if std == 0.0:
        return [0.0 for _ in rewards]
    epsilon = 1e-6
    return [(reward - mean) / (std + epsilon) for reward in rewards]


def proportion(count: int, total: int) -> float:
    return count / total if total else 0.0


def summarize_groups(groups: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(groups)
    all_one = sum(group["class"] == "all-1" for group in groups)
    all_zero = sum(group["class"] == "all-0" for group in groups)
    mixed = sum(group["class"] == "mixed" for group in groups)
    nonzero = sum(group["theoretical_nonzero_advantage"] for group in groups)
    return {
        "groups": total,
        "rollouts": sum(len(group["rewards"]) for group in groups),
        "all_1_groups": all_one,
        "all_0_groups": all_zero,
        "mixed_reward_groups": mixed,
        "all_1_proportion": proportion(all_one, total),
        "all_0_proportion": proportion(all_zero, total),
        "mixed_reward_proportion": proportion(mixed, total),
        "nonzero_variance_groups": nonzero,
        "nonzero_variance_proportion": proportion(nonzero, total),
        "mean_unique_answers_per_group": statistics.mean(
            group["unique_answers"] for group in groups
        )
        if groups
        else 0.0,
        "mean_normalized_exact_duplicate_rate": statistics.mean(
            group["normalized_exact_duplicate_rate"] for group in groups
        )
        if groups
        else 0.0,
        "exact_duplicate_group_proportion": proportion(
            sum(group["unique_answers"] < len(group["rewards"]) for group in groups),
            total,
        ),
        "mean_group_reward": statistics.mean(
            statistics.mean(group["rewards"]) for group in groups
        )
        if groups
        else 0.0,
        "theoretical_advantage_abs_mean": statistics.mean(
            [abs(value) for group in groups for value in group["theoretical_advantages"]]
        )
        if groups
        else 0.0,
        "theoretical_advantage_min": min(
            (value for group in groups for value in group["theoretical_advantages"]),
            default=0.0,
        ),
        "theoretical_advantage_max": max(
            (value for group in groups for value in group["theoretical_advantages"]),
            default=0.0,
        ),
        "groups_with_reliable_path_signature": sum(
            group["path_signature_available"] for group in groups
        ),
        "mean_unique_path_signatures_per_group": statistics.mean(
            group["unique_path_signatures"] for group in groups
            if group["path_signature_available"]
        )
        if any(group["path_signature_available"] for group in groups)
        else None,
    }


def load_source_map(path: Path) -> dict[int, dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("Run this audit in the AgentFlow conda environment") from exc
    frame = pd.read_parquet(path)
    source_map: dict[int, dict[str, Any]] = {}
    for row in frame.to_dict(orient="records"):
        extra = row.get("extra_info") or {}
        if isinstance(extra, str):
            try:
                extra = json.loads(extra)
            except json.JSONDecodeError:
                extra = {}
        idx = extra.get("idx")
        if idx is None:
            continue
        source_map[int(idx)] = {
            "source": str(row.get("source", "unknown")),
            "dataset_id": row.get("id"),
        }
    return source_map


def main() -> None:
    args = parse_args()
    source_map = load_source_map(args.data)
    train_files = sorted(
        path
        for path in (args.rollout_root / "train").glob("step_*/idx_*/rollout_*.json")
    )
    if not train_files:
        raise SystemExit("No training rollout JSON files found")

    rows = []
    for path in train_files:
        row = load_json(path)
        idx = int(row["idx"])
        if idx not in source_map:
            raise SystemExit(f"Rollout idx {idx} is absent from source parquet")
        reward = float(row["reward"])
        if reward not in (0.0, 1.0):
            raise SystemExit(f"Non-binary reward in {path}: {reward}")
        rows.append(
            {
                "path": str(path.relative_to(args.rollout_root)),
                "idx": idx,
                "id": int(row["id"]),
                "source": source_map[idx]["source"],
                "reward": reward,
                "answer_norm": normalize_answer(row.get("answer_extracted", "")),
                "path_signature": extract_tool_signature(row.get("total_result")),
            }
        )

    by_idx: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_idx[row["idx"]].append(row)

    groups = []
    for idx, members in sorted(by_idx.items()):
        members.sort(key=lambda item: item["path"])
        rewards = [member["reward"] for member in members]
        unique_answers = len({member["answer_norm"] for member in members})
        duplicate_rate = 1.0 - unique_answers / len(members) if members else 0.0
        path_values = [member["path_signature"] for member in members]
        path_available = all(value is not None for value in path_values)
        unique_paths = len(set(path_values)) if path_available else 0
        adv = theoretical_advantages(rewards)
        group_class = (
            "all-1" if all(reward == 1.0 for reward in rewards) else
            "all-0" if all(reward == 0.0 for reward in rewards) else
            "mixed"
        )
        groups.append(
            {
                "idx": idx,
                "dataset_id": source_map[idx]["dataset_id"],
                "source": source_map[idx]["source"],
                "n": len(members),
                "rewards": rewards,
                "class": group_class,
                "reward_mean": statistics.mean(rewards),
                "reward_variance_population": statistics.pvariance(rewards)
                if len(rewards) > 1
                else 0.0,
                "reward_std_torch_unbiased": sample_std(rewards),
                "theoretical_advantages": adv,
                "theoretical_nonzero_advantage": any(abs(value) > 1e-9 for value in adv),
                "unique_answers": unique_answers,
                "normalized_exact_duplicate_rate": duplicate_rate,
                "path_signature_available": path_available,
                "unique_path_signatures": unique_paths,
            }
        )

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        by_source[group["source"]].append(group)

    logged = parse_training_metrics(args.train_log)
    logged_adv = [
        {
            "global_step": int(item["training/global_step"]),
            "mean": item.get("critic/advantages/mean"),
            "min": item.get("critic/advantages/min"),
            "max": item.get("critic/advantages/max"),
            "reward_mean": item.get("critic/rewards/mean"),
            "pg_loss": item.get("actor/pg_loss"),
            "grad_norm": item.get("actor/grad_norm"),
            "n_sample_to_train": item.get("agent_mode/n_sample_to_train"),
            "n_dropped_sample_because_of_mini_batch": item.get(
                "agent_mode/n_dropped_sample_because_of_mini_batch"
            ),
        }
        for item in logged
    ]
    logged_extrema_nonzero = any(
        abs(float(item.get("min") or 0.0)) > 1e-9
        or abs(float(item.get("max") or 0.0)) > 1e-9
        for item in logged_adv
    )
    mixed_count = sum(group["class"] == "mixed" for group in groups)
    pipeline_consistent = mixed_count == 0 or logged_extrema_nonzero

    output = {
        "audit": {
            "script": "scripts/audit_rollout_diversity_20260826.py",
            "training_only_glob": "train/step_*/idx_*/rollout_*.json",
            "validation_excluded": True,
            "rollout_root": str(args.rollout_root),
            "source_data": str(args.data),
            "rollout_files": len(rows),
            "groups": len(groups),
            "all_groups_have_n2": all(group["n"] == 2 for group in groups),
            "source_counts": dict(Counter(row["source"] for row in rows)),
        },
        "overall": summarize_groups(groups),
        "by_source": {
            source: summarize_groups(source_groups)
            for source, source_groups in sorted(by_source.items())
        },
        "groups": groups,
        "logged_training_metrics": logged_adv,
        "advantage_pipeline_check": {
            "mixed_reward_groups": mixed_count,
            "theoretical_nonzero_advantage_groups": sum(
                group["theoretical_nonzero_advantage"] for group in groups
            ),
            "logged_advantage_extrema_nonzero": logged_extrema_nonzero,
            "consistent_with_logged_advantage_extrema": pipeline_consistent,
            "stop_sweep_if_false": not pipeline_consistent,
            "note": (
                "GRPO implementation uses torch.std default unbiased=True and adds "
                "epsilon=1e-6. A group-level advantage mean is expected to be zero; "
                "the decisive log check is advantage min/max."
            ),
        },
        "runtime_sampling_evidence": {
            "agentflow_train_temperature": 0.7,
            "agentflow_test_temperature": 0.0,
            "planner_top_p": 0.99,
            "planner_top_k": "not configured in AgentFlow planner request",
            "planner_do_sample": "OpenAI-compatible sampling request; not separately emitted",
            "trainer_actor_rollout_n": 2,
            "trainer_resolved_rollout_temperature": 1.0,
            "trainer_resolved_rollout_top_p": 1.0,
            "trainer_resolved_rollout_top_k": -1,
            "trainer_resolved_rollout_do_sample": True,
            "interpretation": (
                "The task rollout planner receives TRAIN_TEMPERATURE=0.7; the trainer "
                "also resolves actor_rollout_ref.rollout.temperature=1.0. The sweep "
                "must change the former, which is the temperature visible in the "
                "AgentFlow planner request, while keeping the latter unchanged."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "rollouts": len(rows),
        "groups": len(groups),
        "overall": output["overall"],
        "by_source": output["by_source"],
        "advantage_pipeline_check": output["advantage_pipeline_check"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
