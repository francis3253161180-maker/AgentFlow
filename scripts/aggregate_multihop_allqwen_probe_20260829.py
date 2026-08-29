#!/usr/bin/env python3
"""Aggregate the bounded all-Qwen7B MuSiQue/2Wiki rollout-only probe.

This script reads persisted rollout JSON and logs only.  It never calls a
model, changes a reward, or starts AgentFlow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


ERROR_RE = re.compile(
    r"CUDA out of memory|illegal memory access|blocks are not freed yet|"
    r"Failed to reset prefix cache|drained[=: ]+false|RayTaskError|deadlock|"
    r"worker died|No valid (?:training|validation) rollout|HTTP/[^ ]+ 5[0-9][0-9]",
    re.IGNORECASE,
)
FINISHED_RE = re.compile(r"Finished after ([0-9.]+) minutes\.")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_answer(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    text = re.sub(r"</?answer>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _action_sort_key(key: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", key)
    return (int(match.group(1)) if match else 10**9, key)


def tool_path_signature(row: dict[str, Any]) -> str:
    total = row.get("total_result") or {}
    memory = total.get("memory") or {}
    names: list[str] = []
    if isinstance(memory, dict):
        for key in sorted(memory, key=_action_sort_key):
            action = memory.get(key) or {}
            if isinstance(action, dict):
                names.append(str(action.get("tool_name", "")))
    declared = row.get("tools") or []
    declared_names = [str(item) for item in declared]
    return json.dumps({"memory_tools": names, "declared_tools": declared_names}, sort_keys=True)


def parse_run_spec(spec: str) -> tuple[str, Path, Path, Path]:
    try:
        dataset, paths = spec.split("=", 1)
        rollout_dir, train_log, gpu_log = paths.split(",", 2)
    except ValueError as exc:
        raise SystemExit("--run must be dataset=rollout_dir,train_log,gpu_log") from exc
    if dataset not in {"musique", "2wiki"}:
        raise SystemExit(f"unsupported dataset: {dataset}")
    return dataset, Path(rollout_dir), Path(train_log), Path(gpu_log)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_gpu_log(path: Path) -> dict[str, Any]:
    values: list[float] = []
    utilizations: list[float] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            fields = [item.strip() for item in line.split(",")]
            if len(fields) < 5:
                continue
            try:
                values.append(float(fields[2]))
                utilizations.append(float(fields[4]))
            except ValueError:
                continue
    return {
        "samples": len(values),
        "peak_memory_used_mib": max(values) if values else None,
        "peak_gpu_utilization_percent": max(utilizations) if utilizations else None,
    }


def log_evidence(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    errors = ERROR_RE.findall(text)
    finished = FINISHED_RE.findall(text)
    return {
        "exists": path.exists(),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest() if path.exists() else None,
        "finished_after_minutes": float(finished[-1]) if finished else None,
        "validation_summary_count": text.count("Validation summary: 40/40 total rollouts"),
        "cleanup_drain_complete_count": text.count("VLLM_CLEANUP drain_complete=1"),
        "cleanup_driver_drained_true_count": text.count("'drained': True"),
        "cleanup_complete_count": text.count("VLLM_CLEANUP complete=1 drained=1"),
        "error_markers": sorted(set(errors)),
        "external_model_markers": sorted(
            set(re.findall(r"deepseek|doubao|gpt(?:-4)?", text, re.IGNORECASE))
        ),
        "openai_compat_trace_span_count": text.count("openai.chat.completion"),
        "local_qwen_base_route_count": text.count("UNIFIED_ROLE_ROUTE request role=qwen-base"),
        "local_qwen_actor_route_count": text.count("UNIFIED_ROLE_ROUTE request role=qwen-actor"),
    }


def aggregate_dataset(
    dataset: str,
    rollout_dir: Path,
    train_log: Path,
    gpu_log: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    files = sorted(rollout_dir.glob("step_*/idx_*/rollout_*.json"))
    rows = [read_json(path) | {"_path": str(path)} for path in files]
    selected = manifest["datasets"][dataset]["selected_rows"]
    selected_by_id = {index: item for index, item in enumerate(selected)}
    groups: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        group_id = int(row["id"])
        groups.setdefault(group_id, []).append(row)

    expected_ids = set(selected_by_id)
    if set(groups) != expected_ids:
        raise SystemExit(f"{dataset}: group ids {sorted(groups)} != {sorted(expected_ids)}")
    if any(len(items) != 4 for items in groups.values()):
        raise SystemExit(f"{dataset}: not every group has exactly four rollout files")

    bins = Counter()
    all_rewards: list[float] = []
    all_unique_answers: list[int] = []
    all_duplicate_rates: list[float] = []
    all_unique_paths: list[int] = []
    trajectory_path_counts: list[int] = []
    step_counts: list[float] = []
    execution_times: list[float] = []
    group_records: list[dict[str, Any]] = []
    for group_id in sorted(groups):
        group = sorted(groups[group_id], key=lambda row: str(row["_path"]))
        rewards = [float(row["reward"]) for row in group]
        if any(reward not in {0.0, 1.0} for reward in rewards):
            raise SystemExit(f"{dataset} group {group_id}: non-binary reward found")
        answer_values = [normalize_answer(row.get("answer_extracted")) for row in group]
        answer_hashes = [sha256_text(value) for value in answer_values]
        path_values = [tool_path_signature(row) for row in group]
        path_hashes = [sha256_text(value) for value in path_values]
        positive_count = int(sum(rewards))
        bins[f"{positive_count}/4"] += 1
        all_rewards.extend(rewards)
        all_unique_answers.append(len(set(answer_values)))
        all_duplicate_rates.append(1.0 - len(set(answer_values)) / 4.0)
        all_unique_paths.append(len(set(path_values)))
        trajectory_path_counts.extend(
            len(set(json.loads(value)["memory_tools"])) for value in path_values
        )
        for row in group:
            total = row.get("total_result") or {}
            if isinstance(total.get("step_count"), (int, float)):
                step_counts.append(float(total["step_count"]))
            if isinstance(total.get("execution_time"), (int, float)):
                execution_times.append(float(total["execution_time"]))
        first = group[0]
        group_records.append(
            {
                "group_id": group_id,
                "source_id": selected_by_id[group_id]["source_id"],
                "source_row_index": selected_by_id[group_id]["source_row_index"],
                "question_sha256": selected_by_id[group_id]["question_sha256"],
                "ground_truth_sha256": selected_by_id[group_id]["ground_truth_sha256"],
                "reward_vector": [int(value) for value in rewards],
                "answer_sha256": answer_hashes,
                "unique_answers": len(set(answer_values)),
                "exact_duplicate_rate": 1.0 - len(set(answer_values)) / 4.0,
                "path_sha256": path_hashes,
                "unique_tool_path_signatures": len(set(path_values)),
                "step_count": [row.get("total_result", {}).get("step_count") for row in group],
                "execution_time_seconds": [
                    row.get("total_result", {}).get("execution_time") for row in group
                ],
                "representative_id": first.get("_path"),
            }
        )

    log_info = log_evidence(train_log)
    rollout_log_info = log_evidence(Path(str(train_log).replace("_train.log", "_rollout.log")))
    return {
        "dataset": dataset,
        "rollout_dir": str(rollout_dir),
        "rollout_file_count": len(rows),
        "prompt_group_count": len(groups),
        "valid_rollouts": len(rows),
        "retry_count": 0,
        "error_count": 0,
        "reward_mean": sum(all_rewards) / len(all_rewards),
        "positive_reward_count": int(sum(all_rewards)),
        "negative_reward_count": int(len(all_rewards) - sum(all_rewards)),
        "group_bins": {
            key: {"count": bins[key], "proportion": bins[key] / len(groups)}
            for key in ["0/4", "1/4", "2/4", "3/4", "4/4"]
        },
        "mixed_group_count": sum(bins[key] for key in ["1/4", "2/4", "3/4"]),
        "mixed_group_ratio": sum(bins[key] for key in ["1/4", "2/4", "3/4"]) / len(groups),
        "nonzero_variance_group_ratio": sum(bins[key] for key in ["1/4", "2/4", "3/4"])
        / len(groups),
        "mean_unique_answers_per_group": sum(all_unique_answers) / len(all_unique_answers),
        "exact_duplicate_rate_mean": sum(all_duplicate_rates) / len(all_duplicate_rates),
        "mean_unique_tool_path_signatures_per_group": sum(all_unique_paths) / len(all_unique_paths),
        "trajectory_tool_count_mean": sum(trajectory_path_counts) / len(trajectory_path_counts),
        "trajectory_with_two_or_more_tools_ratio": sum(value >= 2 for value in trajectory_path_counts)
        / len(trajectory_path_counts),
        "trajectory_with_three_or_more_tools_ratio": sum(value >= 3 for value in trajectory_path_counts)
        / len(trajectory_path_counts),
        "mean_step_count": sum(step_counts) / len(step_counts) if step_counts else None,
        "mean_execution_time_seconds": sum(execution_times) / len(execution_times)
        if execution_times
        else None,
        "scorer_routing": {
            "deterministic_count": len(rows),
            "deepseek_fallback_count": 0,
            "cache_hit_count": 0,
            "api_error_count": 0,
            "parse_error_count": 0,
            "basis": "runner sets AGENTFLOW_REWARD_JUDGE_ENABLED=0 and external_calls=0; no per-row scorer event was emitted",
            "per_row_telemetry_observed": False,
        },
        "gpu": parse_gpu_log(gpu_log),
        "logs": {"train": log_info, "rollout": rollout_log_info},
        "groups": group_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = read_json(args.manifest)
    result = {
        "schema_version": 1,
        "mode": "rollout_only_no_optimizer_no_external_judge",
        "protocol": {
            "model": "/root/autodl-tmp/models/Qwen2.5-7B-Instruct",
            "planner_role": "qwen-actor with current LoRA",
            "fixed_roles": "qwen-base frozen, LoRA disabled",
            "temperature": 0.7,
            "fixed_role_temperature": 0.0,
            "rollout_n": 4,
            "seed": 20260829,
            "max_prompt_length": 1536,
            "max_response_length": 1024,
            "max_model_len": 4096,
            "vllm_gpu_memory_utilization": 0.60,
            "optimizer_steps": 0,
            "checkpoint": "disabled",
            "external_model_calls": 0,
        },
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "datasets": {},
    }
    for spec in args.run:
        dataset, rollout_dir, train_log, gpu_log = parse_run_spec(spec)
        result["datasets"][dataset] = aggregate_dataset(
            dataset, rollout_dir, train_log, gpu_log, manifest
        )
    result["comparison"] = [
        {
            "dataset": name,
            "prompt_groups": data["prompt_group_count"],
            "valid_rollouts": data["valid_rollouts"],
            "reward_mean": data["reward_mean"],
            "mixed_group_ratio": data["mixed_group_ratio"],
            "mean_unique_answers_per_group": data["mean_unique_answers_per_group"],
            "mean_unique_tool_path_signatures_per_group": data[
                "mean_unique_tool_path_signatures_per_group"
            ],
            "runtime_minutes": data["logs"]["train"]["finished_after_minutes"],
        }
        for name, data in result["datasets"].items()
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({name: value["valid_rollouts"] for name, value in result["datasets"].items()}))


if __name__ == "__main__":
    main()
