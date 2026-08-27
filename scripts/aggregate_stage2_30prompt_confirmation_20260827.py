#!/usr/bin/env python3
"""Merge the historical and incremental stage-2 benchmark probes offline.

This script never calls a scorer or an external API.  Historical groups are
matched to the incremental manifest by source-row index and every dataset is
required to have exactly 30 complete n=4 groups before results are written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from aggregate_benchmark_difficulty_probe_20260827 import gpu_info, parse_events, runtime_info
from audit_rollout_diversity_20260826 import extract_tool_signature, normalize_answer, theoretical_advantages


DATASETS = ("2wiki", "gameof24", "aime24")


def parse_kv(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, path = value.split("=", 1)
    return name, Path(path)


def class_for(rewards: list[float]) -> str:
    return f"{sum(value == 1.0 for value in rewards)}/{len(rewards)}"


def wilson95(successes: int, total: int) -> list[float] | None:
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return [center - half, center + half]


def source_row_map(manifest_dataset: dict[str, Any], phase: str) -> dict[int, dict[str, Any]]:
    rows = manifest_dataset["historical_rows"] if phase == "historical" else manifest_dataset["incremental_rows"]
    return {int(row["sample_order"] if phase == "incremental" else row["source_row_index"]): row for row in rows}


def group_summary(groups: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [group for group in groups if group["n"] == 4]
    values = [value for group in complete for value in group["rewards"]]
    bins = {f"{count}/4": sum(group["class"] == f"{count}/4" for group in complete) for count in range(5)}
    mixed = sum(group["class"] in {"1/4", "2/4", "3/4"} for group in complete)
    denominator = len(complete)
    path_groups = [group for group in complete if group.get("path_signature_available")]
    return {
        "group_bin_counts": bins,
        "group_bin_proportions": {key: value / denominator for key, value in bins.items()} if denominator else {},
        "mixed_group_count": mixed,
        "mixed_group_ratio": mixed / denominator if denominator else None,
        "mixed_group_wilson_95_ci": wilson95(mixed, denominator),
        "nonzero_variance_group_count": sum(group["nonzero_theoretical_advantage"] for group in complete),
        "nonzero_variance_group_ratio": sum(group["nonzero_theoretical_advantage"] for group in complete) / denominator if denominator else None,
        "reward_mean": statistics.mean(values) if values else None,
        "positive_reward_count": sum(value == 1.0 for value in values),
        "negative_reward_count": sum(value == 0.0 for value in values),
        "mean_unique_answers_per_group": statistics.mean(group["unique_answers"] for group in complete) if complete else None,
        "exact_duplicate_rate": statistics.mean(group["duplicate_rate"] for group in complete) if complete else None,
        "groups_with_exact_duplicates": sum(group["unique_answers"] < 4 for group in complete),
        "mean_unique_path_signatures_per_group": statistics.mean(group["unique_path_signatures"] for group in path_groups) if path_groups else None,
        "path_signature_available_groups": len(path_groups),
        "metrics_group_denominator": denominator,
    }


def historical_groups(result: dict[str, Any], manifest_dataset: dict[str, Any]) -> list[dict[str, Any]]:
    by_source = {int(group["source_row_index"]): group for group in result["groups"]}
    rows = source_row_map(manifest_dataset, "historical")
    groups: list[dict[str, Any]] = []
    for source_index in sorted(rows):
        if source_index not in by_source:
            raise SystemExit(f"historical result missing source row {source_index}")
        old = by_source[source_index]
        rewards = [float(value) for value in old["rewards"]]
        if len(rewards) != 4 or any(value not in (0.0, 1.0) for value in rewards):
            raise SystemExit(f"historical group {source_index} is not complete binary n=4")
        group = dict(old)
        group.update({"phase": "historical", "source_row_index": source_index, "n": 4, "rewards": rewards})
        groups.append(group)
    return groups


def incremental_groups(dataset: str, meta_path: Path, train_log: Path, rollout_log: Path, manifest_dataset: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    rows = source_row_map(manifest_dataset, "incremental")
    root = Path(meta["train_rollout_dir"])
    paths = sorted(root.glob("step_*/idx_*/rollout_*.json"))
    by_id: dict[int, list[dict[str, Any]]] = {}
    for path in paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        data_id = int(row["id"])
        if data_id not in rows:
            raise SystemExit(f"{dataset}: rollout id {data_id} absent from incremental manifest")
        reward = float(row["reward"])
        if reward not in (0.0, 1.0):
            raise SystemExit(f"{dataset}: non-binary reward {reward}")
        by_id.setdefault(data_id, []).append({
            "reward": reward,
            "answer_norm": normalize_answer(row.get("answer_extracted", "")),
            "path_signature": extract_tool_signature(row.get("total_result")),
        })
    groups: list[dict[str, Any]] = []
    for data_id in sorted(rows):
        members = by_id.get(data_id, [])
        protocol = members[:4]
        rewards = [member["reward"] for member in protocol]
        signatures = [member["path_signature"] for member in protocol]
        if len(protocol) != 4:
            n = len(protocol)
        else:
            n = 4
        unique_answers = len({member["answer_norm"] for member in members})
        path_available = bool(signatures) and all(signature is not None for signature in signatures)
        groups.append({
            "id": data_id,
            "phase": "incremental",
            "source_row_index": int(rows[data_id]["source_row_index"]),
            "n": n,
            "attempt_count": len(members),
            "extra_retry_attempts": max(0, len(members) - 4),
            "rewards": rewards,
            "class": class_for(rewards),
            "reward_mean": statistics.mean(rewards) if rewards else 0.0,
            "reward_variance_population": statistics.pvariance(rewards) if len(rewards) > 1 else 0.0,
            "theoretical_advantages": theoretical_advantages(rewards),
            "nonzero_theoretical_advantage": any(abs(value) > 1e-9 for value in theoretical_advantages(rewards)),
            "unique_answers": unique_answers,
            "duplicate_rate": 1.0 - unique_answers / len(members) if members else 0.0,
            "path_signature_available": path_available,
            "unique_path_signatures": len(set(signatures)) if path_available else 0,
        })
    runtime = runtime_info([train_log, rollout_log])
    reported_valid = (runtime.get("validation_summary") or {}).get("valid")
    telemetry = {
        "scorer_routing": parse_events([train_log, rollout_log]),
        "runtime": runtime,
        "gpu": gpu_info(Path(meta["gpu_log"])),
        "raw_evidence": {"meta": str(meta_path), "train_log": meta.get("train_log"), "rollout_log": meta.get("rollout_log"), "train_rollout_dir": meta.get("train_rollout_dir")},
        "reported_valid_rollouts": reported_valid,
        "attempted_rollout_files": len(paths),
    }
    return groups, telemetry


def combine_telemetry(old_items: list[dict[str, Any]], new: dict[str, Any]) -> dict[str, Any]:
    routing = [item.get("scorer_routing", {}) for item in old_items] + [new["scorer_routing"]]
    keys = ("event_count", "deterministic_count", "judge_fallback_count", "judge_api_call_count", "cache_hit_count", "api_or_parse_error_count")
    combined = {key: sum(int(item.get(key, 0) or 0) for item in routing) for key in keys}
    route_counts = Counter()
    errors = Counter()
    latencies: list[float] = []
    for item in routing:
        route_counts.update(item.get("route_counts", {}))
        errors.update(item.get("error_values", {}))
        if item.get("latency_ms_mean") is not None and item.get("event_count", 0):
            latencies.extend([float(item["latency_ms_mean"])] * int(item["event_count"]))
    combined.update({"route_counts": dict(route_counts), "error_values": dict(errors), "latency_ms_mean": statistics.mean(latencies) if latencies else None, "latency_ms_median": statistics.median(latencies) if latencies else None})
    old_runtime = [item.get("runtime", {}) for item in old_items]
    runtimes = old_runtime + [new["runtime"]]
    elapsed = [item.get("elapsed_minutes") for item in runtimes if item.get("elapsed_minutes") is not None]
    combined["runtime_minutes_sum"] = sum(elapsed) if elapsed else None
    combined["historical_runtime_minutes"] = sum(item.get("elapsed_minutes", 0.0) or 0.0 for item in old_runtime)
    combined["incremental_runtime_minutes"] = new["runtime"].get("elapsed_minutes")
    combined["retry_count"] = sum(int(item.get("retry_count", 0) or 0) for item in runtimes)
    combined["cleanup_drained_true"] = sum(int(item.get("cleanup_drained_true", 0) or 0) for item in runtimes)
    combined["cleanup_drained_false"] = sum(int(item.get("cleanup_drained_false", 0) or 0) for item in runtimes)
    combined["cleanup_markers"] = [marker for item in runtimes for marker in item.get("cleanup_markers", [])]
    combined["errors"] = [line for item in runtimes for line in item.get("errors", [])]
    combined["unexpected_update_markers"] = [line for item in runtimes for line in item.get("unexpected_update_markers", [])]
    return combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--historical-result", action="append", type=parse_kv, required=True)
    parser.add_argument("--incremental", nargs=4, action="append", required=True, metavar=("NAME", "META", "TRAIN_LOG", "ROLLOUT_LOG"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    historical_results = {}
    for name, path in args.historical_result:
        document = json.loads(path.read_text(encoding="utf-8"))
        historical_results[name] = document["datasets"][name] if "datasets" in document else document
    incremental_args = {parts[0]: tuple(Path(value) for value in parts[1:]) for parts in args.incremental}
    if set(historical_results) != set(DATASETS) or set(incremental_args) != set(DATASETS):
        raise SystemExit(f"expected exactly {DATASETS} for both historical and incremental inputs")
    output: dict[str, Any] = {
        "schema_version": 1,
        "manifest": str(args.manifest),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "protocol": {"model": "/root/autodl-tmp/models/Qwen2.5-3B-Instruct", "temperature": 0.7, "rollout_n": 4, "rollout_only": True, "optimizer_steps": 0, "checkpoint_disabled": True, "scorer": "current hybrid scorer; unchanged during stage-2", "benchmark_examples_are_probe_only": True},
        "datasets": {},
    }
    for dataset in DATASETS:
        md = manifest["datasets"][dataset]
        old = historical_groups(historical_results[dataset], md)
        new_groups, telemetry = incremental_groups(dataset, *incremental_args[dataset], md)
        if len(old) != 20 if dataset == "2wiki" else len(old) != 10:
            raise SystemExit(f"{dataset}: unexpected historical group count {len(old)}")
        if len(new_groups) != (10 if dataset == "2wiki" else 20):
            raise SystemExit(f"{dataset}: unexpected incremental group count {len(new_groups)}")
        combined = sorted(old + new_groups, key=lambda group: group["source_row_index"])
        if len(combined) != 30 or len({group["source_row_index"] for group in combined}) != 30 or any(group["n"] != 4 for group in combined):
            raise SystemExit(f"{dataset}: combined result is not exactly 30 complete unique groups")
        historical_item = {"phase": "historical", "prompts": len(old), "valid_rollouts": len(old) * 4, "groups": old, "summary": group_summary(old), "source_rows": [group["source_row_index"] for group in old], "provenance": "existing completed probe"}
        incremental_item = {"phase": "incremental", "prompts": len(new_groups), "valid_rollouts": len(new_groups) * 4, "groups": new_groups, "summary": group_summary(new_groups), "source_rows": [group["source_row_index"] for group in new_groups], **telemetry}
        old_telemetry = [historical_results[dataset]]
        combined_item = {"phase": "combined_30_groups", "prompts": 30, "valid_rollouts": 120, "groups": combined, "summary": group_summary(combined), "source_rows": [group["source_row_index"] for group in combined], "telemetry": combine_telemetry(old_telemetry, telemetry)}
        output["datasets"][dataset] = {"source": {"path": md["source_path"], "sha256": md["source_sha256"], "historical_manifests": md["historical_manifests"], "historical_count": len(old), "incremental_count": len(new_groups)}, "historical": historical_item, "incremental": incremental_item, "combined": combined_item}
    output["comparison_30_groups"] = [{"dataset": dataset, "sample_size": 30, **output["datasets"][dataset]["combined"]["summary"]} for dataset in DATASETS]
    output["integrity"] = {"all_datasets_complete": True, "total_groups": 90, "total_valid_rollouts": 360, "no_new_rollout_api_calls_outside_normal_scorer": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({dataset: output["datasets"][dataset]["combined"]["summary"] for dataset in DATASETS}, sort_keys=True))


if __name__ == "__main__":
    main()
