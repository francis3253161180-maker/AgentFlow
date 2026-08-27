#!/usr/bin/env python3
"""Aggregate benchmark probe rollout JSON and runtime telemetry offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from audit_rollout_diversity_20260826 import ANSI_RE, extract_tool_signature, normalize_answer, theoretical_advantages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prior-results", type=Path, required=True, help="results JSON for the three completed probes")
    parser.add_argument("--dataset", nargs=4, action="append", required=True, metavar=("NAME", "META", "TRAIN_LOG", "ROLLOUT_LOG"))
    parser.add_argument("--not-run", action="append", default=[], metavar="DATASET=REASON", help="benchmark blocked before rollout")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def class_for(rewards: list[float]) -> str:
    return f"{sum(value == 1.0 for value in rewards)}/{len(rewards)}"


def parse_events(paths: list[Path]) -> dict[str, Any]:
    events: list[dict[str, str]] = []
    for path in paths:
        text = ANSI_RE.sub("", path.read_text(encoding="utf-8", errors="replace"))
        for line in text.splitlines():
            if "HYBRID_REWARD_EVENT " in line:
                fields = dict(re.findall(r"(route|score|cache_hit|reason|error|latency_ms)=([^\s]+)", line))
                if fields:
                    events.append(fields)
    routes = Counter(event.get("route", "unknown") for event in events)
    latencies = [float(event["latency_ms"]) for event in events if event.get("latency_ms")]
    errors = [event for event in events if event.get("error", "none") != "none"]
    return {
        "event_count": len(events),
        "route_counts": dict(routes),
        "deterministic_count": routes.get("deterministic", 0),
        "judge_fallback_count": routes.get("judge", 0) + routes.get("judge_cache", 0),
        "judge_api_call_count": routes.get("judge", 0),
        "cache_hit_count": sum(event.get("cache_hit") == "1" for event in events),
        "api_or_parse_error_count": len(errors),
        "error_values": dict(Counter(event.get("error", "unknown") for event in errors)),
        "latency_ms_mean": statistics.mean(latencies) if latencies else None,
        "latency_ms_median": statistics.median(latencies) if latencies else None,
    }


def runtime_info(paths: list[Path]) -> dict[str, Any]:
    text = "\n".join(ANSI_RE.sub("", path.read_text(encoding="utf-8", errors="replace")) for path in paths)
    progress = re.findall(r"Progress: .*?\((\d+)/(\d+)\), Valid: (\d+), Retries: (\d+)", text)
    summaries = re.findall(r"Validation summary: (\d+)/(\d+) total rollouts .*?, (\d+) valid rollouts", text)
    elapsed = re.findall(r"Finished after ([0-9.]+) minutes", text)
    cleanup = [line.strip() for line in text.splitlines() if "VLLM_CLEANUP_DRIVER" in line]
    error_re = re.compile(r"CUDA out of memory|OutOfMemoryError|illegal memory access|device-side assert|blocks are not freed|Failed to reset prefix cache|RayTaskError|deadlock|No valid rollout|worker died|drained.*False|HTTP/\S+\s+5\d\d|status[_ ]?code[=: ]+5\d\d", re.I)
    errors = [line.strip()[-500:] for line in text.splitlines() if error_re.search(line)]
    updates = [line.strip()[-500:] for line in text.splitlines() if re.search(r"Training data keys|optimizer\.step|backward\(|update_actor|actor/pg_loss", line, re.I)]
    validation_summary = {"completed": int(summaries[-1][0]), "attempted": int(summaries[-1][1]), "valid": int(summaries[-1][2])} if summaries else None
    return {
        "progress_last": {"completed": int(progress[-1][0]), "queued": int(progress[-1][1]), "valid": int(progress[-1][2]), "retries": int(progress[-1][3])} if progress else None,
        "validation_summary": validation_summary,
        "retry_count": max(0, validation_summary["attempted"] - validation_summary["completed"]) if validation_summary else None,
        "elapsed_minutes": float(elapsed[-1]) if elapsed else None,
        "cleanup_markers": cleanup,
        "cleanup_drained_true": sum("drained': True" in line or "drained=True" in line for line in cleanup),
        "cleanup_drained_false": sum("drained': False" in line or "drained=False" in line for line in cleanup),
        "errors": errors[:20],
        "unexpected_update_markers": updates[:20],
    }


def gpu_info(path: Path) -> dict[str, Any]:
    memory: list[float] = []
    utilization: list[float] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            fields = [part.strip() for part in line.split(",")]
            if len(fields) >= 5:
                try:
                    memory.append(float(fields[2]))
                    utilization.append(float(fields[4]))
                except ValueError:
                    pass
    return {"samples": len(memory), "observed_peak_memory_mib": max(memory, default=None), "observed_peak_utilization_percent": max(utilization, default=None)}


def aggregate_dataset(name: str, meta_path: Path, train_log: Path, rollout_log: Path, manifest_dataset: dict[str, Any]) -> dict[str, Any]:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    selected = {int(row["sample_order"]): row for row in manifest_dataset["selected_rows"]}
    root = Path(meta["train_rollout_dir"])
    paths = sorted(root.glob("step_*/idx_*/rollout_*.json"))
    by_id: dict[int, list[dict[str, Any]]] = {}
    for path in paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        data_id = int(row["id"])
        if data_id not in selected:
            raise SystemExit(f"{name}: rollout id {data_id} absent from manifest")
        reward = float(row["reward"])
        if reward not in (0.0, 1.0):
            raise SystemExit(f"{name}: non-binary reward {reward}")
        by_id.setdefault(data_id, []).append({
            "reward": reward,
            "answer_norm": normalize_answer(row.get("answer_extracted", "")),
            "path_signature": extract_tool_signature(row.get("total_result")),
        })
    groups: list[dict[str, Any]] = []
    for data_id in sorted(selected):
        members = by_id.get(data_id, [])
        # Retries can leave more than n files for an id. Use the first n
        # deterministic path-sorted attempts for the protocol vector and
        # preserve extra attempts as validity telemetry.
        protocol_members = members[:4]
        rewards = [member["reward"] for member in protocol_members]
        signatures = [member["path_signature"] for member in protocol_members]
        unique_answers = len({member["answer_norm"] for member in members})
        path_available = bool(signatures) and all(signature is not None for signature in signatures)
        advantages = theoretical_advantages(rewards)
        groups.append({
            "id": data_id,
            "source_row_index": selected[data_id]["source_row_index"],
            "n": len(protocol_members),
            "attempt_count": len(members),
            "extra_retry_attempts": max(0, len(members) - 4),
            "rewards": rewards,
            "class": class_for(rewards),
            "reward_mean": statistics.mean(rewards) if rewards else 0.0,
            "reward_variance_population": statistics.pvariance(rewards) if len(rewards) > 1 else 0.0,
            "theoretical_advantages": advantages,
            "nonzero_theoretical_advantage": any(abs(value) > 1e-9 for value in advantages),
            "unique_answers": unique_answers,
            "duplicate_rate": 1.0 - unique_answers / len(members) if members else 0.0,
            "path_signature_available": path_available,
            "unique_path_signatures": len(set(signatures)) if path_available else 0,
        })
    complete_groups = [group for group in groups if group["n"] == 4]
    values = [value for group in complete_groups for value in group["rewards"]]
    bins = {f"{count}/4": sum(group["class"] == f"{count}/4" for group in complete_groups) for count in range(5)}
    mixed = sum(group["class"] in {"1/4", "2/4", "3/4"} for group in complete_groups)
    denominator = len(complete_groups)
    runtime = runtime_info([train_log, rollout_log])
    reported_valid = (runtime.get("validation_summary") or {}).get("valid")
    attempted_rollouts = len(paths)
    protocol_rollouts = len(selected) * 4
    return {
        "dataset": name,
        "source": {
            "path": manifest_dataset["source_path"],
            "ref": manifest_dataset.get("source_ref"),
            "split": manifest_dataset.get("split"),
            "sha256": manifest_dataset["source_sha256"],
            "sample_count": manifest_dataset["selected_sample_count"],
        },
        "stage": "remaining_stage1_screen",
        "status": "complete" if reported_valid == protocol_rollouts and len(complete_groups) == len(selected) else "partial",
        "prompts": len(groups),
        "planned_prompts": len(selected),
        "protocol_rollouts": protocol_rollouts,
        "attempted_rollouts": attempted_rollouts,
        "valid_rollouts": reported_valid if reported_valid is not None else len(values),
        "reward_rows_used": len(values),
        "completeness": {
            "expected_groups": len(selected),
            "complete_n4_groups": len(complete_groups),
            "partial_or_missing_groups": len(groups) - len(complete_groups),
            "reported_valid_rollouts": reported_valid,
            "protocol_complete": reported_valid == protocol_rollouts and len(complete_groups) == len(selected),
        },
        "groups": groups,
        "summary": {
            "group_bin_counts": bins,
            "group_bin_proportions": {key: value / denominator for key, value in bins.items()} if denominator else {key: None for key in bins},
            "mixed_group_count": mixed,
            "mixed_group_ratio": mixed / denominator if denominator else None,
            "nonzero_variance_group_count": sum(group["nonzero_theoretical_advantage"] for group in complete_groups),
            "nonzero_variance_group_ratio": sum(group["nonzero_theoretical_advantage"] for group in complete_groups) / denominator if denominator else None,
            "reward_mean": statistics.mean(values) if values else 0.0,
            "positive_reward_count": sum(value == 1.0 for value in values),
            "negative_reward_count": sum(value == 0.0 for value in values),
            "mean_unique_answers_per_group": statistics.mean(group["unique_answers"] for group in complete_groups) if complete_groups else None,
            "exact_duplicate_rate": statistics.mean(group["duplicate_rate"] for group in complete_groups) if complete_groups else None,
            "groups_with_exact_duplicates": sum(group["unique_answers"] < 4 for group in complete_groups),
            "mean_unique_path_signatures_per_group": statistics.mean([group["unique_path_signatures"] for group in complete_groups if group["path_signature_available"]]) if any(group["path_signature_available"] for group in complete_groups) else None,
            "path_signature_available_groups": sum(group["path_signature_available"] for group in complete_groups),
            "metrics_group_denominator": denominator,
        },
        "scorer_routing": parse_events([train_log, rollout_log]),
        "runtime": runtime,
        "gpu": gpu_info(Path(meta["gpu_log"])),
        "raw_evidence": {"train_log": meta["train_log"], "rollout_log": meta["rollout_log"], "train_rollout_dir": meta["train_rollout_dir"]},
    }


def not_run_dataset(name: str, reason: str, manifest_dataset: dict[str, Any]) -> dict[str, Any]:
    bins = {f"{count}/4": 0 for count in range(5)}
    return {
        "dataset": name,
        "source": {
            "path": manifest_dataset["source_path"],
            "ref": manifest_dataset.get("source_ref"),
            "split": manifest_dataset.get("split"),
            "sha256": manifest_dataset["source_sha256"],
            "sample_count": manifest_dataset["selected_sample_count"],
        },
        "stage": "remaining_stage1_screen_not_run",
        "status": "not_run",
        "not_run_reason": reason,
        "prompts": 0,
        "planned_prompts": manifest_dataset["selected_sample_count"],
        "protocol_rollouts": 0,
        "attempted_rollouts": 0,
        "valid_rollouts": 0,
        "reward_rows_used": 0,
        "completeness": {
            "expected_groups": manifest_dataset["selected_sample_count"],
            "complete_n4_groups": 0,
            "partial_or_missing_groups": manifest_dataset["selected_sample_count"],
            "reported_valid_rollouts": 0,
            "protocol_complete": False,
        },
        "groups": [],
        "summary": {
            "group_bin_counts": bins,
            "group_bin_proportions": {key: None for key in bins},
            "mixed_group_count": 0,
            "mixed_group_ratio": None,
            "nonzero_variance_group_count": 0,
            "nonzero_variance_group_ratio": None,
            "reward_mean": None,
            "positive_reward_count": 0,
            "negative_reward_count": 0,
            "mean_unique_answers_per_group": None,
            "exact_duplicate_rate": None,
            "groups_with_exact_duplicates": 0,
            "mean_unique_path_signatures_per_group": None,
            "path_signature_available_groups": 0,
            "metrics_group_denominator": 0,
        },
        "scorer_routing": {
            "event_count": 0,
            "route_counts": {},
            "deterministic_count": 0,
            "judge_fallback_count": 0,
            "judge_api_call_count": 0,
            "cache_hit_count": 0,
            "api_or_parse_error_count": 0,
            "error_values": {},
            "latency_ms_mean": None,
            "latency_ms_median": None,
        },
        "runtime": {
            "progress_last": None,
            "validation_summary": None,
            "elapsed_minutes": None,
            "cleanup_markers": [],
            "cleanup_drained_true": 0,
            "cleanup_drained_false": 0,
            "errors": [],
            "unexpected_update_markers": [],
        },
        "gpu": {"samples": 0, "observed_peak_memory_mib": None, "observed_peak_utilization_percent": None},
        "raw_evidence": {},
    }


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    output: dict[str, Any] = {
        "schema_version": 1,
        "manifest": str(args.manifest),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "protocol": {
            "model": "/root/autodl-tmp/models/Qwen2.5-3B-Instruct",
            "temperature": 0.7,
            "rollout_n": 4,
            "rollout_only": True,
            "optimizer_steps": 0,
            "checkpoint_disabled": True,
            "scorer": "current hybrid scorer from commit 9447d83; no per-dataset rules",
            "benchmark_examples_are_probe_only": True,
        },
        "datasets": {},
    }
    for name, meta, train_log, rollout_log in args.dataset:
        if name in output["datasets"]:
            raise SystemExit(f"duplicate dataset: {name}")
        if name not in manifest["datasets"]:
            raise SystemExit(f"dataset absent from sample manifest: {name}")
        output["datasets"][name] = aggregate_dataset(name, Path(meta), Path(train_log), Path(rollout_log), manifest["datasets"][name])

    for value in args.not_run:
        if "=" not in value:
            raise SystemExit(f"--not-run must use DATASET=REASON: {value}")
        name, reason = value.split("=", 1)
        if name in output["datasets"] or name not in manifest["datasets"]:
            raise SystemExit(f"invalid or duplicate not-run dataset: {name}")
        output["datasets"][name] = not_run_dataset(name, reason, manifest["datasets"][name])

    prior = json.loads(args.prior_results.read_text(encoding="utf-8"))
    for name, item in prior.get("datasets", {}).items():
        if name in output["datasets"]:
            raise SystemExit(f"dataset appears in both new and prior results: {name}")
        item = dict(item)
        item["stage"] = "prior_probe"
        item["status"] = "complete"
        item["planned_prompts"] = item.get("prompts")
        item["source"] = {
            "kind": "completed prior probe",
            "results": str(args.prior_results),
            "sample_count": item.get("prompts"),
        }
        output["datasets"][name] = item

    if len(output["datasets"]) != 10:
        raise SystemExit(f"expected unified 10-benchmark comparison, got {len(output['datasets'])}")
    output["overall"] = {
        "benchmarks": len(output["datasets"]),
        "prompts": sum(item.get("planned_prompts", item["prompts"]) for item in output["datasets"].values()),
        "observed_prompt_groups": sum(item["prompts"] for item in output["datasets"].values()),
        "valid_rollouts": sum(item["valid_rollouts"] for item in output["datasets"].values()),
        "remaining_stage1_prompts": sum(item["prompts"] for item in output["datasets"].values() if item["stage"] == "remaining_stage1_screen"),
        "remaining_stage1_planned_prompts": sum(item.get("planned_prompts", item["prompts"]) for item in output["datasets"].values() if item["stage"].startswith("remaining_stage1_screen")),
        "prior_probe_prompts": sum(item["prompts"] for item in output["datasets"].values() if item["stage"] == "prior_probe"),
        "not_run_benchmarks": sum(item["stage"] == "remaining_stage1_screen_not_run" for item in output["datasets"].values()),
    }
    output["unified_comparison"] = []
    for name in sorted(output["datasets"]):
        item = output["datasets"][name]
        summary = item["summary"]
        output["unified_comparison"].append({
            "dataset": name,
            "stage": item["stage"],
            "status": item.get("status", "complete"),
            "sample_size": item.get("planned_prompts", item["prompts"]),
            "observed_prompt_groups": item["prompts"],
            "valid_rollouts": item["valid_rollouts"],
            "reward_mean": summary["reward_mean"],
            "group_bin_counts": summary["group_bin_counts"],
            "mixed_group_ratio": summary["mixed_group_ratio"],
            "nonzero_variance_group_ratio": summary["nonzero_variance_group_ratio"],
            "mean_unique_answers_per_group": summary["mean_unique_answers_per_group"],
            "exact_duplicate_rate": summary["exact_duplicate_rate"],
            "mean_unique_path_signatures_per_group": summary["mean_unique_path_signatures_per_group"],
            "scorer_routing": item["scorer_routing"],
            "runtime_minutes": item["runtime"]["elapsed_minutes"],
            "gpu_peak_memory_mib": item["gpu"]["observed_peak_memory_mib"],
            "cleanup_drained_false": item["runtime"]["cleanup_drained_false"],
            "cleanup_drained_true": item["runtime"]["cleanup_drained_true"],
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({name: item["summary"] for name, item in output["datasets"].items()}, sort_keys=True))


if __name__ == "__main__":
    main()
