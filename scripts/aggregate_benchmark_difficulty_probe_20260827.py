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
    parser.add_argument("--dataset", nargs=4, action="append", required=True, metavar=("NAME", "META", "TRAIN_LOG", "ROLLOUT_LOG"))
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
    return {
        "progress_last": {"completed": int(progress[-1][0]), "queued": int(progress[-1][1]), "valid": int(progress[-1][2]), "retries": int(progress[-1][3])} if progress else None,
        "validation_summary": {"completed": int(summaries[-1][0]), "queued": int(summaries[-1][1]), "valid": int(summaries[-1][2])} if summaries else None,
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
    for data_id, members in sorted(by_id.items()):
        rewards = [member["reward"] for member in members]
        signatures = [member["path_signature"] for member in members]
        unique_answers = len({member["answer_norm"] for member in members})
        path_available = all(signature is not None for signature in signatures)
        advantages = theoretical_advantages(rewards)
        groups.append({
            "id": data_id,
            "source_row_index": selected[data_id]["source_row_index"],
            "n": len(members),
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
    if len(groups) != len(selected) or any(group["n"] != 4 for group in groups):
        raise SystemExit(f"{name}: expected {len(selected)} complete n=4 groups; got {len(groups)}")
    values = [value for group in groups for value in group["rewards"]]
    bins = {f"{count}/4": sum(group["class"] == f"{count}/4" for group in groups) for count in range(5)}
    mixed = sum(group["class"] in {"1/4", "2/4", "3/4"} for group in groups)
    return {
        "dataset": name,
        "prompts": len(groups),
        "valid_rollouts": len(values),
        "groups": groups,
        "summary": {
            "group_bin_counts": bins,
            "group_bin_proportions": {key: value / len(groups) for key, value in bins.items()},
            "mixed_group_count": mixed,
            "mixed_group_ratio": mixed / len(groups),
            "nonzero_variance_group_count": sum(group["nonzero_theoretical_advantage"] for group in groups),
            "nonzero_variance_group_ratio": sum(group["nonzero_theoretical_advantage"] for group in groups) / len(groups),
            "reward_mean": statistics.mean(values) if values else 0.0,
            "positive_reward_count": sum(value == 1.0 for value in values),
            "negative_reward_count": sum(value == 0.0 for value in values),
            "mean_unique_answers_per_group": statistics.mean(group["unique_answers"] for group in groups),
            "exact_duplicate_rate": statistics.mean(group["duplicate_rate"] for group in groups),
            "groups_with_exact_duplicates": sum(group["unique_answers"] < 4 for group in groups),
            "mean_unique_path_signatures_per_group": statistics.mean([group["unique_path_signatures"] for group in groups if group["path_signature_available"]]) if any(group["path_signature_available"] for group in groups) else None,
            "path_signature_available_groups": sum(group["path_signature_available"] for group in groups),
        },
        "scorer_routing": parse_events([train_log, rollout_log]),
        "runtime": runtime_info([train_log, rollout_log]),
        "gpu": gpu_info(Path(meta["gpu_log"])),
        "raw_evidence": {"train_log": meta["train_log"], "rollout_log": meta["rollout_log"], "train_rollout_dir": meta["train_rollout_dir"]},
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
    output["overall"] = {"prompts": sum(item["prompts"] for item in output["datasets"].values()), "valid_rollouts": sum(item["valid_rollouts"] for item in output["datasets"].values())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({name: item["summary"] for name, item in output["datasets"].items()}, sort_keys=True))


if __name__ == "__main__":
    main()
