#!/usr/bin/env python3
"""Aggregate the fixed 100-prompt, n=4 rollout-only difficulty audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from audit_rollout_diversity_20260826 import (
    ANSI_RE,
    extract_tool_signature,
    load_json,
    normalize_answer,
    sample_std,
    theoretical_advantages,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--rollout-log", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--selected-data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gpu-log", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def selected_map(path: Path) -> dict[int, dict[str, Any]]:
    import pandas as pd

    frame = pd.read_parquet(path)
    result: dict[int, dict[str, Any]] = {}
    for row in frame.to_dict(orient="records"):
        info = row["extra_info"] or {}
        if isinstance(info, str):
            info = json.loads(info)
        result[int(row["id"])] = {
            "source": str(row["source"]),
            "idx": int(info["idx"]),
        }
    return result


def class_for(rewards: list[float]) -> str:
    return f"{sum(value == 1.0 for value in rewards)}/{len(rewards)}"


def group_summary(groups: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [value for group in groups for value in group["rewards"]]
    total = len(groups)
    bins = {
        f"{count}/4": sum(group["class"] == f"{count}/4" for group in groups)
        for count in range(5)
    }
    mixed = sum(group["class"] not in ("0/4", "4/4") for group in groups)
    nonzero = sum(group["nonzero_theoretical_advantage"] for group in groups)
    advantages = [value for group in groups for value in group["theoretical_advantages"]]
    return {
        "groups": total,
        "rollouts": len(rows),
        "bin_counts": bins,
        "bin_proportions": {key: value / total if total else 0.0 for key, value in bins.items()},
        "mixed_informative_groups": mixed,
        "mixed_informative_proportion": mixed / total if total else 0.0,
        "overall_rollout_reward_mean": statistics.mean(rows) if rows else 0.0,
        "positive_reward_count": sum(value == 1.0 for value in rows),
        "negative_reward_count": sum(value == 0.0 for value in rows),
        "nonzero_reward_variance_groups": nonzero,
        "nonzero_reward_variance_proportion": nonzero / total if total else 0.0,
        "mean_unique_answers_per_group": statistics.mean(group["unique_answers"] for group in groups) if groups else 0.0,
        "mean_normalized_exact_duplicate_rate": statistics.mean(group["duplicate_rate"] for group in groups) if groups else 0.0,
        "exact_duplicate_group_proportion": sum(group["unique_answers"] < 4 for group in groups) / total if total else 0.0,
        "path_signature_available_groups": sum(group["path_signature_available"] for group in groups),
        "mean_unique_path_signatures_per_group": statistics.mean(
            group["unique_path_signatures"] for group in groups if group["path_signature_available"]
        ) if any(group["path_signature_available"] for group in groups) else None,
        "theoretical_advantage_min": min(advantages, default=0.0),
        "theoretical_advantage_max": max(advantages, default=0.0),
        "theoretical_advantage_mean": statistics.mean(advantages) if advantages else 0.0,
        "theoretical_advantage_std": statistics.pstdev(advantages) if len(advantages) > 1 else 0.0,
    }


def collect_groups(args: argparse.Namespace, selected: dict[int, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paths = sorted(args.rollout_root.glob("train/step_*/idx_*/rollout_*.json"))
    rows: list[dict[str, Any]] = []
    by_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        row = load_json(path)
        data_id = int(row["id"])
        if data_id not in selected:
            raise SystemExit(f"rollout id {data_id} absent from selected parquet: {path}")
        reward = float(row["reward"])
        if reward not in (0.0, 1.0):
            raise SystemExit(f"non-binary reward in {path}: {reward}")
        item = {
            "path": str(path.relative_to(args.rollout_root)),
            "id": data_id,
            "idx": int(row["idx"]),
            "source": selected[data_id]["source"],
            "reward": reward,
            "answer_norm": normalize_answer(row.get("answer_extracted", "")),
            "path_signature": extract_tool_signature(row.get("total_result")),
        }
        rows.append(item)
        by_id[data_id].append(item)

    groups: list[dict[str, Any]] = []
    for data_id, members in sorted(by_id.items()):
        members.sort(key=lambda item: item["path"])
        rewards = [member["reward"] for member in members]
        path_values = [member["path_signature"] for member in members]
        path_available = all(value is not None for value in path_values)
        unique_answers = len({member["answer_norm"] for member in members})
        advantages = theoretical_advantages(rewards)
        groups.append({
            "id": data_id,
            "idx": selected[data_id]["idx"],
            "source": selected[data_id]["source"],
            "n": len(members),
            "rewards": rewards,
            "class": class_for(rewards),
            "reward_mean": statistics.mean(rewards) if rewards else 0.0,
            "reward_variance_population": statistics.pvariance(rewards) if len(rewards) > 1 else 0.0,
            "reward_std_torch_unbiased": sample_std(rewards),
            "theoretical_advantages": advantages,
            "nonzero_theoretical_advantage": any(abs(value) > 1e-9 for value in advantages),
            "unique_answers": unique_answers,
            "duplicate_rate": 1.0 - unique_answers / len(members) if members else 0.0,
            "path_signature_available": path_available,
            "unique_path_signatures": len(set(path_values)) if path_available else 0,
        })
    return groups, rows


def parse_runtime(log: Path) -> dict[str, Any]:
    text = ANSI_RE.sub("", log.read_text(encoding="utf-8", errors="replace"))
    progress = re.findall(r"Progress: .*?\((\d+)/(\d+)\), Valid: (\d+), Retries: (\d+)", text)
    summary = re.findall(r"Validation summary: (\d+)/(\d+) total rollouts .*?, (\d+) valid rollouts", text)
    completed = re.findall(r"Completed(?:\s+\d+\.\d+%)?\s*\((\d+)/(\d+)\)", text)
    return {
        "progress_last": {
            "completed": int(progress[-1][0]), "queued": int(progress[-1][1]),
            "valid": int(progress[-1][2]), "retries": int(progress[-1][3]),
        } if progress else None,
        "validation_summary": {
            "completed": int(summary[-1][0]), "queued": int(summary[-1][1]),
            "valid": int(summary[-1][2]),
        } if summary else None,
        "timeout_or_partial_completion": {
            "completed": int(completed[-1][0]),
            "queued": int(completed[-1][1]),
        } if completed else None,
        "rollout_only_marker": "Rollout-only group mode" in text,
        "no_optimizer_marker": "no optimizer step will run" in text,
        "unexpected_update_markers": [
            line.strip() for line in text.splitlines()
            if re.search(r"Training data keys|optimizer\.step|actor/pg_loss|backward\(", line, re.I)
        ][:20],
        "oom_or_exception_lines": [
            line.strip() for line in text.splitlines()
            if re.search(r"CUDA out of memory|OutOfMemoryError|Traceback|illegal memory access|device-side assert|No valid rollout", line, re.I)
        ][:20],
    }


def parse_scorer_routing(logs: list[Path]) -> dict[str, Any]:
    events: list[dict[str, str]] = []
    for log in logs:
        text = ANSI_RE.sub("", log.read_text(encoding="utf-8", errors="replace"))
        for line in text.splitlines():
            if "HYBRID_REWARD_EVENT " not in line:
                continue
            fields = dict(re.findall(r"(route|score|cache_hit|reason|error|latency_ms)=([^\s]+)", line))
            if fields:
                events.append(fields)
    routes = Counter(event.get("route", "unknown") for event in events)
    errors = [event for event in events if event.get("error", "none") != "none"]
    latencies = [float(event["latency_ms"]) for event in events if event.get("latency_ms") not in (None, "")]
    return {
        "event_count": len(events),
        "route_counts": dict(routes),
        "deterministic_count": routes.get("deterministic", 0),
        "judge_fallback_count": routes.get("judge", 0),
        "cache_hit_count": sum(event.get("cache_hit") == "1" for event in events),
        "api_or_parse_error_count": len(errors),
        "error_values": dict(Counter(event.get("error", "unknown") for event in errors)),
        "latency_ms_mean": statistics.mean(latencies) if latencies else None,
        "latency_ms_median": statistics.median(latencies) if latencies else None,
    }


def gpu_peak(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"samples": 0, "observed_peak_memory_mib": None, "observed_peak_utilization_percent": None}
    memory: list[float] = []
    utilization: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) < 5:
            continue
        try:
            memory.append(float(fields[2]))
            utilization.append(float(fields[4]))
        except ValueError:
            continue
    return {
        "samples": len(memory),
        "observed_peak_memory_mib": max(memory, default=None),
        "observed_peak_utilization_percent": max(utilization, default=None),
    }


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest_rows = {int(row["id"]): row for row in manifest["rows"]}
    selected = selected_map(args.selected_data)
    if set(manifest_rows) != set(selected):
        raise SystemExit("manifest and selected parquet IDs differ")
    groups, rows = collect_groups(args, selected)
    expected_groups = int(manifest["selected_count"])
    incomplete_groups = [group for group in groups if group["n"] != 4]
    if (len(groups) != expected_groups or incomplete_groups) and not args.allow_incomplete:
        raise SystemExit(f"expected {expected_groups} groups with n=4; got {len(groups)}")
    complete_groups = [group for group in groups if group["n"] == 4]
    status = "complete" if not incomplete_groups and len(groups) == expected_groups else "aborted_incomplete_sample_set"

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in complete_groups:
        by_source[group["source"]].append(group)
    runtime = parse_runtime(args.train_log)
    runtime_rollout = parse_runtime(args.rollout_log)
    scorer = parse_scorer_routing([args.train_log, args.rollout_log])
    representative = [
        {
            "id": group["id"], "idx": group["idx"], "source": group["source"],
            "rewards": group["rewards"], "theoretical_advantages": group["theoretical_advantages"],
            "reward_variance_population": group["reward_variance_population"],
            "unique_answers": group["unique_answers"],
        }
        for group in groups if group["class"] not in ("0/4", "4/4")
    ][:5]
    output = {
        "audit": {
            "script": "scripts/audit_rollout_difficulty_20260826.py",
            "source_data": str(args.data), "selected_data": str(args.selected_data),
            "selected_data_sha256": hashlib.sha256(args.selected_data.read_bytes()).hexdigest(),
            "manifest": str(args.manifest),
            "selection_seed": manifest["selection_seed"],
            "status": status,
            "sampled_prompt_count": len(manifest_rows),
            "sampled_source_counts": dict(Counter(row["source"] for row in manifest_rows.values())),
            "rollout_n": 4, "temperature": 0.7,
            "llm_similarity_used": False,
            "group_key": "global data id; original idx retained as path evidence",
        },
        "sample_manifest": {
            "sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "rows": sorted(manifest_rows.values(), key=lambda row: row["order"]),
        },
        "overall": group_summary(complete_groups),
        "by_source": {source: group_summary(values) for source, values in sorted(by_source.items())},
        "representative_mixed_groups": representative,
        "groups": groups,
        "completion": {
            "expected_groups": expected_groups,
            "observed_groups": len(groups),
            "complete_groups": len(complete_groups),
            "incomplete_groups": len(incomplete_groups),
            "incomplete_group_ids": [group["id"] for group in incomplete_groups],
            "excluded_incomplete_rollouts": sum(group["n"] for group in incomplete_groups),
            "note": "Distribution summaries use complete n=4 groups only when the run is aborted.",
        },
        "runtime": {
            "train_log": runtime, "rollout_log": runtime_rollout,
            "valid_rollout_files": len(rows), "expected_rollout_files": expected_groups * 4,
            "all_rollouts_valid": len(rows) == expected_groups * 4,
            "retries": (runtime["progress_last"] or {}).get("retries"),
            "gpu": gpu_peak(args.gpu_log), "checkpoint_files_observed": 0,
        },
        "scorer_routing": scorer,
        "training_metrics": {
            "optimizer_steps": 0, "advantage_logged": False,
            "pg_loss": "not applicable: rollout-only mode exited before _train_step",
            "grad_norm": "not applicable: rollout-only mode exited before _train_step",
            "entropy": "not applicable: rollout-only mode exited before _train_step",
            "old_log_prob": "not applicable: rollout-only mode exited before _train_step",
            "global_step": 0,
        },
        "theoretical_advantage_check": {
            "mixed_groups": sum(group["class"] not in ("0/4", "4/4") for group in complete_groups),
            "mixed_groups_with_nonzero_advantage": sum(
                group["class"] not in ("0/4", "4/4") and group["nonzero_theoretical_advantage"]
                for group in complete_groups
            ),
            "formula": "(reward - group_mean) / (torch.std(unbiased=True) + 1e-6)",
            "representative_groups": representative[:3],
        },
        "errors": {
            "train_log_oom_or_exceptions": runtime["oom_or_exception_lines"],
            "rollout_log_oom_or_exceptions": runtime_rollout["oom_or_exception_lines"],
            "scorer_api_or_parse_errors": scorer["api_or_parse_error_count"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"overall": output["overall"], "scorer_routing": scorer}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
