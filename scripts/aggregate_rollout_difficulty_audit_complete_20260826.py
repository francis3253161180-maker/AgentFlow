#!/usr/bin/env python3
"""Aggregate the four fixed-order chunks of the 100-prompt rollout audit.

This is an offline, deterministic reader.  It never calls a model, changes
rollout data, or treats validation output as a training rollout.
"""

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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selected-data", type=Path, required=True)
    parser.add_argument(
        "--chunk",
        nargs=5,
        action="append",
        required=True,
        metavar=("INDEX", "TRAIN_DIR", "TRAIN_LOG", "ROLLOUT_LOG", "GPU_LOG"),
        help="Repeat four times; paths must identify one completed chunk.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_chunks(values: list[list[str]]) -> list[dict[str, Any]]:
    chunks = []
    for value in values:
        index, train_dir, train_log, rollout_log, gpu_log = value
        chunks.append({
            "index": int(index),
            "train_dir": Path(train_dir),
            "train_log": Path(train_log),
            "rollout_log": Path(rollout_log),
            "gpu_log": Path(gpu_log),
        })
    indices = [chunk["index"] for chunk in chunks]
    if sorted(indices) != [0, 1, 2, 3]:
        raise SystemExit(f"expected chunk indices 0,1,2,3; got {indices}")
    return sorted(chunks, key=lambda chunk: chunk["index"])


def load_selected(path: Path) -> dict[int, dict[str, Any]]:
    import pandas as pd

    frame = pd.read_parquet(path)
    selected: dict[int, dict[str, Any]] = {}
    for row in frame.to_dict(orient="records"):
        extra = row["extra_info"]
        if isinstance(extra, str):
            extra = json.loads(extra)
        selected[int(row["id"])] = {
            "source": str(row["source"]),
            "idx": int(extra["idx"]),
        }
    return selected


def group_class(rewards: list[float]) -> str:
    return f"{sum(value == 1.0 for value in rewards)}/{len(rewards)}"


def collect_chunk(chunk: dict[str, Any], selected: dict[int, dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    paths = sorted(chunk["train_dir"].glob("step_*/idx_*/rollout_*.json"))
    if not paths:
        raise SystemExit(f"no training rollout JSON files: {chunk['train_dir']}")
    by_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        row = load_json(path)
        data_id = int(row["id"])
        if data_id not in selected:
            raise SystemExit(f"id {data_id} absent from selected parquet: {path}")
        if int(row["idx"]) != selected[data_id]["idx"]:
            raise SystemExit(f"idx mismatch for id {data_id}: {path}")
        reward = float(row["reward"])
        if reward not in (0.0, 1.0):
            raise SystemExit(f"non-binary reward {reward}: {path}")
        by_id[data_id].append({
            "path": str(path),
            "id": data_id,
            "idx": int(row["idx"]),
            "source": selected[data_id]["source"],
            "reward": reward,
            "answer_norm": normalize_answer(row.get("answer_extracted", "")),
            "path_signature": extract_tool_signature(row.get("total_result")),
        })
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
            "class": group_class(rewards),
            "reward_variance_population": statistics.pvariance(rewards) if len(rewards) > 1 else 0.0,
            "reward_std_sample": sample_std(rewards),
            "theoretical_advantages": advantages,
            "nonzero_theoretical_advantage": any(abs(value) > 1e-9 for value in advantages),
            "unique_answers": unique_answers,
            "duplicate_rate": 1.0 - unique_answers / len(members) if members else 0.0,
            "path_signature_available": path_available,
            "unique_path_signatures": len(set(path_values)) if path_available else 0,
        })
    return groups, len(paths)


def summarize(groups: list[dict[str, Any]]) -> dict[str, Any]:
    values = [reward for group in groups for reward in group["rewards"]]
    bins = {f"{count}/4": sum(group["class"] == f"{count}/4" for group in groups) for count in range(5)}
    mixed = sum(group["class"] in {"1/4", "2/4", "3/4"} for group in groups)
    nonzero = sum(group["nonzero_theoretical_advantage"] for group in groups)
    path_groups = [group for group in groups if group["path_signature_available"]]
    return {
        "groups": len(groups),
        "rollouts": len(values),
        "bin_counts": bins,
        "bin_proportions": {key: value / len(groups) if groups else 0.0 for key, value in bins.items()},
        "mixed_informative_groups": mixed,
        "mixed_informative_proportion": mixed / len(groups) if groups else 0.0,
        "overall_rollout_reward_mean": statistics.mean(values) if values else 0.0,
        "positive_reward_count": sum(value == 1.0 for value in values),
        "negative_reward_count": sum(value == 0.0 for value in values),
        "nonzero_reward_variance_groups": nonzero,
        "nonzero_reward_variance_proportion": nonzero / len(groups) if groups else 0.0,
        "mean_unique_answers_per_group": statistics.mean(group["unique_answers"] for group in groups) if groups else 0.0,
        "mean_normalized_exact_duplicate_rate": statistics.mean(group["duplicate_rate"] for group in groups) if groups else 0.0,
        "exact_duplicate_group_proportion": sum(group["unique_answers"] < 4 for group in groups) / len(groups) if groups else 0.0,
        "path_signature_available_groups": len(path_groups),
        "mean_unique_path_signatures_per_group": statistics.mean(group["unique_path_signatures"] for group in path_groups) if path_groups else None,
    }


def parse_runtime(paths: list[Path]) -> dict[str, Any]:
    text = "\n".join(ANSI_RE.sub("", path.read_text(encoding="utf-8", errors="replace")) for path in paths if path.exists())
    progress = re.findall(r"Progress: .*?\((\d+)/(\d+)\), Valid: (\d+), Retries: (\d+)", text)
    summaries = re.findall(r"Validation summary: (\d+)/(\d+) total rollouts .*?, (\d+) valid rollouts", text)
    cleanup = re.findall(r"VLLM_CLEANUP_DRIVER[^\r\n]+", text)
    error_patterns = re.compile(
        r"CUDA out of memory|OutOfMemoryError|illegal memory access|device-side assert|"
        r"blocks are not freed|Failed to reset prefix cache|RayTaskError|deadlock|"
        r"No valid rollout|worker died|drained.*False|HTTP/[^\s]+\s+5\d\d|"
        r"status[_ ]?code[=: ]+5\d\d",
        re.I,
    )
    errors = [line.strip()[-500:] for line in text.splitlines() if error_patterns.search(line)]
    update_markers = [line.strip()[-500:] for line in text.splitlines() if re.search(
        r"Training data keys|optimizer\.step|backward\(|update_actor|actor/pg_loss", line, re.I)]
    return {
        "progress_last": {
            "completed": int(progress[-1][0]), "queued": int(progress[-1][1]),
            "valid": int(progress[-1][2]), "retries": int(progress[-1][3]),
        } if progress else None,
        "validation_summary": {
            "completed": int(summaries[-1][0]), "queued": int(summaries[-1][1]),
            "valid": int(summaries[-1][2]),
        } if summaries else None,
        "cleanup_markers": cleanup,
        "cleanup_drained_true": sum("drained': True" in line or "drained=True" in line for line in cleanup),
        "cleanup_drained_false": sum("drained': False" in line or "drained=False" in line for line in cleanup),
        "errors": errors[:20],
        "unexpected_update_markers": update_markers[:20],
    }


def parse_routes(paths: list[Path]) -> dict[str, Any]:
    events: list[dict[str, str]] = []
    for path in paths:
        if not path.exists():
            continue
        text = ANSI_RE.sub("", path.read_text(encoding="utf-8", errors="replace"))
        for line in text.splitlines():
            if "HYBRID_REWARD_EVENT " not in line:
                continue
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


def gpu_peak(paths: list[Path]) -> dict[str, Any]:
    memory: list[float] = []
    utilization: list[float] = []
    for path in paths:
        if not path.exists():
            continue
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
    chunks = parse_chunks(args.chunk)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest_rows = {int(row["id"]): row for row in manifest["rows"]}
    selected = load_selected(args.selected_data)
    if set(manifest_rows) != set(selected) or len(manifest_rows) != 100:
        raise SystemExit("manifest and selected parquet do not describe the same 100 rows")

    all_groups: list[dict[str, Any]] = []
    chunk_results: list[dict[str, Any]] = []
    all_logs: list[Path] = []
    for chunk in chunks:
        groups, file_count = collect_chunk(chunk, selected)
        if len(groups) != 25 or any(group["n"] != 4 for group in groups):
            raise SystemExit(f"chunk {chunk['index']} is not exactly 25 groups x 4 rollouts")
        runtime = parse_runtime([chunk["train_log"], chunk["rollout_log"]])
        if runtime["cleanup_drained_false"] or runtime["unexpected_update_markers"] or runtime["errors"]:
            raise SystemExit(f"chunk {chunk['index']} has unsafe runtime markers")
        chunk_results.append({
            "chunk": chunk["index"],
            "train_rollout_dir": str(chunk["train_dir"]),
            "rollout_json_files": file_count,
            "summary": summarize(groups),
            "runtime": runtime,
        })
        all_groups.extend(groups)
        all_logs.extend([chunk["train_log"], chunk["rollout_log"]])

    if len(all_groups) != 100 or len({group["id"] for group in all_groups}) != 100:
        raise SystemExit("combined chunks do not contain 100 unique prompt groups")
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in all_groups:
        by_source[group["source"]].append(group)
    mixed = [group for group in all_groups if group["class"] in {"1/4", "2/4", "3/4"}]
    representative = [
        {key: group[key] for key in ("id", "idx", "source", "rewards", "theoretical_advantages", "reward_variance_population", "unique_answers")}
        for group in mixed[:5]
    ]
    runtime = parse_runtime(all_logs)
    output = {
        "audit": {
            "status": "complete",
            "manifest": str(args.manifest),
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "selected_data": str(args.selected_data),
            "selected_data_sha256": hashlib.sha256(args.selected_data.read_bytes()).hexdigest(),
            "selection_seed": manifest["selection_seed"],
            "sampled_prompt_count": 100,
            "sampled_source_counts": dict(Counter(row["source"] for row in manifest["rows"])),
            "model": "/root/autodl-tmp/models/Qwen2.5-3B-Instruct",
            "lora_rank": 8,
            "lora_alpha": 16,
            "temperature": 0.7,
            "rollout_n": 4,
            "rollout_only": True,
            "optimizer_steps": 0,
            "checkpoint_files_observed": 0,
            "llm_similarity_used": False,
        },
        "chunks": chunk_results,
        "overall": summarize(all_groups),
        "by_source": {source: summarize(groups) for source, groups in sorted(by_source.items())},
        "representative_mixed_groups": representative,
        "theoretical_advantage_check": {
            "formula": "(reward - group_mean) / (torch.std(reward, unbiased=True) + 1e-6)",
            "mixed_groups": len(mixed),
            "mixed_groups_with_nonzero_advantage": sum(group["nonzero_theoretical_advantage"] for group in mixed),
            "representative_groups": representative[:3],
        },
        "runtime": {
            "valid_rollout_files": len(all_groups) * 4,
            "expected_rollout_files": 400,
            "all_rollouts_valid": True,
            "retries": sum((chunk["runtime"]["progress_last"] or {}).get("retries", 0) for chunk in chunk_results),
            "cleanup_drained_true": sum(chunk["runtime"]["cleanup_drained_true"] for chunk in chunk_results),
            "cleanup_drained_false": sum(chunk["runtime"]["cleanup_drained_false"] for chunk in chunk_results),
            "gpu": gpu_peak([chunk["gpu_log"] for chunk in chunks]),
            "errors": runtime["errors"],
            "unexpected_update_markers": runtime["unexpected_update_markers"],
        },
        "scorer_routing": parse_routes(all_logs),
        "groups": all_groups,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["audit"]["status"], "overall": output["overall"], "scorer_routing": output["scorer_routing"]}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
