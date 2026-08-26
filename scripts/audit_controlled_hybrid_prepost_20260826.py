#!/usr/bin/env python3
"""Summarize the completed controlled hybrid pre/train/post run.

This reads local rollout JSON and console logs only.  It deliberately emits
aggregate metrics and paths, not question/answer text or environment values.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EVENT_RE = re.compile(
    r"HYBRID_REWARD_EVENT\s+"
    r"route=(?P<route>\S+)\s+"
    r"score=(?P<score>[01])\s+"
    r"cache_hit=(?P<cache>[01])\s+"
    r"reason=(?P<reason>\S+)\s+"
    r"error=(?P<error>\S+)\s+"
    r"latency_ms=(?P<latency>[0-9.eE+-]+)"
)
METRIC_RE = re.compile(
    r"(?P<key>[A-Za-z0-9_./-]+):"
    r"(?:np\.float(?:32|64)\()?"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    r"\)?"
)
STEP_RE = re.compile(r"\bstep:(?P<step>\d+)\s+-\s+")
VALID_FINISHED_RE = re.compile(
    r"Finished after .*?Completion rate: (?P<rate>[0-9.]+)% "
    r"\((?P<completed>\d+)/(?P<total>\d+)\), Valid rollouts: (?P<valid>\d+)"
)
RETRIES_RE = re.compile(r"Retries:\s*(?P<retries>\d+)")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def json_files(path: Path) -> list[Path]:
    pattern = "idx_*/*.json" if path.name.startswith("step_") else "step_*/idx_*/*.json"
    return sorted(path.glob(pattern))


def source_map(path: Path) -> dict[int, str]:
    import pandas as pd

    frame = pd.read_parquet(path, columns=["id", "source"])
    return {int(row.id): str(row.source).lower() for row in frame.itertuples()}


def reward_summary(rows: list[dict[str, Any]], sources: dict[int, str]) -> dict[str, Any]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        groups["overall"].append(float(row["reward"]))
        groups[sources.get(int(row["id"]), "unknown")].append(float(row["reward"]))

    result: dict[str, Any] = {}
    for group, values in sorted(groups.items(), key=lambda item: (item[0] != "overall", item[0])):
        positive = sum(value > 0.5 for value in values)
        negative = len(values) - positive
        result[group] = {
            "count": len(values),
            "reward_positive": positive,
            "reward_negative": negative,
            "reward_mean": statistics.fmean(values) if values else None,
            "reward_variance_population": statistics.pvariance(values) if len(values) > 1 else 0.0,
        }
    return result


def parse_events(path: Path) -> list[dict[str, Any]]:
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = EVENT_RE.search(line)
        if match:
            item = match.groupdict()
            item["score"] = int(item["score"])
            item["cache_hit"] = int(item.pop("cache"))
            item["latency_ms"] = float(item["latency"])
            del item["latency"]
            events.append(item)
    return events


def event_summary(events: list[dict[str, Any]], phase: str) -> dict[str, Any]:
    routes = Counter(event["route"] for event in events)
    errors = Counter(event["error"] for event in events if event["error"] != "none")
    uncached = [event["latency_ms"] for event in events if event["route"] == "judge"]
    cached = [event["latency_ms"] for event in events if event["route"] == "judge_cache"]

    def latency(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"count": 0, "mean_ms": None, "median_ms": None, "p95_ms": None}
        ordered = sorted(values)
        index = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
        return {
            "count": len(values),
            "mean_ms": statistics.fmean(values),
            "median_ms": statistics.median(values),
            "p95_ms": ordered[index],
        }

    return {
        "phase": phase,
        "event_count": len(events),
        "route_counts": dict(sorted(routes.items())),
        "deterministic_count": routes.get("deterministic", 0),
        "judge_fallback_count": routes.get("judge", 0) + routes.get("judge_cache", 0),
        "judge_api_call_count": routes.get("judge", 0),
        "cache_hit_count": sum(event["cache_hit"] for event in events),
        "api_error_count": sum(errors.values()),
        "api_errors": dict(sorted(errors.items())),
        "uncached_latency": latency(uncached),
        "cached_latency": latency(cached),
    }


def parse_train_metrics(path: Path) -> list[dict[str, Any]]:
    metrics = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        step_match = STEP_RE.search(line)
        if not step_match:
            continue
        values = {match.group("key"): float(match.group("value")) for match in METRIC_RE.finditer(line)}
        values["step"] = int(step_match.group("step"))
        metrics.append(values)
    return metrics


def rollout_progress(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    finished = [match.groupdict() for match in VALID_FINISHED_RE.finditer(text)]
    retries = [int(match.group("retries")) for match in RETRIES_RE.finditer(text)]
    return {
        "finished_batches": len(finished),
        "finished_batch_records": [
            {
                "completion_rate_percent": float(item["rate"]),
                "completed": int(item["completed"]),
                "total": int(item["total"]),
                "valid": int(item["valid"]),
            }
            for item in finished
        ],
        "valid_rollouts_total_from_finished_lines": sum(int(item["valid"]) for item in finished),
        "retry_observations": retries,
        "retry_count_sum": sum(retries),
        "retry_count_max": max(retries) if retries else 0,
    }


def phase_events(events: list[dict[str, Any]], pre_count: int, train_count: int) -> dict[str, Any]:
    return {
        "pre_validation": event_summary(events[:pre_count], "pre_validation"),
        "training_rollouts": event_summary(events[pre_count : pre_count + train_count], "training_rollouts"),
        "post_validation": event_summary(events[pre_count + train_count :], "post_validation"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--rollout-log", type=Path, required=True)
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train_root = args.rollout_root / "train"
    validation_root = args.rollout_root / "validation"
    sources = source_map(args.data)
    pre_rows = [load_json(path) for path in json_files(validation_root / "step_0")]
    post_rows = [load_json(path) for path in json_files(validation_root / "step_20")]
    train_rows = [load_json(path) for path in json_files(train_root)]
    events = parse_events(args.rollout_log)
    metric_rows = [
        row for row in parse_train_metrics(args.train_log)
        if row.get("training/global_step", 0.0) > 0
    ]
    finished = rollout_progress(args.train_log)

    checkpoint_files = [str(path) for path in args.checkpoint_dir.rglob("*") if path.is_file()] if args.checkpoint_dir.exists() else []
    fatal_text = (args.train_log.read_text(encoding="utf-8", errors="replace") + args.rollout_log.read_text(encoding="utf-8", errors="replace"))
    fatal_patterns = ["CUDA out of memory", "OutOfMemoryError", "No valid rollout", "No valid rollouts"]
    fatal_matches = {pattern: fatal_text.lower().count(pattern.lower()) for pattern in fatal_patterns}
    framework_errors = [
        line.strip()[:300]
        for line in args.rollout_log.read_text(encoding="utf-8", errors="replace").splitlines()
        if "[ERROR]" in line
    ]

    # The event stream is emitted once per reward call in this run: 20 pre,
    # 40 train (20 examples x rollout.n=2), 20 post.
    grouped_events = phase_events(events, len(pre_rows), len(train_rows))
    train_metric_keys = [
        "critic/rewards/mean",
        "critic/advantages/mean",
        "actor/pg_loss",
        "actor/grad_norm",
        "actor/entropy_loss",
        "timing_s/old_log_prob",
        "timing_s/update_actor",
        "training/global_step",
    ]
    train_metrics = [
        {key: row[key] for key in ["step", *train_metric_keys] if key in row}
        for row in metric_rows
    ]

    by_id = Counter(int(row["id"]) for row in train_rows)
    output = {
        "experiment": "controlled_flow_grpo_hybrid_prepost_20260826",
        "run_id": "20260826_hybrid_prepost_112050",
        "head_before_run": "c1d78f2",
        "data": {
            "train_and_validation": str(args.data),
            "validation_note": "Validation intentionally reused the 20-row mini20 file for NQ/mathhard paired pre/post comparison; it is not held-out generalization.",
            "source_counts": dict(sorted(Counter(sources.values()).items())),
        },
        "fixed_configuration": {
            "model": "Qwen2.5-3B-Instruct",
            "lora_rank": 8,
            "lora_alpha": 16,
            "dataset": "mini20 seed20260825",
            "train_batch_size": 2,
            "ppo_mini_batch_size": 2,
            "micro_batch_size_per_gpu": 1,
            "rollout_n": 2,
            "max_prompt_length": 1280,
            "max_response_length": 384,
            "learning_rate": 1e-5,
            "gpus": 1,
            "epochs": 1,
            "trainer_val_before_train": True,
            "trainer_save_freq": 0,
            "hydra_validation_override": str(args.data),
        },
        "pre_validation": {
            "json_count": len(pre_rows),
            "reward": reward_summary(pre_rows, sources),
        },
        "post_validation": {
            "json_count": len(post_rows),
            "reward": reward_summary(post_rows, sources),
        },
        "training_rollouts": {
            "json_count": len(train_rows),
            "unique_example_ids": len(by_id),
            "rollouts_per_example_counts": dict(sorted(Counter(by_id.values()).items())),
            "reward": reward_summary(train_rows, sources),
        },
        "training_metrics": {
            "metric_row_count": len(train_metrics),
            "rows": train_metrics,
            "max_gpu_memory_allocated_gb": max((row.get("perf/max_memory_allocated_gb", 0.0) for row in metric_rows), default=None),
            "max_gpu_memory_reserved_gb": max((row.get("perf/max_memory_reserved_gb", 0.0) for row in metric_rows), default=None),
            "max_gpu_memory_metric_source": "perf/max_memory_allocated_gb and perf/max_memory_reserved_gb in train log",
        },
        "rollout_progress": finished,
        "scorer": {
            "event_count": len(events),
            "phase_assignment": "event order: 20 pre, 40 training (20 rows x rollout.n=2), 20 post",
            **grouped_events,
            "overall": event_summary(events, "overall"),
        },
        "safety_checks": {
            "fatal_pattern_matches": fatal_matches,
            "framework_error_lines_in_rollout_log": framework_errors,
            "checkpoint_files_in_experiment_dir": checkpoint_files,
            "processes_stopped_after_run": True,
            "gpu_idle_after_run_observed": True,
        },
        "evidence_paths": {
            "train_log": str(args.train_log),
            "rollout_log": str(args.rollout_log),
            "rollout_root": str(args.rollout_root),
            "checkpoint_dir": str(args.checkpoint_dir),
        },
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
