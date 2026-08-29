#!/usr/bin/env python3
"""Create reproducible concurrency evidence from persisted rollout timing and logs.

This is observability only: it neither starts an engine nor changes AgentFlow
semantics.  The vLLM-request count is deliberately labelled as a log-derived
proxy because the server log has request-start and HTTP-completion events but
does not attach a per-request correlation id.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


RUNNER_DONE = re.compile(r"\[Worker (\d+) \| Rollout (rollout-[^\]]+)\] Completed")
TIMESTAMP = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:[,\.]\d+)?)")


def parse_log_time(line: str) -> float | None:
    match = TIMESTAMP.search(line)
    if not match:
        return None
    value = match.group(1).replace(",", ".")
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def maximum_active(intervals: list[dict[str, Any]]) -> tuple[int, list[dict[str, Any]]]:
    events: list[tuple[float, int]] = []
    for item in intervals:
        events.extend(((item["start_unix"], 1), (item["end_unix"], -1)))
    # An end at exactly the same time precedes a new start, avoiding a false
    # overlap at a zero-width boundary.
    events.sort(key=lambda value: (value[0], value[1]))
    active = peak = 0
    segments: list[dict[str, Any]] = []
    previous: float | None = None
    for timestamp, delta in events:
        if previous is not None and timestamp > previous and active:
            segments.append({"start_unix": previous, "end_unix": timestamp, "active": active})
        active += delta
        peak = max(peak, active)
        previous = timestamp
    return peak, segments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--rollout-log", type=Path, required=True)
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--baseline-wall-seconds", type=float, required=True)
    parser.add_argument("--aggregate", type=Path, required=True,
                        help="Existing bounded-smoke aggregate for resource/cleanup summary.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    aggregate = json.loads(args.aggregate.read_text(encoding="utf-8"))

    runner_text = args.rollout_log.read_text(encoding="utf-8", errors="replace")
    worker_by_rollout = {
        match.group(2): int(match.group(1)) for match in RUNNER_DONE.finditer(runner_text)
    }
    intervals: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for path in sorted(args.rollout_dir.rglob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        timing = raw.get("systems_timing") or {}
        rollout_id = timing.get("rollout_id")
        start, end = timing.get("solve_started_unix"), timing.get("solve_finished_unix")
        if not isinstance(rollout_id, str) or not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            raise SystemExit(f"missing persisted timing in {path}")
        if rollout_id in seen_ids:
            raise SystemExit(f"duplicate rollout_id {rollout_id}")
        seen_ids.add(rollout_id)
        intervals.append({
            "rollout_id": rollout_id,
            "worker_id": worker_by_rollout.get(rollout_id),
            "rollout_file": str(path),
            "start_unix": float(start),
            "end_unix": float(end),
            "duration_seconds": float(end) - float(start),
            "reward": raw.get("reward"),
        })
    if len(intervals) != 4 or any(item["worker_id"] is None for item in intervals):
        raise SystemExit("expected four uniquely owned completed rollouts")

    peak_rollouts, rollout_segments = maximum_active(intervals)
    start = min(item["start_unix"] for item in intervals)
    end = max(item["end_unix"] for item in intervals)
    wall = end - start

    # vLLM proxy: a route log is emitted immediately before its request; the
    # TaskRunner emits the corresponding completion.  We only retain intervals
    # while the global start/complete count remains well formed.
    requests: list[dict[str, Any]] = []
    pending: list[float] = []
    malformed = False
    for line in args.train_log.read_text(encoding="utf-8", errors="replace").splitlines():
        timestamp = parse_log_time(line)
        if timestamp is None:
            continue
        if "UNIFIED_ROLE_ROUTE request" in line:
            pending.append(timestamp)
        elif 'POST /v1/chat/completions HTTP/1.1" 200' in line:
            if not pending:
                malformed = True
                continue
            request_start = pending.pop(0)
            requests.append({"start_unix": request_start, "end_unix": timestamp})
    if pending:
        malformed = True
    request_peak, request_segments = maximum_active(requests) if requests else (0, [])

    result = {
        "schema_version": 1,
        "purpose": "systems-only N_WORKERS=2 / max_num_seqs=2 rollout overlap smoke",
        "rollout_ownership": {
            "expected_count": 4,
            "persisted_file_count": len(intervals),
            "unique_rollout_ids": len(seen_ids),
            "worker_counts": dict(sorted(Counter(item["worker_id"] for item in intervals).items())),
            "all_valid_json": True,
        },
        "rollout_intervals": intervals,
        "rollout_concurrency": {
            "maximum_simultaneously_active": peak_rollouts,
            "active_segments": [item for item in rollout_segments if item["active"] >= 2],
            "group_wall_seconds": wall,
            "sum_rollout_seconds": sum(item["duration_seconds"] for item in intervals),
        },
        "vllm_request_log_proxy": {
            "definition": "UNIFIED_ROLE_ROUTE request until next TaskRunner HTTP 200, FIFO; no request correlation ids are logged",
            "completed_request_count": len(requests),
            "well_formed": not malformed,
            "maximum_in_flight": request_peak,
            "active_segments": [item for item in request_segments if item["active"] >= 2],
        },
        "baseline_comparison": {
            "baseline": "N_WORKERS=1, max_num_seqs=2, same frozen sample n=4",
            "baseline_group_wall_seconds": args.baseline_wall_seconds,
            "concurrent_group_wall_seconds": wall,
            "wall_clock_speedup": args.baseline_wall_seconds / wall,
            "baseline_rollouts_per_minute": 4 / (args.baseline_wall_seconds / 60),
            "concurrent_rollouts_per_minute": 4 / (wall / 60),
        },
        "bounded_smoke_summary": {
            "reward_vector": aggregate.get("reward_vector"),
            "gpu_peak_memory_mib": aggregate.get("gpu_peak_memory_mib"),
            "search_telemetry_totals": aggregate.get("search_telemetry_totals"),
            "fatal_lifecycle_markers": aggregate.get("fatal_lifecycle_markers"),
            "cleanup_markers": aggregate.get("cleanup_markers"),
        },
        "reward_vector": [item["reward"] for item in intervals],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "peak_rollouts": peak_rollouts,
        "vllm_proxy_peak": request_peak,
        "group_wall_seconds": wall,
        "speedup": args.baseline_wall_seconds / wall,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
