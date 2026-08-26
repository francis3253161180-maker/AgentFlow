#!/usr/bin/env python3
"""Prepare a fixed, stratified manual review from the completed audit.

This script only reads existing rollout JSON and scorer logs.  It never calls
the scorer, an API, a model, or a GPU.  The review labels are intentionally
left for the human/agent auditor to fill after reading each answer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


EVENT_RE = re.compile(r"(route|score|cache_hit|reason|error|latency_ms)=([^\s]+)")


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--chunk", nargs=3, action="append", required=True,
                        metavar=("INDEX", "TRAIN_DIR", "ROLLOUT_LOG"))
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--review-input", type=Path, required=True)
    return parser.parse_args()


def parse_chunks(values: list[list[str]]) -> list[dict[str, Any]]:
    chunks = [{"index": int(i), "train_dir": Path(d), "log": Path(l)} for i, d, l in values]
    if sorted(c["index"] for c in chunks) != [0, 1, 2, 3]:
        raise SystemExit("expected chunk indices 0,1,2,3")
    return sorted(chunks, key=lambda c: c["index"])


def read_events(path: Path) -> list[dict[str, str]]:
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "HYBRID_REWARD_EVENT " not in line:
            continue
        fields = dict(EVENT_RE.findall(line))
        if fields:
            events.append(fields)
    return events


def normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def select_groups(groups: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for category, total, quotas in (
        ("0/4", 12, {"mathhard": 3, "nq": 9}),
        ("4/4", 20, {"mathhard": 10, "nq": 10}),
    ):
        by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for group in groups:
            if group["class"] == category:
                by_source[group["source"]].append(group)
        for source in sorted(by_source):
            rng.shuffle(by_source[source])
        chosen: list[dict[str, Any]] = []
        for source in ("mathhard", "nq"):
            chosen.extend(sorted(by_source[source][:quotas[source]], key=lambda x: x["id"]))
        if len(chosen) != total:
            raise SystemExit(f"cannot select {total} groups for {category}; got {len(chosen)}")
        selected.extend(chosen)
    return sorted(selected, key=lambda group: (group["source"], group["class"], group["idx"], group["id"]))


def main() -> None:
    ns = args()
    result = json.loads(ns.results.read_text(encoding="utf-8"))
    source_manifest = json.loads(ns.manifest.read_text(encoding="utf-8"))
    chunks = parse_chunks(ns.chunk)
    groups = result["groups"]
    selected = select_groups(groups, 20260826)
    selected_keys = {(int(g["id"]), int(g["idx"])) for g in selected}

    rollout_rows: list[dict[str, Any]] = []
    for chunk in chunks:
        files = []
        for path in sorted(chunk["train_dir"].glob("step_*/idx_*/rollout_*.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            files.append((str(row["timestamp"]), path, row))
        events = read_events(chunk["log"])
        if len(files) != len(events):
            raise SystemExit(f"chunk {chunk['index']} file/event count mismatch: {len(files)} != {len(events)}")
        for (timestamp, path, row), event in zip(sorted(files, key=lambda value: value[0]), events):
            key = (int(row["id"]), int(row["idx"]))
            if key not in selected_keys:
                continue
            if int(event["score"]) != int(float(row["reward"])):
                raise SystemExit(f"telemetry score mismatch: {path}")
            total_result = row.get("total_result") or {}
            path_signature = {
                "tool_result_keys": sorted(k for k in total_result if k.startswith("tool_result_")),
                "tool_commander_keys": sorted(k for k in total_result if k.startswith("tool_commander_")),
                "step_count": total_result.get("step_count"),
            }
            rollout_rows.append({
                "chunk": chunk["index"],
                "id": int(row["id"]),
                "idx": int(row["idx"]),
                "source": next(g["source"] for g in selected if (int(g["id"]), int(g["idx"])) == key),
                "group_class": next(g["class"] for g in selected if (int(g["id"]), int(g["idx"])) == key),
                "file": str(path),
                "timestamp": timestamp,
                "question": row.get("prompt", ""),
                "groundtruth": row.get("groundtruth", ""),
                "answer_extracted": row.get("answer_extracted", ""),
                "reward": float(row["reward"]),
                "route": event.get("route"),
                "cache_hit": int(event.get("cache_hit", "0")),
                "reason": event.get("reason"),
                "latency_ms": float(event.get("latency_ms", "0")),
                "telemetry_score_match": True,
                "answer_normalized": normalize(row.get("answer_extracted", "")),
                "path_signature": path_signature,
            })

    if len(rollout_rows) != 32 * 4:
        raise SystemExit(f"expected 128 selected rollouts; got {len(rollout_rows)}")
    by_key: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rollout_rows:
        by_key[(row["id"], row["idx"])].append(row)
    if any(len(rows) != 4 for rows in by_key.values()) or len(by_key) != 32:
        raise SystemExit("selected groups do not all have four rollouts")

    sample = {
        "audit": "outcome_reward_validity_audit_20260826",
        "source_results": str(ns.results),
        "source_results_sha256": hashlib.sha256(ns.results.read_bytes()).hexdigest(),
        "source_manifest": str(ns.manifest),
        "source_manifest_sha256": hashlib.sha256(ns.manifest.read_bytes()).hexdigest(),
        "selection_seed": 20260826,
        "selection_rule": {
            "all_equal_only": True,
            "categories": {"0/4": {"total": 12, "mathhard": 3, "nq": 9},
                            "4/4": {"total": 20, "mathhard": 10, "nq": 10}},
            "method": "random.Random(20260826), independently shuffled within source/category; no answer-content filtering",
        },
        "selected_groups": [
            {"id": int(g["id"]), "idx": int(g["idx"]), "source": g["source"],
             "category": g["class"], "rewards": g["rewards"]}
            for g in selected
        ],
    }
    ns.sample_manifest.parent.mkdir(parents=True, exist_ok=True)
    ns.sample_manifest.write_text(json.dumps(sample, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    review = {"sample": sample, "rollouts": sorted(rollout_rows, key=lambda row: (row["source"], row["group_class"], row["idx"], row["timestamp"]))}
    ns.review_input.parent.mkdir(parents=True, exist_ok=True)
    ns.review_input.write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"groups": len(selected), "rollouts": len(rollout_rows), "by_category": {"0/4": 12, "4/4": 20}, "by_source": {"nq": 19, "mathhard": 13}}, sort_keys=True))


if __name__ == "__main__":
    main()
