#!/usr/bin/env python3
"""Prepare the full 100-group outcome audit and blind the 44 unexposed groups.

The script reads existing rollout evidence only.  It does not call a model or
alter the scorer.  The blind file contains only question/ground truth/answers
for groups not present in either prior exposure manifest.
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--complete-results", type=Path, required=True)
    p.add_argument("--legacy-manifest", type=Path, required=True)
    p.add_argument("--metadata-results", type=Path, required=True)
    p.add_argument("--chunk", nargs=3, action="append", required=True,
                   metavar=("INDEX", "TRAIN_DIR", "ROLLOUT_LOG"))
    p.add_argument("--seed", type=int, default=20260826)
    p.add_argument("--blinded-output", type=Path, required=True)
    p.add_argument("--exposure-manifest-output", type=Path, required=True)
    p.add_argument("--sealed-output", type=Path, required=True)
    return p.parse_args()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def route_events(path: Path) -> list[dict[str, str]]:
    pattern = re.compile(r"(route|score|cache_hit|reason|error|latency_ms)=([^\s]+)")
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "HYBRID_REWARD_EVENT " in line:
            fields = dict(pattern.findall(line))
            if fields:
                events.append(fields)
    return events


def collect_chunk(index: int, train_dir: Path, rollout_log: Path) -> list[dict[str, Any]]:
    paths = list(train_dir.glob("step_*/idx_*/rollout_*.json"))
    pairs = [(path, load(path)) for path in paths]
    pairs.sort(key=lambda item: item[1].get("timestamp", ""))
    events = route_events(rollout_log)
    if len(pairs) != len(events):
        raise SystemExit(f"chunk {index}: JSON/event mismatch {len(pairs)} vs {len(events)}")
    rows = []
    for (path, row), event in zip(pairs, events):
        rows.append({
            "chunk": index,
            "file": str(path),
            "id": int(row["id"]),
            "idx": int(row["idx"]),
            "question": str(row.get("prompt", "")),
            "ground_truth": str(row.get("groundtruth", "")),
            "candidate_answer": str(row.get("answer_extracted", "")),
            "reward": float(row["reward"]),
            "route": event.get("route", "unknown"),
            "cache_hit": event.get("cache_hit", "0") == "1",
            "reason": event.get("reason", "unknown"),
            "error": event.get("error", "none"),
            "latency_ms": event.get("latency_ms"),
            "timestamp": row.get("timestamp", ""),
            "tools": row.get("tools", []),
            "total_result": row.get("total_result"),
        })
    return rows


def key(source: str, idx: int) -> str:
    return f"{source}:{idx}"


def main() -> None:
    a = parse_args()
    complete = load(a.complete_results)
    legacy = load(a.legacy_manifest)
    metadata = load(a.metadata_results)
    group_meta = {
        key(str(group["source"]), int(group["idx"])): group
        for group in complete["groups"]
    }
    if len(group_meta) != 100:
        raise SystemExit(f"expected 100 complete groups, got {len(group_meta)}")

    legacy_keys = {
        key(str(group["source"]), int(group["idx"]))
        for group in legacy["selected_groups"]
    }
    metadata_keys = {
        key(str(group["source"]), int(group["idx"]))
        for group in metadata["groups"]
    }
    if len(legacy_keys) != 32 or len(metadata_keys) != 24 or legacy_keys & metadata_keys:
        raise SystemExit("exposure manifests are not disjoint 32 + 24 groups")
    exposed = legacy_keys | metadata_keys
    remaining = set(group_meta) - exposed
    if len(exposed) != 56 or len(remaining) != 44:
        raise SystemExit(f"expected 56 exposed/44 unexposed, got {len(exposed)}/{len(remaining)}")

    chunks = []
    for index, train_dir, rollout_log in a.chunk:
        chunks.append((int(index), Path(train_dir), Path(rollout_log)))
    if sorted(index for index, _, _ in chunks) != [0, 1, 2, 3]:
        raise SystemExit("expected chunks 0,1,2,3")
    raw_rows = []
    for index, train_dir, rollout_log in sorted(chunks):
        raw_rows.extend(collect_chunk(index, train_dir, rollout_log))
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        source = str(group_meta[key(next(g["source"] for g in complete["groups"] if int(g["id"]) == row["id"] and int(g["idx"]) == row["idx"]), row["idx"])]["source"])
        # The complete result is keyed by source and idx; IDs are retained only
        # in the sealed map and are never used in the blind selection rule.
        row["source"] = source
        by_key[key(source, row["idx"])].append(row)
    if len(by_key) != 100 or any(len(values) != 4 for values in by_key.values()):
        raise SystemExit("raw evidence is not exactly 100 groups x 4 rollouts")

    rng = random.Random(a.seed)
    # Sort before seeded shuffling; set iteration order is process/hash-seed
    # dependent and would otherwise make the audit mapping non-reproducible.
    blind_keys = sorted(remaining)
    rng.shuffle(blind_keys)
    blinded_groups = []
    sealed_groups = []
    exposure_groups = []
    # Exposure manifest is intentionally metadata-rich; it is never read by
    # the manual blind-review phase.
    for group_key in sorted(group_meta):
        meta = group_meta[group_key]
        if group_key in legacy_keys:
            status = "exposed"
            provenance = "exposed_legacy_32_prior_manual_audit"
        elif group_key in metadata_keys:
            status = "exposed"
            provenance = "exposed_metadata_24_same_session_audit"
        else:
            status = "unexposed"
            provenance = "unexposed_remaining_full_audit_blind_phase"
        exposure_groups.append({
            "group_key": group_key,
            "source": meta["source"],
            "id": int(meta["id"]),
            "idx": int(meta["idx"]),
            "exposure_status": status,
            "exposure_provenance": provenance,
            "prior_legacy_manifest_member": group_key in legacy_keys,
            "metadata_audit_member": group_key in metadata_keys,
        })

    for number, group_key in enumerate(blind_keys, 1):
        opaque_id = f"full-blind-{number:03d}"
        members = list(by_key[group_key])
        members.sort(key=lambda row: row["timestamp"])
        # Candidate order is randomized independently of metadata/reward.
        rng.shuffle(members)
        question = members[0]["question"]
        ground_truth = members[0]["ground_truth"]
        if any(row["question"] != question or row["ground_truth"] != ground_truth for row in members):
            raise SystemExit(f"question/GT mismatch in {group_key}")
        blinded_groups.append({
            "opaque_id": opaque_id,
            "question": question,
            "ground_truth": ground_truth,
            "candidates": [
                {"candidate_id": f"candidate-{i}", "candidate_answer": row["candidate_answer"]}
                for i, row in enumerate(members, 1)
            ],
        })
        sealed_groups.append({
            "opaque_id": opaque_id,
            "group_key": group_key,
            "source": group_meta[group_key]["source"],
            "id": int(group_meta[group_key]["id"]),
            "idx": int(group_meta[group_key]["idx"]),
            "original_group_class": group_meta[group_key]["class"],
            "original_rewards": group_meta[group_key]["rewards"],
            "members": [
                {
                    "candidate_id": f"candidate-{i}",
                    "reward": row["reward"],
                    "route": row["route"],
                    "cache_hit": row["cache_hit"],
                    "reason": row["reason"],
                    "error": row["error"],
                    "latency_ms": row["latency_ms"],
                    "file": row["file"],
                    "timestamp": row["timestamp"],
                    "tools": row["tools"],
                    "total_result": row["total_result"],
                }
                for i, row in enumerate(members, 1)
            ],
        })

    blinded = {
        "audit": "2026-08-26 full outcome-reward audit blind phase",
        "selection_seed": a.seed,
        "selected_count": len(blinded_groups),
        "groups": blinded_groups,
    }
    exposure_manifest = {
        "audit": "2026-08-26 full outcome-reward validity audit exposure manifest",
        "base_commit": "4cc7090",
        "selection_seed": a.seed,
        "complete_results": str(a.complete_results),
        "complete_results_sha256": digest(a.complete_results),
        "legacy_manifest": str(a.legacy_manifest),
        "legacy_manifest_sha256": digest(a.legacy_manifest),
        "metadata_results": str(a.metadata_results),
        "metadata_results_sha256": digest(a.metadata_results),
        "total_groups": 100,
        "exposed_groups": len(exposed),
        "unexposed_groups": len(remaining),
        "exposure_provenance_counts": {
            "exposed_legacy_32_prior_manual_audit": len(legacy_keys),
            "exposed_metadata_24_same_session_audit": len(metadata_keys),
            "unexposed_remaining_full_audit_blind_phase": len(remaining),
        },
        "groups": exposure_groups,
    }
    sealed = {
        "audit": "SEALED full-audit mapping; open only after blind labels are saved and validated",
        "selection_seed": a.seed,
        "blinded_sha256": hashlib.sha256(json.dumps(blinded, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        "groups": sealed_groups,
    }
    for path, value in ((a.blinded_output, blinded), (a.exposure_manifest_output, exposure_manifest), (a.sealed_output, sealed)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"total_groups": 100, "exposed_groups": 56, "unexposed_groups": 44,
                      "blinded_output": str(a.blinded_output),
                      "exposure_manifest_output": str(a.exposure_manifest_output),
                      "sealed_output": str(a.sealed_output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
