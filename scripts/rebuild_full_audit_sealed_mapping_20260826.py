#!/usr/bin/env python3
"""Rebuild the final 44-group sealed mapping from raw rollout evidence.

Used after the discarded set-order preparation runs.  Group identity is
matched by question/ground truth; candidate metadata order is reconstructed
from the original seeded candidate shuffle and timestamp order.  No model or
network call is made.
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


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--complete-results", type=Path, required=True)
    p.add_argument("--blinded", type=Path, required=True)
    p.add_argument("--chunk", nargs=3, action="append", required=True,
                   metavar=("INDEX", "TRAIN_DIR", "ROLLOUT_LOG"))
    p.add_argument("--output-sealed", type=Path, required=True)
    p.add_argument("--exposure-manifest", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260826)
    return p.parse_args()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def events(path: Path) -> list[dict[str, str]]:
    pattern = re.compile(r"(route|score|cache_hit|reason|error|latency_ms)=([^\s]+)")
    result = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "HYBRID_REWARD_EVENT " in line:
            fields = dict(pattern.findall(line))
            if fields:
                result.append(fields)
    return result


def main() -> None:
    a = args()
    complete = load(a.complete_results)
    blinded = load(a.blinded)
    group_by_id_idx = {(int(g["id"]), int(g["idx"])): g for g in complete["groups"]}
    raw_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    chunks = sorted((int(i), Path(d), Path(l)) for i, d, l in a.chunk)
    for index, train_dir, rollout_log in chunks:
        paths = list(train_dir.glob("step_*/idx_*/rollout_*.json"))
        pairs = [(path, load(path)) for path in paths]
        pairs.sort(key=lambda item: item[1].get("timestamp", ""))
        route = events(rollout_log)
        if len(pairs) != len(route):
            raise SystemExit(f"chunk {index}: JSON/event mismatch")
        for (path, row), telemetry in zip(pairs, route):
            meta = group_by_id_idx[(int(row["id"]), int(row["idx"]))]
            member = {
                "reward": float(row["reward"]),
                "route": telemetry.get("route", "unknown"),
                "cache_hit": telemetry.get("cache_hit", "0") == "1",
                "reason": telemetry.get("reason", "unknown"),
                "error": telemetry.get("error", "none"),
                "latency_ms": telemetry.get("latency_ms"),
                "file": str(path), "timestamp": row.get("timestamp", ""),
                "tools": row.get("tools", []), "total_result": row.get("total_result"),
            }
            raw_groups[(str(row.get("prompt", "")), str(row.get("groundtruth", "")))].append(member)
    if len(raw_groups) != 100 or any(len(values) != 4 for values in raw_groups.values()):
        raise SystemExit("raw evidence did not produce 100 unique question/GT groups x4")

    rng = random.Random(a.seed)
    dummy = list(range(44))
    rng.shuffle(dummy)
    permutations = []
    for _ in range(44):
        order = list(range(4))
        rng.shuffle(order)
        permutations.append(order)

    sealed_groups = []
    for position, blind_group in enumerate(blinded["groups"]):
        raw = list(raw_groups[(str(blind_group["question"]), str(blind_group["ground_truth"]))])
        raw.sort(key=lambda member: member["timestamp"])
        order = permutations[position]
        ordered = [raw[index] for index in order]
        answer_texts = [candidate["candidate_answer"] for candidate in blind_group["candidates"]]
        # Check the reconstructed permutation against the blinded answer order.
        # Raw answer text is not stored in the temporary member structure, so
        # use the source JSON path to validate via a second direct read.
        source_answers = []
        for member in ordered:
            source_answers.append(str(load(Path(member["file"])).get("answer_extracted", "")))
        if source_answers != answer_texts:
            raise SystemExit(f"candidate order mismatch at {blind_group['opaque_id']}")
        source_path = Path(ordered[0]["file"])
        source_row = load(source_path)
        meta = group_by_id_idx[(int(source_row["id"]), int(source_row["idx"]))]
        sealed_groups.append({
            "opaque_id": blind_group["opaque_id"],
            "group_key": f"{meta['source']}:{int(meta['idx'])}",
            "source": meta["source"], "id": int(meta["id"]), "idx": int(meta["idx"]),
            "original_group_class": meta["class"], "original_rewards": meta["rewards"],
            "members": [
                {"candidate_id": f"candidate-{i}", **member}
                for i, member in enumerate(ordered, 1)
            ],
        })
    sealed = {
        "audit": "SEALED full-audit mapping rebuilt from raw evidence",
        "selection_seed": a.seed,
        "blinded_sha256": hashlib.sha256(json.dumps(blinded, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        "groups": sealed_groups,
    }
    a.output_sealed.write_text(json.dumps(sealed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    exposure = load(a.exposure_manifest)
    exposure["canonical_blind_order"] = [
        {"opaque_id": group["opaque_id"], "group_key": group["group_key"],
         "candidate_ids": [member["candidate_id"] for member in group["members"]]}
        for group in sealed_groups
    ]
    exposure["blind_mapping_recovery"] = {
        "used_after_manual_labels": True,
        "reason": "The final sealed mapping was rebuilt from raw chunk evidence after discarding set-order-mismatched intermediate mappings.",
        "validated_candidate_order": True,
    }
    a.exposure_manifest.write_text(json.dumps(exposure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sealed_groups": len(sealed_groups), "candidate_order_validated": True}))


if __name__ == "__main__":
    main()
