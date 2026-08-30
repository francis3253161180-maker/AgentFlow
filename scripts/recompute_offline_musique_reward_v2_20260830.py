#!/usr/bin/env python3
"""Recompute a completed MuSiQue pack under coverage-v1 and exact-set-v2."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentflow.offline_musique import (
    OfflineCorpus,
    sha256_file,
    terminal_reward,
    terminal_reward_coverage_v1,
)


def summarize(rows: list[dict], qids: list[str], n: int) -> dict:
    by_qid = defaultdict(list)
    for row in rows:
        by_qid[row["qid"]].append(row)
    per_qid = []
    histogram = Counter()
    for qid in qids:
        group = sorted(by_qid[qid], key=lambda item: item["rollout_index"])
        if [row["rollout_index"] for row in group] != list(range(n)):
            raise ValueError(f"incomplete rollout group: {qid}")
        vector = [row["reward"] for row in group]
        count = sum(vector)
        histogram[f"{count}/{n}"] += 1
        per_qid.append(
            {
                "qid": qid,
                "reward_vector": vector,
                "success_count": count,
                "population_variance": statistics.pvariance(vector),
            }
        )
    positives = sum(row["reward"] for row in rows)
    return {
        "histogram": {f"{k}/{n}": histogram.get(f"{k}/{n}", 0) for k in range(n + 1)},
        "positive_count": positives,
        "reward_mean": positives / len(rows),
        "mixed_group_count": sum(0 < row["success_count"] < n for row in per_qid),
        "mixed_group_rate": sum(0 < row["success_count"] < n for row in per_qid) / len(qids),
        "per_qid": per_qid,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--detail", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    corpus = OfflineCorpus.load(args.corpus)
    detail = json.loads(args.detail.read_text(encoding="utf-8"))
    qids = detail["qids"]
    trajectories = detail["trajectories"]
    n = int(detail["configuration"]["n"])
    if len(qids) != 32 or n != 8 or len(trajectories) != 256:
        raise SystemExit("expected completed 32x8 diagnostic pack")

    old_rows = []
    new_rows = []
    changed = []
    duplicate_rejections = 0
    for trajectory in trajectories:
        selected = trajectory["selected_pids"]
        old = terminal_reward_coverage_v1(
            corpus, trajectory["qid"], trajectory["final_answer"], selected
        )
        new = terminal_reward(corpus, trajectory["qid"], trajectory["final_answer"], selected)
        persisted = int(trajectory["reward_detail"]["reward"])
        if persisted != old["reward"]:
            raise ValueError(f"persisted historical reward mismatch: {trajectory['trajectory_id']}")
        common = {
            "trajectory_id": trajectory["trajectory_id"],
            "qid": trajectory["qid"],
            "rollout_index": trajectory["rollout_index"],
        }
        old_rows.append({**common, "reward": old["reward"]})
        new_rows.append({**common, "reward": new["reward"]})
        if old["reward"] != new["reward"]:
            changed.append({**common, "coverage_v1": old, "exact_set_v2": new})
        duplicate_rejections += sum(
            rejected.get("reason") == "duplicate_evidence"
            for transition in trajectory["transitions"]
            if transition["mode"] == "EVIDENCE_UPDATE"
            for rejected in transition["validation_result"].get("rejected", [])
        )

    result = {
        "schema_version": 1,
        "experiment": "completed_n8_reward_v1_vs_v2_recompute",
        "source_detail_path": str(args.detail.resolve()),
        "source_detail_sha256": sha256_file(args.detail),
        "corpus_path": str(args.corpus.resolve()),
        "corpus_sha256": sha256_file(args.corpus),
        "qid_count": len(qids),
        "rollout_n": n,
        "trajectory_count": len(trajectories),
        "coverage_v1": summarize(old_rows, qids, n),
        "exact_set_v2": summarize(new_rows, qids, n),
        "changed_trajectory_count": len(changed),
        "changed_trajectories": changed,
        "duplicate_selection_rejection_count": duplicate_rejections,
        "conclusion": "exact-set v2 is the frozen terminal reward for PRE, train, and POST",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
