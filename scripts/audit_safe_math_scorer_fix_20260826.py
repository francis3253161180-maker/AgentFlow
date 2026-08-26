#!/usr/bin/env python3
"""Replay the safe deterministic math scorer change on the saved full audit.

This is an offline impact report.  It never invokes the hybrid judge: rows
that were judge-routed in the saved audit are retained as recorded, while the
new local decision is evaluated independently for routing/error analysis.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.utils import DeterministicDecision, deterministic_decision


def confusion(rows: list[dict[str, Any]], reward_key: str = "reward") -> dict[str, int]:
    counts = Counter()
    for row in rows:
        label = row.get("manual_label")
        if label not in {"correct", "incorrect"}:
            continue
        positive = label == "correct"
        reward = bool(float(row[reward_key]))
        if positive and reward:
            counts["TP"] += 1
        elif positive:
            counts["FN"] += 1
        elif reward:
            counts["FP"] += 1
        else:
            counts["TN"] += 1
    return {name: counts[name] for name in ("TP", "TN", "FP", "FN")}


def by_key(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for value in sorted({str(row.get(key)) for row in rows}):
        output[value] = confusion([row for row in rows if str(row.get(key)) == value])
    return output


def decision_record(row: dict[str, Any]) -> DeterministicDecision:
    return deterministic_decision(row["ground_truth"], row["candidate_answer"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    rows = audit["rows"]
    decisions = [decision_record(row) for row in rows]

    local_rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    affected: list[dict[str, Any]] = []
    for row, decision in zip(rows, decisions):
        local_row = dict(row)
        local_row["new_decision"] = decision.value
        local_row["new_reason"] = decision.reason
        local_rows.append(local_row)

        replay_row = dict(row)
        # Preserve saved judge outcomes.  Only a newly high-confidence local
        # decision is substituted in this no-network replay.
        replay_row["post_fix_reward"] = (
            float(decision.value) if decision.value is not None else float(row["reward"])
        )
        replay_rows.append(replay_row)
        if decision.value is not None and bool(decision.value) != bool(float(row["reward"])):
            affected.append(
                {
                    "group_key": row["group_key"],
                    "candidate_id": row["candidate_id"],
                    "source": row["source"],
                    "pre_fix_reward": float(row["reward"]),
                    "post_fix_reward": float(decision.value),
                    "pre_fix_route": row["route"],
                    "pre_fix_reason": row["scorer_reason"],
                    "post_fix_reason": decision.reason,
                    "manual_label": row["manual_label"],
                    "manual_reason": row["manual_reason"],
                }
            )

    deterministic_rows = [
        row for row, decision in zip(local_rows, decisions) if decision.value is not None
    ]
    for row in deterministic_rows:
        row["post_fix_reward"] = float(row["new_decision"])

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in replay_rows:
        grouped[row["group_key"]].append(row)

    output = {
        "audit": "2026-08-26 safe_math_scorer_fix offline replay",
        "source_audit": str(args.audit),
        "network_calls": 0,
        "record_count": len(rows),
        "manual_clear_count": sum(row.get("manual_label") in {"correct", "incorrect"} for row in rows),
        "manual_ambiguous_count": sum(row.get("manual_label") == "ambiguous" for row in rows),
        "pre_fix_saved_reward_confusion_clear": confusion(rows),
        "post_fix_replay_confusion_clear": confusion(replay_rows, "post_fix_reward"),
        "pre_fix_deterministic_confusion_clear": confusion(
            [row for row in rows if row.get("route_family") == "deterministic"]
        ),
        "pre_fix_deterministic_by_source": by_key(
            [row for row in rows if row.get("route_family") == "deterministic"], "source"
        ),
        "post_fix_deterministic_confusion_clear": confusion(deterministic_rows, "post_fix_reward"),
        "post_fix_deterministic_by_source": {
            source: confusion(
                [row for row in deterministic_rows if row["source"] == source],
                "post_fix_reward",
            )
            for source in sorted({row["source"] for row in deterministic_rows})
        },
        "pre_fix_route_counts": dict(Counter(row["route_family"] for row in rows)),
        "post_fix_local_decision_counts": dict(
            Counter(
                "true" if decision.value is True else "false" if decision.value is False else "judge"
                for decision in decisions
            )
        ),
        "post_fix_reason_counts": dict(Counter(decision.reason for decision in decisions)),
        "deterministic_route_count_before": sum(
            row.get("route_family") == "deterministic" for row in rows
        ),
        "deterministic_route_count_after_offline": len(deterministic_rows),
        "affected_row_count": len(affected),
        "affected_group_count": len({row["group_key"] for row in affected}),
        "affected_rows": affected,
        "new_false_positive_count_against_clear_manual": sum(
            row["manual_label"] == "incorrect" and row["post_fix_reward"] == 1.0
            for row in affected
        ),
        "new_false_negative_count_against_clear_manual": sum(
            row["manual_label"] == "correct" and row["post_fix_reward"] == 0.0
            for row in affected
        ),
        "group_reward_replay": {
            key: {
                "pre_fix_rewards": [float(row["reward"]) for row in group_rows],
                "post_fix_rewards": [float(row["post_fix_reward"]) for row in group_rows],
                "pre_fix_class": group_rows[0]["original_group_class"],
                "post_fix_class": (
                    f"{sum(row['post_fix_reward'] for row in group_rows)}/4"
                    if all(row.get("manual_label") != "ambiguous" for row in group_rows)
                    else "unresolved"
                ),
            }
            for key, group_rows in sorted(grouped.items())
            if any(row["group_key"] in {item["group_key"] for item in affected} for row in group_rows)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
