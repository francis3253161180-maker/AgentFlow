#!/usr/bin/env python3
"""Summarize the manually reviewed sample without making model/API calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-input", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--source-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def binary_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    all_rows = list(rows)
    ambiguous = sum(row["manual_label"] is None for row in all_rows)
    rows = [row for row in all_rows if row["manual_label"] is not None]
    tp = sum(row["reward"] == 1.0 and row["manual_label"] == 1 for row in rows)
    tn = sum(row["reward"] == 0.0 and row["manual_label"] == 0 for row in rows)
    fp = sum(row["reward"] == 1.0 and row["manual_label"] == 0 for row in rows)
    fn = sum(row["reward"] == 0.0 and row["manual_label"] == 1 for row in rows)
    actual_positive = tp + fn
    actual_negative = tn + fp
    return {
        "reviewed_clear_rollouts": len(rows),
        "ambiguous_or_uncertain_excluded": ambiguous,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "agreement": (tp + tn) / len(rows) if rows else None,
        "false_positive_rate": fp / actual_negative if actual_negative else None,
        "false_negative_rate": fn / actual_positive if actual_positive else None,
    }


def subset_summary(rows: list[dict[str, Any]], key: str, value: Any) -> dict[str, Any]:
    return binary_summary(row for row in rows if row.get(key) == value)


def route_family(row: dict[str, Any]) -> str:
    return "deterministic" if row["route"] == "deterministic" else "judge_fallback"


def group_status(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_key: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[(row["source"], row["idx"])].append(row)
    counts = Counter()
    details = []
    for (source, idx), members in sorted(by_key.items()):
        labels = [row["manual_label"] for row in members]
        if any(label is None for label in labels):
            status = "ambiguous"
        elif len(set(labels)) == 1:
            status = "manual_all_1" if labels[0] == 1 else "manual_all_0"
        else:
            status = "manual_mixed"
        counts[status] += 1
        details.append({
            "source": source,
            "idx": idx,
            "id": members[0]["id"],
            "scorer_class": members[0]["group_class"],
            "manual_status": status,
            "manual_labels": labels,
            "conditional_labels": [row["conditional_label"] for row in members],
        })
    return {"counts": dict(counts), "groups": details}


def conditional_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    converted = []
    for row in rows:
        copy = dict(row)
        copy["manual_label"] = row["conditional_label"]
        converted.append(copy)
    result = binary_summary(converted)
    result["interpretation"] = "sensitivity analysis only; conditional labels include ambiguous/time-dependent cases and are not the primary audit confusion matrix"
    return result


def main() -> None:
    ns = parse_args()
    review = json.loads(ns.review_input.read_text(encoding="utf-8"))
    labels = json.loads(ns.labels.read_text(encoding="utf-8"))["groups"]
    source = json.loads(ns.source_results.read_text(encoding="utf-8"))
    rows = []
    for row in review["rollouts"]:
        key = f"{row['source']}:{row['idx']}"
        annotation = labels[key]
        position = sum(1 for previous in review["rollouts"]
                       if previous["source"] == row["source"] and previous["idx"] == row["idx"]
                       and previous["timestamp"] <= row["timestamp"]) - 1
        row = dict(row)
        row["manual_label"] = annotation["labels"][position]
        row["conditional_label"] = annotation.get("conditional_labels", annotation["labels"])[position]
        row["manual_status"] = annotation["status"]
        row["manual_reason"] = annotation["reason"]
        row["manual_root_cause"] = annotation.get("root_cause")
        row["route_family"] = route_family(row)
        rows.append(row)

    clear = [row for row in rows if row["manual_label"] is not None]
    ambiguous = [row for row in rows if row["manual_label"] is None]
    by_source = {source_name: subset_summary(rows, "source", source_name) for source_name in ("nq", "mathhard")}
    by_category = {category: subset_summary(rows, "group_class", category) for category in ("0/4", "4/4")}
    by_route = {route: subset_summary(rows, "route_family", route) for route in ("deterministic", "judge_fallback")}
    all_equal = group_status(rows)
    selected_groups = {(row["source"], row["idx"]): row for row in source["groups"]
                       if (row["source"], row["idx"]) in {(x["source"], x["idx"]) for x in rows}}
    duplicate = []
    for key, group in selected_groups.items():
        duplicate.append({"source": key[0], "idx": key[1], "scorer_class": group["class"],
                          "unique_answers": group["unique_answers"], "duplicate_rate": group["duplicate_rate"],
                          "unique_path_signatures": group["unique_path_signatures"]})
    route_counts = Counter(row["route"] for row in rows)
    root_causes = Counter(row["manual_root_cause"] for row in clear + ambiguous if row["manual_root_cause"])
    primary = binary_summary(clear)
    primary["ambiguous_or_uncertain_excluded"] = len(ambiguous)
    output = {
        "audit": "outcome_reward_validity_audit_20260826",
        "source_results": str(ns.source_results),
        "source_results_sha256": hashlib.sha256(ns.source_results.read_bytes()).hexdigest(),
        "labels_sha256": hashlib.sha256(ns.labels.read_bytes()).hexdigest(),
        "manual_protocol": {
            "reviewer": "Codex direct human-style review; no LLM/API judge called",
            "primary_rule": "null/ambiguous labels excluded from TP/TN/FP/FN",
            "sampled_groups": 32,
            "sampled_rollouts": 128,
            "clear_rollouts": len(clear),
            "ambiguous_rollouts": len(ambiguous),
        },
        "scorer_routing_sample": {"route_counts": dict(route_counts), "cache_hits": sum(row["cache_hit"] for row in rows)},
        "primary_confusion_matrix": primary,
        "by_source": by_source,
        "by_scorer_group_class": by_category,
        "by_route_family": by_route,
        "conditional_sensitivity_analysis": conditional_summary(rows),
        "group_outcome": all_equal,
        "selected_group_answer_diversity": {
            "mean_unique_answers": statistics.mean(item["unique_answers"] for item in duplicate),
            "mean_exact_duplicate_rate": statistics.mean(item["duplicate_rate"] for item in duplicate),
            "mean_unique_path_signatures": statistics.mean(item["unique_path_signatures"] for item in duplicate),
            "groups": duplicate,
        },
        "root_cause_counts_on_annotated_rows": dict(root_causes),
        "rows": rows,
    }
    ns.output.parent.mkdir(parents=True, exist_ok=True)
    ns.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"primary": output["primary_confusion_matrix"], "groups": all_equal["counts"], "ambiguous": len(ambiguous)}, sort_keys=True))


if __name__ == "__main__":
    main()
