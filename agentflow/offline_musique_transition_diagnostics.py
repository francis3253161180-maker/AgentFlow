"""Post-hoc scorer-only transition diagnostics for offline MuSiQue.

This module is deliberately not imported by the actor runner.  It consumes a
completed raw trajectory pack and scorer records after generation, so none of
the values computed here can enter compact memory, prompts, or rewards.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from typing import Any, Iterable


TOLERANCE = 1e-12


def support_scores(selected: set[str], gold: set[str] | frozenset[str]) -> dict[str, Any]:
    intersection = selected & set(gold)
    precision = len(intersection) / len(selected) if selected else 1.0
    recall = len(intersection) / len(gold) if gold else 1.0
    f1_denominator = precision + recall
    f2_denominator = 4 * precision + recall
    f1 = 2 * precision * recall / f1_denominator if f1_denominator else 0.0
    f2 = 5 * precision * recall / f2_denominator if f2_denominator else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "F2": f2,
        "selected_gold_count": len(intersection),
        "selected_total_count": len(selected),
        "gold_support_count": len(gold),
        "empty_selected_bookkeeping_precision": not selected,
        "full_support_coverage": set(gold).issubset(selected),
        "exact_support_set": selected == set(gold),
    }


def unique_with_tolerance(values: Iterable[float], tolerance: float = TOLERANCE) -> int:
    unique: list[float] = []
    for value in sorted(values):
        if not unique or abs(value - unique[-1]) > tolerance:
            unique.append(value)
    return len(unique)


def distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "variance": None,
            "min": None,
            "max": None,
            "range": None,
            "unique_with_tolerance": 0,
        }
    variance = statistics.pvariance(values)
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "std": math.sqrt(variance),
        "variance": variance,
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
        "unique_with_tolerance": unique_with_tolerance(values),
    }


def signed_distribution(values: list[float], tolerance: float = TOLERANCE) -> dict[str, Any]:
    counts = Counter(
        "positive" if value > tolerance else "negative" if value < -tolerance else "zero"
        for value in values
    )
    total = len(values)
    return {
        **distribution(values),
        "positive_count": counts["positive"],
        "zero_count": counts["zero"],
        "negative_count": counts["negative"],
        "positive_rate": counts["positive"] / total if total else 0.0,
        "zero_rate": counts["zero"] / total if total else 0.0,
        "negative_rate": counts["negative"] / total if total else 0.0,
    }


def annotate_trajectory(trajectory: dict[str, Any], gold: set[str] | frozenset[str]) -> dict[str, Any]:
    selected: set[str] = set()
    previous_f1 = 0.0
    previous_f2 = 0.0
    updates: list[dict[str, Any]] = []
    for transition in trajectory["transitions"]:
        if transition["mode"] != "EVIDENCE_UPDATE":
            continue
        validation = transition["validation_result"]
        if validation.get("format_failure") or not validation.get("schema_valid", False):
            continue
        accepted = {row["pid"] for row in validation.get("accepted", [])}
        new = accepted - selected
        selected.update(accepted)
        scores = support_scores(selected, gold)
        delta_f1 = scores["F1"] - previous_f1
        delta_f2 = scores["F2"] - previous_f2
        updates.append(
            {
                "evidence_update_ordinal": len(updates) + 1,
                "transition_index": transition["transition_index"],
                **scores,
                "delta_F1": delta_f1,
                "delta_F2": delta_f2,
                "new_gold_support_count": len(new & set(gold)),
                "new_distractor_count": len(new - set(gold)),
            }
        )
        previous_f1 = scores["F1"]
        previous_f2 = scores["F2"]

    expected_selected = set(trajectory["selected_pids"])
    if selected != expected_selected:
        raise ValueError(
            f"validated transition pids disagree with final selected_pids for {trajectory['trajectory_id']}: "
            f"{sorted(selected)} != {sorted(expected_selected)}"
        )
    final = support_scores(selected, gold)
    return {
        "trajectory_id": trajectory["trajectory_id"],
        "qid": trajectory["qid"],
        "rollout_index": trajectory["rollout_index"],
        "transition_scores": updates,
        "final_support_scores": final,
        "has_positive_delta_F1": any(row["delta_F1"] > TOLERANCE for row in updates),
        "has_positive_delta_F2": any(row["delta_F2"] > TOLERANCE for row in updates),
    }


def summarize_diagnostics(
    trajectories: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    gold_by_qid: dict[str, set[str] | frozenset[str]],
    qid_order: list[str],
    rollout_n: int = 8,
) -> dict[str, Any]:
    by_qid_trajectories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_qid_annotations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    annotations_by_id = {row["trajectory_id"]: row for row in annotations}
    for trajectory in trajectories:
        by_qid_trajectories[trajectory["qid"]].append(trajectory)
        by_qid_annotations[trajectory["qid"]].append(annotations_by_id[trajectory["trajectory_id"]])

    if set(by_qid_trajectories) != set(qid_order):
        raise ValueError("trajectory qids do not match fixed qid order")
    for qid in qid_order:
        rows = by_qid_trajectories[qid]
        indexes = sorted(row["rollout_index"] for row in rows)
        if indexes != list(range(rollout_n)):
            raise ValueError(f"qid {qid} does not contain exactly rollout indexes 0..{rollout_n - 1}")

    outcome_histogram = Counter()
    groups: list[dict[str, Any]] = []
    all_delta_f1: list[float] = []
    all_delta_f2: list[float] = []
    ordinal_f1: dict[int, list[float]] = defaultdict(list)
    ordinal_f2: dict[int, list[float]] = defaultdict(list)
    final_f1_mixed_count = 0
    final_f2_mixed_count = 0
    transition_f1_signal_count = 0
    transition_f2_signal_count = 0
    both_sign_f1_count = 0
    both_sign_f2_count = 0

    for qid in qid_order:
        trajectory_rows = sorted(by_qid_trajectories[qid], key=lambda row: row["rollout_index"])
        annotation_rows = sorted(by_qid_annotations[qid], key=lambda row: row["rollout_index"])
        rewards = [int(row["reward_detail"]["reward"]) for row in trajectory_rows]
        successes = sum(rewards)
        outcome_histogram[f"{successes}/{rollout_n}"] += 1
        final_f1 = [row["final_support_scores"]["F1"] for row in annotation_rows]
        final_f2 = [row["final_support_scores"]["F2"] for row in annotation_rows]
        f1_stats = distribution(final_f1)
        f2_stats = distribution(final_f2)
        f1_mixed = f1_stats["variance"] > TOLERANCE
        f2_mixed = f2_stats["variance"] > TOLERANCE
        final_f1_mixed_count += f1_mixed
        final_f2_mixed_count += f2_mixed

        q_delta_f1 = [
            update["delta_F1"] for row in annotation_rows for update in row["transition_scores"]
        ]
        q_delta_f2 = [
            update["delta_F2"] for row in annotation_rows for update in row["transition_scores"]
        ]
        all_delta_f1.extend(q_delta_f1)
        all_delta_f2.extend(q_delta_f2)
        for row in annotation_rows:
            for update in row["transition_scores"]:
                ordinal_f1[update["evidence_update_ordinal"]].append(update["delta_F1"])
                ordinal_f2[update["evidence_update_ordinal"]].append(update["delta_F2"])
        f1_nonzero = any(abs(value) > TOLERANCE for value in q_delta_f1)
        f2_nonzero = any(abs(value) > TOLERANCE for value in q_delta_f2)
        f1_signal = f1_nonzero and unique_with_tolerance(q_delta_f1) > 1
        f2_signal = f2_nonzero and unique_with_tolerance(q_delta_f2) > 1
        transition_f1_signal_count += f1_signal
        transition_f2_signal_count += f2_signal
        both_f1 = any(value > TOLERANCE for value in q_delta_f1) and any(
            value < -TOLERANCE for value in q_delta_f1
        )
        both_f2 = any(value > TOLERANCE for value in q_delta_f2) and any(
            value < -TOLERANCE for value in q_delta_f2
        )
        both_sign_f1_count += both_f1
        both_sign_f2_count += both_f2
        groups.append(
            {
                "qid": qid,
                "terminal_reward_vector": rewards,
                "terminal_success_count": successes,
                "outcome_group_type": "all_zero" if successes == 0 else "all_one" if successes == rollout_n else "mixed",
                "final_F1": {**f1_stats, "values": final_f1, "mixed": f1_mixed},
                "final_F2": {**f2_stats, "values": final_f2, "mixed": f2_mixed},
                "delta_F1": {**signed_distribution(q_delta_f1), "question_signal": f1_signal},
                "delta_F2": {**signed_distribution(q_delta_f2), "question_signal": f2_signal},
                "contains_both_positive_and_negative_delta_F1": both_f1,
                "contains_both_positive_and_negative_delta_F2": both_f2,
            }
        )

    group_count = len(qid_order)
    mixed_count = sum(row["outcome_group_type"] == "mixed" for row in groups)
    all_zero_count = sum(row["outcome_group_type"] == "all_zero" for row in groups)
    all_one_count = sum(row["outcome_group_type"] == "all_one" for row in groups)
    reward_count = sum(sum(row["terminal_reward_vector"]) for row in groups)
    trajectory_count = len(trajectories)

    failure_counts = Counter()
    for trajectory in trajectories:
        gold = set(gold_by_qid[trajectory["qid"]])
        retrieved = set(trajectory["retrieved_pids"])
        selected = set(trajectory["selected_pids"])
        failure_counts["retrieval_miss"] += not gold.issubset(retrieved)
        failure_counts["support_returned_but_not_selected"] += bool((retrieved & gold) - selected)
        failure_counts["distractor_selection"] += bool(selected - gold)
        failure_counts["premature_answer"] += (
            trajectory["termination_reason"] == "answer" and not gold.issubset(selected)
        )
    max_failure = max(failure_counts.values(), default=0)

    return {
        "qid_count": group_count,
        "rollout_n": rollout_n,
        "trajectory_count": trajectory_count,
        "outcome": {
            "per_qid": [
                {
                    "qid": row["qid"],
                    "terminal_reward_vector": row["terminal_reward_vector"],
                    "success_count": row["terminal_success_count"],
                    "group_type": row["outcome_group_type"],
                }
                for row in groups
            ],
            "success_histogram": {f"{k}/{rollout_n}": outcome_histogram.get(f"{k}/{rollout_n}", 0) for k in range(rollout_n + 1)},
            "all_zero_group_count": all_zero_count,
            "all_zero_group_rate": all_zero_count / group_count,
            "mixed_group_count": mixed_count,
            "mixed_group_rate": mixed_count / group_count,
            "all_one_group_count": all_one_count,
            "all_one_group_rate": all_one_count / group_count,
            "grounded_positive_count": reward_count,
            "mean_terminal_reward": reward_count / trajectory_count,
        },
        "final_support_score_diversity": {
            "per_qid": [
                {"qid": row["qid"], "final_F1": row["final_F1"], "final_F2": row["final_F2"]}
                for row in groups
            ],
            "final_F1_mixed_group_count": final_f1_mixed_count,
            "final_F1_mixed_group_rate": final_f1_mixed_count / group_count,
            "final_F2_mixed_group_count": final_f2_mixed_count,
            "final_F2_mixed_group_rate": final_f2_mixed_count / group_count,
            "exact_support_set_trajectory_count": sum(
                row["final_support_scores"]["exact_support_set"] for row in annotations
            ),
            "exact_support_set_trajectory_rate": sum(
                row["final_support_scores"]["exact_support_set"] for row in annotations
            ) / trajectory_count,
        },
        "question_level_transition_signal_availability": {
            "label": "question-level transition signal availability; heterogeneous transition states are not valid GRPO normalization groups",
            "delta_F1_overall": signed_distribution(all_delta_f1),
            "delta_F2_overall": signed_distribution(all_delta_f2),
            "question_transition_F1_signal_count": transition_f1_signal_count,
            "question_transition_F1_signal_rate": transition_f1_signal_count / group_count,
            "question_transition_F2_signal_count": transition_f2_signal_count,
            "question_transition_F2_signal_rate": transition_f2_signal_count / group_count,
            "questions_with_both_positive_and_negative_delta_F1_count": both_sign_f1_count,
            "questions_with_both_positive_and_negative_delta_F1_rate": both_sign_f1_count / group_count,
            "questions_with_both_positive_and_negative_delta_F2_count": both_sign_f2_count,
            "questions_with_both_positive_and_negative_delta_F2_rate": both_sign_f2_count / group_count,
            "trajectories_with_positive_delta_F1_count": sum(row["has_positive_delta_F1"] for row in annotations),
            "trajectories_with_positive_delta_F1_rate": sum(row["has_positive_delta_F1"] for row in annotations) / trajectory_count,
            "trajectories_with_positive_delta_F2_count": sum(row["has_positive_delta_F2"] for row in annotations),
            "trajectories_with_positive_delta_F2_rate": sum(row["has_positive_delta_F2"] for row in annotations) / trajectory_count,
            "evidence_update_ordinal_delta_F1": {
                str(ordinal): signed_distribution(values) for ordinal, values in sorted(ordinal_f1.items())
            },
            "evidence_update_ordinal_delta_F2": {
                str(ordinal): signed_distribution(values) for ordinal, values in sorted(ordinal_f2.items())
            },
            "per_qid": [
                {
                    "qid": row["qid"],
                    "delta_F1": row["delta_F1"],
                    "delta_F2": row["delta_F2"],
                    "contains_both_positive_and_negative_delta_F1": row["contains_both_positive_and_negative_delta_F1"],
                    "contains_both_positive_and_negative_delta_F2": row["contains_both_positive_and_negative_delta_F2"],
                }
                for row in groups
            ],
        },
        "failure_taxonomy": {
            key: {"trajectory_count": failure_counts[key], "trajectory_rate": failure_counts[key] / trajectory_count}
            for key in ("retrieval_miss", "support_returned_but_not_selected", "distractor_selection", "premature_answer")
        }
        | {"dominant_by_trajectory_count": sorted(key for key, value in failure_counts.items() if value == max_failure)},
    }
