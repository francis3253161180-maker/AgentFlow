#!/usr/bin/env python3
"""Merge and summarize the complete 100-group outcome-reward audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from audit_rollout_diversity_20260826 import extract_tool_signature, normalize_answer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--complete-results", type=Path, required=True)
    p.add_argument("--legacy-results", type=Path, required=True)
    p.add_argument("--metadata-results", type=Path, required=True)
    p.add_argument("--blinded", type=Path, required=True)
    p.add_argument("--new-labels", type=Path, required=True)
    p.add_argument("--sealed", type=Path, required=True)
    p.add_argument("--exposure-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manual-labels-output", type=Path, required=True)
    return p.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group_key(source: str, idx: int) -> str:
    return f"{source}:{int(idx)}"


def label(value: Any) -> str:
    if value in (1, "1", "correct"):
        return "correct"
    if value in (0, "0", "incorrect"):
        return "incorrect"
    return "ambiguous"


def route_family(route: str) -> str:
    if route == "deterministic":
        return "deterministic"
    if route in {"judge", "judge_cache"}:
        return "judge_fallback"
    return "unknown"


def confusion(rows: list[dict[str, Any]]) -> dict[str, Any]:
    clear = [row for row in rows if row["manual_label"] != "ambiguous"]
    tp = sum(row["reward"] == 1.0 and row["manual_label"] == "correct" for row in clear)
    tn = sum(row["reward"] == 0.0 and row["manual_label"] == "incorrect" for row in clear)
    fp = sum(row["reward"] == 1.0 and row["manual_label"] == "incorrect" for row in clear)
    fn = sum(row["reward"] == 0.0 and row["manual_label"] == "correct" for row in clear)
    return {
        "rows_total": len(rows), "rows_clear": len(clear), "ambiguous": len(rows) - len(clear),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "agreement_clear": (tp + tn) / len(clear) if clear else None,
        "fp_rate_clear": fp / len(clear) if clear else None,
        "fn_rate_clear": fn / len(clear) if clear else None,
        "positive_reward": sum(row["reward"] == 1.0 for row in rows),
        "negative_reward": sum(row["reward"] == 0.0 for row in rows),
    }


def breakdown(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row[field])].append(row)
    return {key: confusion(values) for key, values in sorted(buckets.items())}


def corrected_status(labels: list[str]) -> tuple[str, str | None]:
    if "ambiguous" in labels:
        return "unresolved", None
    count = labels.count("correct")
    return "resolvable", f"{count}/4"


def corrected_group_summary(groups: list[dict[str, Any]]) -> dict[str, Any]:
    resolvable = [group for group in groups if group["corrected_status"] == "resolvable"]
    unresolved = [group for group in groups if group["corrected_status"] == "unresolved"]
    bins = Counter(group["corrected_class"] for group in resolvable)
    mixed = sum(group["corrected_class"] in {"1/4", "2/4", "3/4"} for group in resolvable)
    original_equal = [group for group in groups if group["original_group_class"] in {"0/4", "4/4"}]
    original_mixed = [group for group in groups if group["original_group_class"] in {"1/4", "2/4", "3/4"}]
    return {
        "groups_total": len(groups), "resolvable_groups": len(resolvable), "unresolved_groups": len(unresolved),
        "unresolved_group_ids": [group["group_key"] for group in unresolved],
        "corrected_bin_counts_resolvable": {f"{i}/4": bins.get(f"{i}/4", 0) for i in range(5)},
        "corrected_bin_proportions_resolvable": {f"{i}/4": bins.get(f"{i}/4", 0) / len(resolvable) if resolvable else 0.0 for i in range(5)},
        "corrected_mixed_groups_resolvable": mixed,
        "corrected_mixed_ratio_resolvable": mixed / len(resolvable) if resolvable else None,
        "corrected_mixed_ratio_lower_bound_all_groups": mixed / len(groups) if groups else None,
        "corrected_mixed_ratio_upper_bound_all_groups": (mixed + len(unresolved)) / len(groups) if groups else None,
        "original_all_equal_groups": len(original_equal),
        "original_all_equal_to_manual_mixed": sum(group["corrected_class"] in {"1/4", "2/4", "3/4"} for group in original_equal if group["corrected_status"] == "resolvable"),
        "original_all_equal_unresolved": sum(group["corrected_status"] == "unresolved" for group in original_equal),
        "original_mixed_groups": len(original_mixed),
        "original_mixed_to_manual_all_equal": sum(group["corrected_class"] in {"0/4", "4/4"} for group in original_mixed if group["corrected_status"] == "resolvable"),
        "original_mixed_unresolved": sum(group["corrected_status"] == "unresolved" for group in original_mixed),
    }


def main() -> None:
    a = parse_args()
    complete = json.loads(a.complete_results.read_text(encoding="utf-8"))
    legacy_results = json.loads(a.legacy_results.read_text(encoding="utf-8"))
    metadata_results = json.loads(a.metadata_results.read_text(encoding="utf-8"))
    blinded = json.loads(a.blinded.read_text(encoding="utf-8"))
    new_labels = json.loads(a.new_labels.read_text(encoding="utf-8"))
    sealed = json.loads(a.sealed.read_text(encoding="utf-8"))
    exposure = json.loads(a.exposure_manifest.read_text(encoding="utf-8"))
    blinded_hash = hashlib.sha256(json.dumps(blinded, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if blinded_hash != sealed["blinded_sha256"]:
        raise SystemExit("blinded file does not match sealed mapping")

    complete_by_key = {
        group_key(str(group["source"]), int(group["idx"])): group
        for group in complete["groups"]
    }
    exposure_by_key = {entry["group_key"]: entry for entry in exposure["groups"]}
    if len(complete_by_key) != 100 or len(exposure_by_key) != 100:
        raise SystemExit("expected 100 complete groups and exposure entries")

    rows: list[dict[str, Any]] = []
    group_labels: dict[str, list[str]] = defaultdict(list)
    group_answers: dict[str, list[str]] = defaultdict(list)
    group_paths: dict[str, list[Any]] = defaultdict(list)
    group_info: dict[str, dict[str, Any]] = {}
    diversity_override: dict[str, dict[str, Any]] = {}
    metadata_groups_by_opaque = {
        group["opaque_id"]: group for group in metadata_results["groups"]
    }
    for item in metadata_results.get("diversity_by_group", []):
        metadata_group = metadata_groups_by_opaque[item["opaque_id"]]
        diversity_override[group_key(str(metadata_group["source"]), int(metadata_group["idx"]))] = item

    def add_row(row: dict[str, Any], exposure_status: str, provenance: str, review_status: str) -> None:
        source = str(row["source"])
        idx = int(row["idx"])
        gkey = group_key(source, idx)
        manual = label(row.get("manual_label"))
        candidate_id = row.get("candidate_id") or f"candidate-{len(group_labels[gkey]) + 1}"
        standardized = {
            "group_key": gkey,
            "group_id": int(row["id"]),
            "idx": idx,
            "candidate_id": candidate_id,
            "source": source,
            "original_group_class": row.get("group_class", complete_by_key[gkey]["class"]),
            "question": row["question"],
            "ground_truth": row["groundtruth"],
            "candidate_answer": row["answer_extracted"],
            "reward": float(row["reward"]),
            "manual_label": manual,
            "manual_reason": row.get("manual_reason", ""),
            "exposure_status": exposure_status,
            "exposure_provenance": provenance,
            "manual_review_status": review_status,
            "route": row.get("route", "unknown"),
            "route_family": route_family(row.get("route", "unknown")),
            "cache_hit": bool(row.get("cache_hit", False)),
            "scorer_reason": row.get("reason", "unknown"),
            "scorer_error": row.get("error", "none"),
        }
        rows.append(standardized)
        group_labels[gkey].append(manual)
        group_answers[gkey].append(normalize_answer(row["answer_extracted"]))
        group_paths[gkey].append(row.get("path_signature"))
        group_info[gkey] = {
            "group_key": gkey, "group_id": int(row["id"]), "idx": idx, "source": source,
            "question": row["question"], "ground_truth": row["groundtruth"],
            "original_group_class": row.get("group_class", complete_by_key[gkey]["class"]),
            "original_rewards": complete_by_key[gkey]["rewards"],
            "exposure_status": exposure_status, "exposure_provenance": provenance,
        }

    # Legacy 32: the prior audit already contains row-level metadata and labels.
    for row in legacy_results["rows"]:
        gkey = group_key(str(row["source"]), int(row["idx"]))
        add_row(row, "exposed", "exposed_legacy_32_prior_manual_audit", "exposed_nonblind_prior_label_reused")

    # Metadata 24: it was metadata-blinded in this same session, but is exposed
    # for this final audit and is not counted as strict independent blind evidence.
    for row in metadata_results["rows"]:
        old = {
            "id": row["id"], "idx": row["idx"], "source": row["source"],
            "question": row["question"], "groundtruth": row["ground_truth"],
            "answer_extracted": row["candidate_answer"], "reward": row["reward"],
            "manual_label": row["manual_label"], "manual_reason": row["manual_reason"],
            "route": row["route"], "cache_hit": row["cache_hit"], "reason": row["reason"], "error": row["error"],
        }
        add_row(old, "exposed", "exposed_metadata_24_same_session_audit", "exposed_nonblind_prior_label_reused")

    # New 44: only now read the sealed mapping and combine with labels.
    label_by_opaque = new_labels["groups"]
    blind_by_opaque = {group["opaque_id"]: group for group in blinded["groups"]}
    for sealed_group in sealed["groups"]:
        opaque = sealed_group["opaque_id"]
        blinded_group = blind_by_opaque[opaque]
        label_group = label_by_opaque[opaque]
        answer_by_candidate = {candidate["candidate_id"]: candidate["candidate_answer"] for candidate in blinded_group["candidates"]}
        for member, manual, reason in zip(sealed_group["members"], label_group["labels"], label_group["reasons"]):
            row = {
                "id": sealed_group["id"], "idx": sealed_group["idx"], "source": sealed_group["source"],
                "question": blinded_group["question"], "groundtruth": blinded_group["ground_truth"],
                "answer_extracted": answer_by_candidate[member["candidate_id"]], "reward": member["reward"],
                "manual_label": manual, "manual_reason": reason, "route": member["route"],
                "cache_hit": member["cache_hit"], "reason": member["reason"], "error": member["error"],
                "path_signature": extract_tool_signature(member.get("total_result")),
            }
            add_row(row, "unexposed", "unexposed_remaining_full_audit_blind_phase", "metadata_blind_same_session_new_label")

    if len(rows) != 400 or len(group_info) != 100 or any(len(group_labels[key]) != 4 for key in group_info):
        raise SystemExit(f"expected 400 rows/100 groups x4, got {len(rows)}/{len(group_info)}")
    if len({row["group_key"] + ":" + row["candidate_id"] for row in rows}) != 400:
        raise SystemExit("duplicate group/candidate rows")

    groups = []
    for gkey in sorted(group_info):
        info = group_info[gkey]
        labels = group_labels[gkey]
        status, corrected_class = corrected_status(labels)
        path_values = group_paths[gkey]
        path_available = all(value is not None for value in path_values)
        override = diversity_override.get(gkey)
        if override is not None:
            path_available = bool(override["path_signature_available"])
            unique_path_signatures = override["unique_path_signatures"]
        else:
            unique_path_signatures = len(set(json.dumps(value, sort_keys=True, ensure_ascii=False) for value in path_values)) if path_available else None
        groups.append({
            **info,
            "manual_labels": labels,
            "manual_ambiguous_count": labels.count("ambiguous"),
            "manual_correct_count": labels.count("correct"),
            "manual_incorrect_count": labels.count("incorrect"),
            "corrected_status": status,
            "corrected_class": corrected_class,
            "unique_normalized_answers": len(set(group_answers[gkey])),
            "normalized_exact_duplicate_rate": 1.0 - len(set(group_answers[gkey])) / 4,
            "path_signature_available": path_available,
            "unique_path_signatures": unique_path_signatures,
        })

    clear = [row for row in rows if row["manual_label"] != "ambiguous"]
    all_equal = [group for group in groups if group["original_group_class"] in {"0/4", "4/4"}]
    original_mixed = [group for group in groups if group["original_group_class"] in {"1/4", "2/4", "3/4"}]
    judge_fp = [row for row in clear if row["route_family"] == "judge_fallback" and row["reward"] == 1.0 and row["manual_label"] == "incorrect"]
    judge_fn = [row for row in clear if row["route_family"] == "judge_fallback" and row["reward"] == 0.0 and row["manual_label"] == "correct"]
    deterministic_fp = [row for row in clear if row["route_family"] == "deterministic" and row["reward"] == 1.0 and row["manual_label"] == "incorrect"]
    deterministic_fn = [row for row in clear if row["route_family"] == "deterministic" and row["reward"] == 0.0 and row["manual_label"] == "correct"]
    ambiguous_rows = [row for row in rows if row["manual_label"] == "ambiguous"]
    math_reason = [row for row in rows if "math" in row["scorer_reason"].casefold()]
    yes_no_reason = [row for row in rows if "yes_no" in row["scorer_reason"].casefold()]

    partitions = {
        "all": rows,
        "exposed_legacy_32": [row for row in rows if row["exposure_provenance"] == "exposed_legacy_32_prior_manual_audit"],
        "exposed_metadata_24_same_session": [row for row in rows if row["exposure_provenance"] == "exposed_metadata_24_same_session_audit"],
        "unexposed_blind_44_same_session_metadata_blind": [row for row in rows if row["exposure_status"] == "unexposed"],
    }
    group_partitions = {
        key: [group for group in groups if (group["exposure_provenance"] == provenance)]
        for key, provenance in {
            "exposed_legacy_32": "exposed_legacy_32_prior_manual_audit",
            "exposed_metadata_24_same_session": "exposed_metadata_24_same_session_audit",
            "unexposed_blind_44_same_session_metadata_blind": "unexposed_remaining_full_audit_blind_phase",
        }.items()
    }
    diversity = {
        "groups": len(groups),
        "mean_unique_normalized_answers_per_group": statistics.mean(group["unique_normalized_answers"] for group in groups),
        "mean_normalized_exact_duplicate_rate": statistics.mean(group["normalized_exact_duplicate_rate"] for group in groups),
        "groups_with_any_exact_duplicate": sum(group["unique_normalized_answers"] < 4 for group in groups),
        "path_signature_available_groups": sum(group["path_signature_available"] for group in groups),
        "mean_unique_path_signatures": statistics.mean(group["unique_path_signatures"] for group in groups if group["path_signature_available"]),
    }
    output = {
        "audit": {
            "name": "full 100-group / 400-rollout outcome-reward validity audit",
            "base_commit": exposure["base_commit"],
            "groups": len(groups), "rollouts": len(rows),
            "new_rollouts": 0, "network_or_llm_calls": 0,
            "same_session_blind_limitation": True,
            "limitation": "The 44-group metadata-blind phase was performed in the same Codex session that had seen the prior 32+24 groups; it is non-overlap metadata-blinded evidence, not an independent human blind study.",
        },
        "exposure": {
            "counts": exposure["exposure_provenance_counts"],
            "strict_independent_blind_groups": 0,
            "metadata_blinded_nonoverlap_groups": 44,
            "exposed_groups": 56,
        },
        "primary_confusion_matrix": confusion(rows),
        "by_source": breakdown(rows, "source"),
        "by_route": breakdown(rows, "route"),
        "by_route_family": breakdown(rows, "route_family"),
        "by_original_group_class": breakdown(rows, "original_group_class"),
        "by_exposure_partition": {key: confusion(value) for key, value in partitions.items()},
        "corrected_group_outcome_all": corrected_group_summary(groups),
        "corrected_group_outcome_by_exposure": {key: corrected_group_summary(value) for key, value in group_partitions.items()},
        "corrected_group_outcome_by_source": {
            source: corrected_group_summary([group for group in groups if group["source"] == source])
            for source in sorted({group["source"] for group in groups})
        },
        "diversity_and_correlation": diversity,
        "recurrence_and_root_cause_checks": {
            "scorer_false_positive_rows_clear": len([row for row in clear if row["reward"] == 1.0 and row["manual_label"] == "incorrect"]),
            "scorer_false_negative_rows_clear": len([row for row in clear if row["reward"] == 0.0 and row["manual_label"] == "correct"]),
            "judge_false_positive_rows": len(judge_fp),
            "judge_false_negative_rows": len(judge_fn),
            "deterministic_false_positive_rows": len(deterministic_fp),
            "deterministic_false_negative_rows": len(deterministic_fn),
            "math_reason_rows": len(math_reason),
            "math_reason_mismatch_rows": sum("mismatch" in row["scorer_reason"] for row in math_reason),
            "yes_no_reason_rows": len(yes_no_reason),
            "ambiguous_rows": len(ambiguous_rows),
            "ambiguous_group_count": sum(group["corrected_status"] == "unresolved" for group in groups),
            "ambiguous_reasons": Counter(row["manual_reason"] for row in ambiguous_rows),
            "judge_false_positive_examples": [
                {"group_key": row["group_key"], "candidate_id": row["candidate_id"], "reason": row["manual_reason"]}
                for row in judge_fp[:20]
            ],
            "false_negative_examples": [
                {"group_key": row["group_key"], "candidate_id": row["candidate_id"], "route": row["route"], "ground_truth": row["ground_truth"], "answer": row["candidate_answer"][:500]}
                for row in [row for row in clear if row["reward"] == 0.0 and row["manual_label"] == "correct"][:20]
            ],
            "manual_ambiguous_group_keys": [group["group_key"] for group in groups if group["corrected_status"] == "unresolved"],
            "no_data_leakage_conclusion": "No direct evidence of data leakage was established by this audit; shared tool/path signatures and repeated answers show correlation but are not proof of leakage.",
        },
        "inputs": {name: sha256(path) for name, path in {
            "complete_results": a.complete_results, "legacy_results": a.legacy_results,
            "metadata_results": a.metadata_results, "blinded": a.blinded,
            "new_labels": a.new_labels, "sealed_mapping": a.sealed,
            "exposure_manifest": a.exposure_manifest,
        }.items()},
        "rows": rows,
        "groups": groups,
    }
    # Counter is not JSON serializable in the nested root-cause section.
    output["recurrence_and_root_cause_checks"]["ambiguous_reasons"] = dict(output["recurrence_and_root_cause_checks"]["ambiguous_reasons"])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manual_output = {
        "audit": "2026-08-26 full outcome-reward audit unified manual labels",
        "protocol": {
            "all_400_rows_present": True,
            "metadata_blind_new_review_groups": 44,
            "exposed_prior_review_groups": 56,
            "same_session_not_independent_blind": True,
            "labels": ["correct", "incorrect", "ambiguous"],
        },
        "rows": rows,
        "groups": groups,
    }
    a.manual_labels_output.parent.mkdir(parents=True, exist_ok=True)
    a.manual_labels_output.write_text(json.dumps(manual_output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "groups": len(groups), "primary": output["primary_confusion_matrix"], "corrected": output["corrected_group_outcome_all"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
