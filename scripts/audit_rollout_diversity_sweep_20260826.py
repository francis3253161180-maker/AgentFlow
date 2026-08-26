#!/usr/bin/env python3
"""Aggregate the four rollout-only diversity conditions deterministically."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_rollout_diversity_20260826 import (  # noqa: E402
    extract_tool_signature,
    load_json,
    load_source_map,
    normalize_answer,
    sample_std,
    theoretical_advantages,
)


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--selected-data", type=Path, required=True)
    parser.add_argument(
        "--condition",
        action="append",
        required=True,
        metavar="NAME|ROOT|LOG",
        help="Repeat once per condition.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def class_for(rewards: list[float]) -> str:
    if all(value == 1.0 for value in rewards):
        return "all-1"
    if all(value == 0.0 for value in rewards):
        return "all-0"
    return "mixed"


def summarize(groups: list[dict[str, Any]], rows: list[dict[str, Any]], expected: int, log: Path) -> dict[str, Any]:
    total = len(groups)
    all_one = sum(group["class"] == "all-1" for group in groups)
    all_zero = sum(group["class"] == "all-0" for group in groups)
    mixed = sum(group["class"] == "mixed" for group in groups)
    nonzero = sum(group["nonzero_advantage"] for group in groups)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        by_source[group["source"]].append(group)

    clean_log = ANSI_RE.sub("", log.read_text(encoding="utf-8", errors="replace"))
    progress = re.findall(r"Progress: .*?\((\d+)/(\d+)\), Valid: (\d+), Retries: (\d+)", clean_log)
    final_summary = re.findall(
        r"Validation summary: (\d+)/(\d+) total rollouts .*?, (\d+) valid rollouts",
        clean_log,
    )
    reward_events = clean_log.count("HYBRID_REWARD_EVENT ")
    reward_errors = sum(
        1
        for line in clean_log.splitlines()
        if "HYBRID_REWARD_EVENT " in line and "error=none" not in line
    )
    update_markers = bool(
        re.search(r"Training data keys|optimizer\.step|(^| )actor/pg_loss:", clean_log, re.I | re.M)
    )

    result = {
        "rollouts": len(rows),
        "expected_rollouts": expected,
        "valid_rollouts_from_files": len(rows),
        "valid_rate": len(rows) / expected if expected else 0.0,
        "groups": total,
        "all_groups_have_expected_n": all(len(group["rewards"]) == expected // 10 for group in groups),
        "all_1_groups": all_one,
        "all_0_groups": all_zero,
        "mixed_reward_groups": mixed,
        "all_1_proportion": all_one / total if total else 0.0,
        "all_0_proportion": all_zero / total if total else 0.0,
        "mixed_reward_proportion": mixed / total if total else 0.0,
        "nonzero_reward_variance_groups": nonzero,
        "nonzero_reward_variance_proportion": nonzero / total if total else 0.0,
        "groups_with_nonzero_theoretical_grpo_advantage": nonzero,
        "nonzero_theoretical_grpo_advantage_proportion": nonzero / total if total else 0.0,
        "mean_reward": statistics.mean(row["reward"] for row in rows) if rows else 0.0,
        "positive_reward_count": sum(row["reward"] == 1.0 for row in rows),
        "negative_reward_count": sum(row["reward"] == 0.0 for row in rows),
        "mean_unique_answers_per_group": statistics.mean(
            group["unique_answers"] for group in groups
        )
        if groups
        else 0.0,
        "normalized_exact_duplicate_rate": statistics.mean(
            group["duplicate_rate"] for group in groups
        )
        if groups
        else 0.0,
        "exact_duplicate_group_proportion": sum(
            group["unique_answers"] < len(group["rewards"]) for group in groups
        ) / total
        if total
        else 0.0,
        "groups_with_reliable_path_signature": sum(
            group["path_signature_available"] for group in groups
        ),
        "mean_unique_path_signatures_per_group": statistics.mean(
            group["unique_path_signatures"] for group in groups
            if group["path_signature_available"]
        )
        if any(group["path_signature_available"] for group in groups)
        else None,
        "theoretical_advantage_abs_mean": statistics.mean(
            abs(value)
            for group in groups
            for value in group["theoretical_advantages"]
        )
        if groups
        else 0.0,
        "by_source": {
            source: summarize_source(source_groups)
            for source, source_groups in sorted(by_source.items())
        },
        "progress_last": (
            {
                "completed": int(progress[-1][0]),
                "queued": int(progress[-1][1]),
                "valid": int(progress[-1][2]),
                "retries": int(progress[-1][3]),
            }
            if progress
            else None
        ),
        "validation_summary": (
            {
                "completed": int(final_summary[-1][0]),
                "queued": int(final_summary[-1][1]),
                "valid": int(final_summary[-1][2]),
            }
            if final_summary
            else None
        ),
        "hybrid_reward_event_count_in_log": reward_events,
        "hybrid_reward_error_event_count_in_log": reward_errors,
        "unexpected_training_update_marker": update_markers,
    }
    return result


def summarize_source(groups: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(groups)
    all_one = sum(group["class"] == "all-1" for group in groups)
    all_zero = sum(group["class"] == "all-0" for group in groups)
    mixed = sum(group["class"] == "mixed" for group in groups)
    nonzero = sum(group["nonzero_advantage"] for group in groups)
    rows = [value for group in groups for value in group["rewards"]]
    return {
        "groups": total,
        "rollouts": len(rows),
        "mean_reward": statistics.mean(rows) if rows else 0.0,
        "positive_reward_count": sum(value == 1.0 for value in rows),
        "negative_reward_count": sum(value == 0.0 for value in rows),
        "all_1_proportion": all_one / total if total else 0.0,
        "all_0_proportion": all_zero / total if total else 0.0,
        "mixed_reward_proportion": mixed / total if total else 0.0,
        "nonzero_reward_variance_proportion": nonzero / total if total else 0.0,
        "mean_unique_answers_per_group": statistics.mean(
            group["unique_answers"] for group in groups
        )
        if groups
        else 0.0,
        "normalized_exact_duplicate_rate": statistics.mean(
            group["duplicate_rate"] for group in groups
        )
        if groups
        else 0.0,
        "mean_unique_path_signatures_per_group": statistics.mean(
            group["unique_path_signatures"] for group in groups
            if group["path_signature_available"]
        )
        if any(group["path_signature_available"] for group in groups)
        else None,
    }


def collect_groups(root: Path, source_map: dict[int, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paths = sorted(root.glob("train/step_*/idx_*/rollout_*.json"))
    rows = []
    by_idx: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        row = load_json(path)
        idx = int(row["idx"])
        reward = float(row["reward"])
        if reward not in (0.0, 1.0):
            raise ValueError(f"non-binary reward in {path}")
        item = {
            "path": str(path.relative_to(root)),
            "idx": idx,
            "source": source_map[idx]["source"],
            "reward": reward,
            "answer_norm": normalize_answer(row.get("answer_extracted", "")),
            "path_signature": extract_tool_signature(row.get("total_result")),
        }
        rows.append(item)
        by_idx[idx].append(item)

    groups = []
    for idx, members in sorted(by_idx.items()):
        members.sort(key=lambda item: item["path"])
        rewards = [item["reward"] for item in members]
        paths_for_group = [item["path_signature"] for item in members]
        path_available = all(value is not None for value in paths_for_group)
        adv = theoretical_advantages(rewards)
        groups.append(
            {
                "idx": idx,
                "dataset_id": source_map[idx]["dataset_id"],
                "source": source_map[idx]["source"],
                "n": len(members),
                "rewards": rewards,
                "class": class_for(rewards),
                "reward_std_torch_unbiased": sample_std(rewards),
                "theoretical_advantages": adv,
                "nonzero_advantage": any(abs(value) > 1e-9 for value in adv),
                "unique_answers": len({item["answer_norm"] for item in members}),
                "duplicate_rate": 1.0 - len({item["answer_norm"] for item in members}) / len(members),
                "path_signature_available": path_available,
                "unique_path_signatures": len(set(paths_for_group)) if path_available else 0,
            }
        )
    return groups, rows


def main() -> None:
    args = parse_args()
    source_map = load_source_map(args.data)
    selected_sha = hashlib.sha256(args.selected_data.read_bytes()).hexdigest()
    conditions: dict[str, Any] = {}
    all_groups: dict[str, list[dict[str, Any]]] = {}
    for raw in args.condition:
        name, root_text, log_text = raw.split("|", 2)
        root = Path(root_text)
        log = Path(log_text)
        groups, rows = collect_groups(root, source_map)
        all_groups[name] = groups
        conditions[name] = {
            "root": str(root),
            "log": str(log),
            "rollout_n": len(groups[0]["rewards"]) if groups else None,
            "expected_rollouts": 10 * len(groups[0]["rewards"]) if groups else 0,
            "summary": summarize(groups, rows, 10 * len(groups[0]["rewards"]) if groups else 0, log),
            "groups": groups,
        }

    output = {
        "audit": {
            "script": "scripts/audit_rollout_diversity_sweep_20260826.py",
            "data_source": str(args.data),
            "selected_prompt_data": str(args.selected_data),
            "selected_prompt_data_sha256": selected_sha,
            "selected_prompt_count": 10,
            "selected_prompt_source_counts": {"nq": 5, "mathhard": 5},
            "same_prompt_file_for_all_conditions": True,
            "llm_similarity_used": False,
            "deterministic_metrics": [
                "binary reward vectors",
                "population reward variance and torch-unbiased std",
                "normalized exact answer duplicate rate",
                "structural tool-result/path signature",
                "theoretical GRPO advantage from production torch.std rule",
            ],
        },
        "conditions": conditions,
        "conclusion_inputs": {
            "all_conditions_have_no_retry_report": all(
                condition["summary"]["progress_last"] is None
                or condition["summary"]["progress_last"]["retries"] == 0
                for condition in conditions.values()
            ),
            "all_conditions_have_no_update_marker": all(
                not condition["summary"]["unexpected_training_update_marker"]
                for condition in conditions.values()
            ),
            "all_conditions_have_no_reward_error_event": all(
                condition["summary"]["hybrid_reward_error_event_count_in_log"] == 0
                for condition in conditions.values()
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    for name, condition in conditions.items():
        print(name, json.dumps(condition["summary"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
