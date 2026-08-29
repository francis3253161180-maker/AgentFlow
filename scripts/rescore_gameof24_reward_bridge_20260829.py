#!/usr/bin/env python3
"""Offline validation of the Game24 reward bridge on persisted rollouts.

The script reads saved rollout JSON only.  It does not create a model, call a
provider, start Ray/vLLM, or write rewards back to the source artifacts.
Production scoring is invoked through ``train.utils.compute_score``.  The
intermediate-stage audit is deliberately observational: it looks for strict
legal expressions in persisted role outputs and never calls a verifier model.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path
from typing import Any

os.environ.setdefault("AGENTFLOW_DISABLE_EXTERNAL_LLM", "1")
os.environ.setdefault("AGENTFLOW_REWARD_JUDGE_ENABLED", "0")

from agentflow.models.structured_outputs import (
    candidate_expressions,
    extract_game24_numbers,
    game24_reward_decision,
    validate_game24_expression,
)
from train.utils import compute_score


_HELPER_PATH = Path(__file__).with_name("audit_reward_audit_len2048_20260828.py")
_SPEC = importlib.util.spec_from_file_location("game24_audit_helpers", _HELPER_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load {_HELPER_PATH}")
_HELPERS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HELPERS)


STAGE_EXCLUDED = {
    "query",
    "image",
    "direct_output",
    "execution_time",
    "step_count",
}


def sha256_text(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compact(value: Any, limit: int = 240) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return re.sub(r"\s+", " ", text).strip()[:limit]


def stage_name(field: str) -> str:
    if field == "query_analysis":
        return "planner_main_analysis"
    if field.startswith("action_predictor_") and field.endswith("_response"):
        return "planner_fixed"
    if field.startswith("tool_commander_") and field.endswith("_response"):
        return "executor_tool_command"
    if field.startswith("tool_result_"):
        return "executor_tool_result"
    if field.startswith("verifier_") and field.endswith("_response"):
        return "verifier_feedback"
    if field == "memory":
        return "memory_state"
    return field


def line_candidates(text: str) -> list[str]:
    """Return only expression-shaped complete lines, never arbitrary prose."""
    found: list[str] = []
    for line in text.splitlines():
        stripped = line.strip().strip("`")
        stripped = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", stripped)
        for item in (stripped, stripped.rsplit(":", 1)[-1].strip()):
            if item and any(op in item for op in "+-*/×÷"):
                found.append(item.strip(" `.!"))
        found.extend(re.findall(r"`([^`]+)`", line))
    return found


def valid_stage_expressions(text: str, numbers: tuple[int, ...]) -> list[str]:
    candidates: list[str] = []
    candidates.extend(candidate_expressions(text))
    candidates.extend(_HELPERS.find_expression_candidates(text))
    candidates.extend(line_candidates(text))
    valid: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate = candidate.strip()
        if candidate in seen:
            continue
        seen.add(candidate)
        checked = validate_game24_expression(candidate, numbers)
        if checked["valid"]:
            valid.append(candidate)
    return valid


def audit_intermediate_stages(total_result: dict[str, Any], numbers: tuple[int, ...]) -> dict[str, Any]:
    observations: list[dict[str, Any]] = []
    for field, value in total_result.items():
        if field in STAGE_EXCLUDED or field.endswith("_prompt"):
            continue
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        valid = valid_stage_expressions(text, numbers)
        if valid:
            observations.append(
                {
                    "field": field,
                    "stage": stage_name(field),
                    "valid_expression_count": len(valid),
                    "expression_excerpts": [compact(item, 160) for item in valid[:3]],
                }
            )
    return {
        "valid_intermediate_stages": observations,
        "any_valid_intermediate": bool(observations),
    }


def group_bins(rows: list[dict[str, Any]], reward_key: str) -> dict[str, int]:
    grouped: dict[str, list[float]] = collections.defaultdict(list)
    for row in rows:
        grouped[str(row["group_id"])].append(float(row[reward_key]))
    if any(len(values) != 4 for values in grouped.values()):
        raise ValueError("every prompt group must contain exactly four rollouts")
    counts = collections.Counter(sum(values) for values in grouped.values())
    return {f"{int(success)}/4": count for success, count in counts.items()}


def ordered_bins(rows: list[dict[str, Any]], reward_key: str) -> dict[str, int]:
    raw = group_bins(rows, reward_key)
    return {f"{index}/4": int(raw.get(f"{index}/4", 0)) for index in range(5)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.rollout_dir.resolve()
    paths = sorted(root.glob("**/rollout_*.json"))
    rows: list[dict[str, Any]] = []
    stage_wrong_rows: list[dict[str, Any]] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        question = str(data.get("prompt", ""))
        answer = str(data.get("answer_extracted", ""))
        groundtruth = str(data.get("groundtruth", ""))
        numbers = extract_game24_numbers(question)
        decision, decision_details = game24_reward_decision(question, answer)
        production_reward = bool(compute_score(question, groundtruth, answer))
        if decision is not None and production_reward != bool(decision):
            raise AssertionError(f"compute_score disagrees with Game24 decision for {path}")
        stage_audit = audit_intermediate_stages(data.get("total_result") or {}, numbers or ())
        row = {
            "file": path.name,
            "file_sha256": sha256_file(path),
            "group_id": str(data.get("idx", "")),
            "rollout_id": str(data.get("id", "")),
            "question_sha256": sha256_text(question),
            "groundtruth": groundtruth,
            "answer_sha256": sha256_text(answer),
            "answer_excerpt": compact(answer),
            "stored_reward": float(data.get("reward", 0.0) or 0.0),
            "production_reward": 1.0 if production_reward else 0.0,
            "route": "game24_strict_deterministic" if decision is not None else "generic_scorer",
            "decision_reason": decision_details.get("reason"),
            "stage_observation": stage_audit,
        }
        rows.append(row)
        if stage_audit["any_valid_intermediate"] and not production_reward:
            stage_wrong_rows.append(
                {
                    "group_id": row["group_id"],
                    "rollout_id": row["rollout_id"],
                    "file": row["file"],
                    "answer_excerpt": row["answer_excerpt"],
                    "stages": stage_audit["valid_intermediate_stages"],
                    "final_decision_reason": decision_details.get("reason"),
                }
            )

    if len(rows) != 120:
        raise ValueError(f"expected 120 persisted rollouts, found {len(rows)}")
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)

    stored_success = sum(int(row["stored_reward"] == 1.0) for row in rows)
    production_success = sum(int(row["production_reward"] == 1.0) for row in rows)
    discrepancies = [
        {
            "file": row["file"],
            "group_id": row["group_id"],
            "stored_reward": row["stored_reward"],
            "production_reward": row["production_reward"],
            "decision_reason": row["decision_reason"],
            "answer_excerpt": row["answer_excerpt"],
        }
        for row in rows
        if row["stored_reward"] != row["production_reward"]
    ]
    route_counts = collections.Counter(row["route"] for row in rows)
    reason_counts = collections.Counter(row["decision_reason"] for row in rows)
    stage_counts = collections.Counter(
        observation["stage"]
        for row in rows
        for observation in row["stage_observation"]["valid_intermediate_stages"]
    )
    wrong_stage_counts = collections.Counter(
        observation["stage"]
        for row in stage_wrong_rows
        for observation in row["stages"]
    )

    result = {
        "schema_version": 1,
        "status": "ok",
        "mode": "offline_production_compute_score_no_external_calls",
        "external_calls": 0,
        "rollout_dir": str(root),
        "rollout_dir_sha256": hashlib.sha256("\n".join(row["file_sha256"] for row in rows).encode()).hexdigest(),
        "trajectory_count": len(rows),
        "group_count": len(groups),
        "group_size_counts": {str(size): count for size, count in collections.Counter(map(len, groups.values())).items()},
        "stored_rewards": {
            "success_count": stored_success,
            "zero_count": len(rows) - stored_success,
            "mean": stored_success / len(rows),
            "group_bins": ordered_bins(rows, "stored_reward"),
        },
        "production_rescore": {
            "success_count": production_success,
            "zero_count": len(rows) - production_success,
            "mean": production_success / len(rows),
            "group_bins": ordered_bins(rows, "production_reward"),
        },
        "discrepancies": discrepancies,
        "discrepancy_count": len(discrepancies),
        "routing": dict(sorted(route_counts.items())),
        "decision_reasons": dict(sorted(reason_counts.items())),
        "intermediate_valid_then_final_wrong": {
            "trajectory_count": len(stage_wrong_rows),
            "all_trajectory_stage_observation_counts": dict(sorted(stage_counts.items())),
            "wrong_trajectory_stage_observation_counts": dict(sorted(wrong_stage_counts.items())),
            "examples": stage_wrong_rows[:12],
            "method": "strict validator over persisted non-prompt total_result fields; direct_output excluded; observational, not causal",
        },
        "reference_comparison": {
            "user_stated_success_count": 61,
            "user_stated_bins": {"0/4": 4, "1/4": 9, "2/4": 10, "3/4": 3, "4/4": 4},
            "stated_bins_sum": 30,
            "stated_bin_implied_success_count": 54,
            "note": "Reference success count and reference bins are internally inconsistent; neither is used to override production rescore.",
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "ok",
        "trajectory_count": len(rows),
        "group_count": len(groups),
        "stored_success": stored_success,
        "production_success": production_success,
        "production_group_bins": result["production_rescore"]["group_bins"],
        "discrepancy_count": len(discrepancies),
        "intermediate_valid_then_final_wrong": len(stage_wrong_rows),
        "output": str(args.output),
    }, ensure_ascii=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
