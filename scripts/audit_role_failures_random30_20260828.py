#!/usr/bin/env python3
"""Offline role-level audit for the persisted random30 Game24 trajectories.

This is deliberately an evidence audit, not a new evaluator.  It decodes the
saved trace locally, parses only explicit arithmetic expressions, and reports
the earliest *observable* stage at which a non-target expression appears.
It never calls a model provider and never starts a GPU process.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


AUDIT_MODULE_PATH = Path(__file__).with_name("audit_reward_audit_len2048_20260828.py")
_spec = importlib.util.spec_from_file_location("game24_audit_helpers", AUDIT_MODULE_PATH)
if _spec is None or _spec.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load {AUDIT_MODULE_PATH}")
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)


def stage_for_prompt(prompt: str) -> str:
    """Map stable AgentFlow prompt templates to a role-level stage."""
    if "Analyze the given query to determine necessary skills and tools" in prompt:
        return "planner_main"
    if "Determine the optimal next step to address the query" in prompt:
        return "planner_fixed"
    if "Generate a precise command to execute the selected tool" in prompt:
        return "executor_tool"
    if "Evaluate if the current memory is complete and accurate enough" in prompt:
        return "verifier_revision"
    if "Generate a concise final answer to the query" in prompt:
        return "final_assembly"
    if re.search(r"Using the numbers \[[^]]+\].*create an expression", prompt, re.I | re.S):
        return "final_extraction"
    return "unknown_stage"


def decode(tokenizer: Any, item: dict[str, Any], key: str) -> str:
    value = item.get(key) or {}
    ids = value.get("token_ids") or value.get(f"{key}_token_ids") or []
    return tokenizer.decode(ids, skip_special_tokens=False) if ids else ""


def compact(text: str, limit: int = 240) -> str:
    return re.sub(r"\s+", " ", text).strip()[:limit]


def classify_trajectory(tokenizer: Any, path: Path) -> dict[str, Any]:
    evidence = json.loads(path.read_text(encoding="utf-8"))
    sample = evidence["original_sample"]
    rollout = evidence["rollout"]
    puzzle = _helpers.parse_puzzle(sample["question"])
    final_answer, answer_source, final_text = _helpers.final_answer(tokenizer, evidence)
    oracle = _helpers.oracle_evaluation(final_answer, puzzle)
    stages = []
    for index, triplet in enumerate(rollout.get("triplets") or []):
        prompt = decode(tokenizer, triplet, "prompt")
        response = decode(tokenizer, triplet, "response")
        parsed = []
        for expression in _helpers.find_expression_candidates(response):
            try:
                value, used = _helpers.parse_expression(expression)
            except ValueError:
                continue
            parsed.append({
                "expression": expression[:160],
                "value": str(value),
                "numbers": list(used),
                "target_numbers": sorted(used) == sorted(puzzle),
                "target_value": str(value) == "24",
            })
        valid = [item for item in parsed if item["target_numbers"] and item["target_value"]]
        stage = stage_for_prompt(prompt)
        stages.append({
            "triplet_index": index,
            "stage": stage,
            "parsed_expression_count": len(parsed),
            "valid_target_expression_count": len(valid),
            "first_parsed": parsed[0] if parsed else None,
            "response_excerpt": compact(response),
        })

    # This attribution is intentionally narrow: it says where the first
    # parseable non-target expression was observed, not that the role caused
    # the error.  A later valid expression is reported separately.
    first_bad = next(
        (
            item
            for item in stages
            if item["parsed_expression_count"] > 0
            and item["valid_target_expression_count"] == 0
        ),
        None,
    )
    any_valid_before_final = any(item["valid_target_expression_count"] > 0 for item in stages[:-1])
    if oracle["category"] == "wrong_number_multiset":
        if first_bad is not None:
            attribution = first_bad["stage"]
            attribution_basis = "first_parseable_non_target_expression"
        else:
            attribution = "insufficient_evidence"
            attribution_basis = "no_parseable_expression_for_stage_attribution"
    elif oracle["category"] == "invalid_or_no_expression":
        if any_valid_before_final:
            attribution = "final_assembly_or_extraction"
            attribution_basis = "earlier_valid_expression_but_final_no_expression"
        elif stages and any(item["parsed_expression_count"] > 0 for item in stages):
            attribution = "final_assembly_or_extraction"
            attribution_basis = "expression_evidence_did_not_survive_final_output"
        else:
            attribution = "insufficient_evidence"
            attribution_basis = "never_observed_parseable_expression"
    else:
        attribution = "not_applicable"
        attribution_basis = "independent_oracle_valid_or_other_category"

    source_row = sample.get("extra_info", {}).get("source_row_index")
    return {
        "evidence_file": path.name,
        "evidence_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "group_id": sample.get("data_id"),
        "source_row_index": source_row,
        "puzzle": list(puzzle),
        "stored_reward": float(rollout.get("final_reward", 0.0) or 0.0),
        "answer_source": answer_source,
        "final_answer_excerpt": compact(final_answer),
        "oracle_category": oracle["category"],
        "oracle_valid": oracle["oracle_valid"],
        "response_triplet_count": len(stages),
        "first_observed_failure_stage": attribution,
        "attribution_basis": attribution_basis,
        "any_valid_expression_before_final": any_valid_before_final,
        "stages": stages,
        "final_response_sha256": hashlib.sha256(final_text.encode("utf-8")).hexdigest(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    paths = sorted(Path(args.evidence_dir).glob("rollout_*.json"))
    rows = [classify_trajectory(tokenizer, path) for path in paths]
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    category_counts = collections.Counter(row["oracle_category"] for row in rows)
    stage_counts = collections.Counter(
        row["first_observed_failure_stage"]
        for row in rows
        if row["first_observed_failure_stage"] not in {"not_applicable"}
    )
    role_expression_counts = collections.Counter()
    role_invalid_counts = collections.Counter()
    for row in rows:
        for stage in row["stages"]:
            if stage["parsed_expression_count"]:
                role_expression_counts[stage["stage"]] += 1
            if stage["parsed_expression_count"] and not stage["valid_target_expression_count"]:
                role_invalid_counts[stage["stage"]] += 1
    result = {
        "schema_version": 1,
        "status": "ok",
        "mode": "offline_local_tokenizer_and_fraction_oracle",
        "external_calls": 0,
        "evidence_dir": str(Path(args.evidence_dir).resolve()),
        "evidence_file_count": len(paths),
        "trajectory_count": len(rows),
        "group_count": len(groups),
        "group_size_counts": {str(k): v for k, v in sorted(collections.Counter(map(len, groups.values())).items())},
        "oracle_category_counts": dict(sorted(category_counts.items())),
        "first_observed_failure_stage_counts": dict(sorted(stage_counts.items())),
        "role_expression_observation_counts": dict(sorted(role_expression_counts.items())),
        "role_non_target_expression_observation_counts": dict(sorted(role_invalid_counts.items())),
        "method_limitations": [
            "Stage attribution is the first parseable non-target arithmetic evidence, not causal proof.",
            "Prompt text is excluded from expression parsing; malformed prose and untagged final output can yield insufficient_evidence.",
            "The persisted evidence contains agent_name='*'; role labels are inferred from stable prompt templates.",
        ],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "ok",
        "trajectory_count": len(rows),
        "group_count": len(groups),
        "oracle_category_counts": dict(category_counts),
        "first_observed_failure_stage_counts": dict(stage_counts),
        "output": str(args.output),
    }, ensure_ascii=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
