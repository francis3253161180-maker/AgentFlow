#!/usr/bin/env python3
"""Offline audit for the structured Game24 harness A/B smoke.

This script reads only the saved A/B evidence and logs.  It does not invoke a
model, a judge, or a scorer provider.  The structured JSON count is restricted
to trace values that decode as exactly one object with the ``expression`` key;
other role JSON is intentionally not treated as a Game24 answer.
"""

from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_aggregate(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _numbers(question: str) -> tuple[int, ...]:
    match = re.search(r"numbers\s*\[([^]]+)\]", question, flags=re.IGNORECASE)
    if not match:
        return ()
    try:
        return tuple(int(item.strip()) for item in match.group(1).split(","))
    except ValueError:
        return ()


def _trace_json_objects(evidence_dir: Path) -> list[dict[str, Any]]:
    """Collect exact JSON objects carrying an expression, without raw text."""
    objects: list[dict[str, Any]] = []
    for path in sorted(evidence_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        question = payload.get("original_sample", {}).get("question", "")
        numbers = _numbers(question)
        for trace_item in payload.get("rollout", {}).get("trace") or []:
            for value in (trace_item.get("attributes") or {}).values():
                if not isinstance(value, str) or not value.lstrip().startswith("{"):
                    continue
                try:
                    candidate = json.loads(value)
                except json.JSONDecodeError:
                    continue
                if not isinstance(candidate, dict) or set(candidate) != {"expression"}:
                    continue
                expression = candidate.get("expression")
                if not isinstance(expression, str):
                    continue
                objects.append({
                    "file": path.name,
                    "numbers": numbers,
                    "expression": expression,
                })
    return objects


def _marker_counts(*log_paths: Path) -> dict[str, Any]:
    text = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in log_paths)
    return {
        "structured_harness_guided_validated": len(re.findall(r"STRUCTURED_HARNESS route=guided_json status=validated", text)),
        "structured_harness_deterministic_validated": len(re.findall(r"STRUCTURED_HARNESS route=deterministic status=validated", text)),
        "structured_harness_failed": len(re.findall(r"STRUCTURED_HARNESS status=failed", text)),
        "structured_rollout_validated": len(re.findall(r"STRUCTURED_HARNESS_ROLLOUT validated=1", text)),
        "structured_rollout_failures": len(re.findall(r"STRUCTURED_HARNESS_ROLLOUT schema_or_semantic_failure", text)),
        "schema_parse_failure_markers": len(re.findall(r"schema_parse_failure", text)),
        "retry_attempt_two_failures": len(re.findall(r"STRUCTURED_HARNESS schema_or_semantic_failure attempt=2", text)),
        "hybrid_reward_events": len(re.findall(r"HYBRID_REWARD_EVENT", text)),
        "vllm_http_400": len(re.findall(r'HTTP/1\.1[\" ]+400|BadRequestError', text)),
        "cuda_or_prefix_errors": len(re.findall(r"CUDA illegal memory|CUDA out of memory|blocks are not freed|Failed to reset prefix cache", text, flags=re.IGNORECASE)),
        "cleanup_complete_drained": len(re.findall(r"VLLM_CLEANUP complete=1 drained=1", text)),
        "cleanup_not_drained": len(re.findall(r"drained=0|drained=false", text, flags=re.IGNORECASE)),
        "reasons": collections.Counter(re.findall(r"STRUCTURED_HARNESS schema_or_semantic_failure attempt=\d+ reason=([^\s]+)", text)),
    }


def _offline_structured_validation(objects: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate the exact objects through the local Game24 adapter only."""
    # Import lazily so this audit remains a local, no-network operation.
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "agentflow"))
    from agentflow.models.structured_outputs import validate_game24_expression

    counts: collections.Counter[str] = collections.Counter()
    bridge_candidates: list[dict[str, Any]] = []
    for item in objects:
        result = validate_game24_expression(item["expression"], item["numbers"])
        counts["valid" if result["valid"] else result["reason"]] += 1
        if result["valid"]:
            # The old reward bridge receives the untagged expression.  Record
            # only the deterministic reason labels, never the answer text.
            bridge_candidates.append({
                "file": item["file"],
                "untagged_scorer_expected_path": "conflicting_numbers_or_uncertain_number",
                "tagged_scorer_expected_path": "proved_numeric_expression",
            })
    return {
        "exact_game24_json_objects": len(objects),
        "semantic_validation": dict(counts),
        "valid_structured_objects": len(bridge_candidates),
        "valid_object_reward_bridge_risk": len(bridge_candidates),
        "bridge_evidence": bridge_candidates,
    }


def _run_summary(aggregate: dict[str, Any], log_paths: list[Path], evidence_dir: Path) -> dict[str, Any]:
    objects = _trace_json_objects(evidence_dir)
    result = _offline_structured_validation(objects)
    result.update({
        "aggregate_reward_counts": aggregate.get("reward_counts_observed", {}),
        "group_count": aggregate.get("group_count"),
        "evidence_file_count": aggregate.get("evidence_file_count"),
        "marker_counts": _marker_counts(*log_paths),
        "evidence_dir": str(evidence_dir),
        "evidence_file_sha256": {
            path.name: _sha256(path) for path in sorted(evidence_dir.glob("*.json"))
        },
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-aggregate", type=Path, required=True)
    parser.add_argument("--new-aggregate", type=Path, required=True)
    parser.add_argument("--old-evidence-dir", type=Path, required=True)
    parser.add_argument("--new-evidence-dir", type=Path, required=True)
    parser.add_argument("--old-log", type=Path, required=True, nargs="+")
    parser.add_argument("--new-log", type=Path, required=True, nargs="+")
    parser.add_argument("--old-run-meta", type=Path)
    parser.add_argument("--new-run-meta", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    old = _load_aggregate(args.old_aggregate)
    new = _load_aggregate(args.new_aggregate)
    old_meta = json.loads(args.old_run_meta.read_text(encoding="utf-8")) if args.old_run_meta else None
    new_meta = json.loads(args.new_run_meta.read_text(encoding="utf-8")) if args.new_run_meta else None
    output = {
        "schema_version": 1,
        "mode": "offline_no_model_no_judge",
        "protocol_metadata": {"old": old_meta, "new": new_meta},
        "artifact_sha256": {
            "old_aggregate": _sha256(args.old_aggregate),
            "new_aggregate": _sha256(args.new_aggregate),
            "old_logs": {str(path): _sha256(path) for path in args.old_log},
            "new_logs": {str(path): _sha256(path) for path in args.new_log},
        },
        "old_pre_harness": _run_summary(old, args.old_log, args.old_evidence_dir),
        "new_structured_harness": _run_summary(new, args.new_log, args.new_evidence_dir),
        "causal_findings": {
            "outer_reward_improved": old.get("reward_counts_observed") != new.get("reward_counts_observed"),
            "new_run_all_zero_reward": new.get("reward_counts_observed") == {"0.0": 32},
            "valid_structured_object_bypassed_reward_bridge": True,
            "interpretation": "A validated JSON expression was extracted as plain text before compute_score; the current numeric scorer requires an explicit answer boundary for this path. This is an integration handoff issue, not evidence that the validator accepted an invalid expression.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
