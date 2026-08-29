#!/usr/bin/env python3
"""Preserve a compact, evidence-only comparison of old positives and gated runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def compact(value: Any, limit: int = 700) -> Any:
    if isinstance(value, dict):
        return {str(key): compact(child, limit) for key, child in value.items()}
    if isinstance(value, list):
        return [compact(child, limit) for child in value]
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "…[truncated]"
    return value


def positive_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("rollout_*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if float(raw.get("reward", 0)) != 1.0:
            continue
        total = raw.get("total_result", {})
        memory = total.get("memory", {})
        rows.append({
            "rollout_file": str(path),
            "reward": raw.get("reward"),
            "answer_extracted": raw.get("answer_extracted"),
            "final_output": compact(total.get("direct_output") or total.get("final_output")),
            "verifier_responses": {
                key: compact(value) for key, value in total.items() if key.startswith("verifier_")
            },
            "memory": [
                {
                    "step": key,
                    "tool": action.get("tool_name"),
                    "command": action.get("command"),
                    "result": compact(action.get("result"), 500),
                }
                for key, action in memory.items() if isinstance(action, dict)
            ],
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a22-root", type=Path, required=True)
    parser.add_argument("--2473-root", type=Path, required=True)
    parser.add_argument("--gated-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gated = json.loads(args.gated_results.read_text(encoding="utf-8"))
    payload = {
        "schema_version": 1,
        "purpose": "offline evidence comparison; no semantic reward recomputation",
        "historical_reward_positive_rollouts": {
            "a22d286": positive_rows(args.a22_root),
            "2473baa": positive_rows(args.__dict__["2473_root"]),
        },
        "latest_evidence_gated_summary": {
            key: gated.get(key) for key in (
                "reward_vector", "reward_mean", "termination_cause_counts",
                "unsupported_final_claim_rollout_count", "search_telemetry_totals",
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
