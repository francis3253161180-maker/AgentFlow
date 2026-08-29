#!/usr/bin/env python3
"""Run the bounded, supervisor-only evidence-plan preflight (no actor/tool/reward)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from agentflow.engine.factory import create_llm_engine
from agentflow.models.planner import Planner, SupervisorBoundaryViolation


MODEL = "doubao-seed-2-0-lite-260428"


def payload(value):
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.getenv("ARK_REASONING_EFFORT") != "minimal":
        raise SystemExit("ARK_REASONING_EFFORT=minimal is required for this no-thinking preflight")
    if not os.getenv("ARK_API_KEY"):
        raise SystemExit("ARK_API_KEY is required but will not be printed")
    rows = pd.read_parquet(args.source)
    selected = rows.loc[rows["id"] == 4]
    if len(selected) != 1:
        raise SystemExit("expected exactly one frozen MuSiQue group-4 row")
    row = selected.iloc[0]
    extra = row["extra_info"]
    if extra.get("idx") != 259 or extra.get("benchmark_id") != "2hop__13592_49388":
        raise SystemExit("frozen sample provenance mismatch")

    planner = object.__new__(Planner)
    planner.max_tokens = 512
    planner.available_tools = []
    planner.llm_engine_fixed = None
    planner.llm_engine_supervisor = create_llm_engine(MODEL, temperature=0.0, max_tokens=512)
    planner.supervisor_call_count = 0
    planner.supervisor_boundary_audits = []
    result = {
        "schema_version": 1,
        "purpose": "supervisor-only boundary preflight; no actor action, tool, scoring, or training",
        "model": MODEL,
        "temperature": 0.0,
        "ark_reasoning_effort": "minimal",
        "source_idx": int(extra["idx"]),
        "benchmark_id": extra["benchmark_id"],
        "question_sha256": hashlib.sha256(str(row["question"]).encode()).hexdigest(),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "hierarchical_plan_max_steps": 3,
    }
    try:
        plan = planner.generate_high_level_plan(
            str(row["question"]), "",
            "Supervisor-only preflight: enumerate independent evidence requirements without action strategy.",
            3,
        )
        result.update({
            "passed": True,
            "plan": payload(plan),
            "coverage": planner.last_high_level_plan_coverage,
            "supervisor_call_count": planner.supervisor_call_count,
            "boundary_audits": planner.supervisor_boundary_audits,
            "last_request_metadata": getattr(planner.llm_engine_supervisor, "last_request_metadata", None),
        })
    except SupervisorBoundaryViolation as exc:
        result.update({
            "passed": False,
            "failure_role": exc.role,
            "failure_telemetry": exc.telemetry,
            "supervisor_call_count": planner.supervisor_call_count,
            "boundary_audits": planner.supervisor_boundary_audits,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    if not result.get("passed"):
        raise SystemExit("supervisor preflight failed safely")


if __name__ == "__main__":
    main()
