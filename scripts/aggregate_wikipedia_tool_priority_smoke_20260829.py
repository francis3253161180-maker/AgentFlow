#!/usr/bin/env python3
"""Summarize the bounded Wikipedia tool-priority smoke without judging content."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def parse_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {"raw": value}


def compact(value: Any, limit: int = 600) -> Any:
    if isinstance(value, dict):
        return {str(k): compact(v, limit) for k, v in value.items()}
    if isinstance(value, list):
        return [compact(item, limit) for item in value]
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "…[truncated]"
    return value


def extract_evidence(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if isinstance(value, list):
        return [entry for item in value for entry in extract_evidence(item)]
    if not isinstance(value, dict):
        return []
    evidence: list[dict[str, Any]] = []
    for key, child in value.items():
        if "relevant_pages" in str(key) and isinstance(child, list):
            for page in child:
                if isinstance(page, dict):
                    evidence.append({
                        "title": page.get("title"),
                        "url": page.get("url"),
                        "excerpt": compact(page.get("abstract", ""), 400),
                    })
        evidence.extend(extract_evidence(child))
    return evidence


def search_telemetry(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if isinstance(value, list):
        return [entry for child in value for entry in search_telemetry(child)]
    if not isinstance(value, dict):
        return []
    found = [value["search_telemetry"]] if isinstance(value.get("search_telemetry"), dict) else []
    for child in value.values():
        found.extend(search_telemetry(child))
    return found


def trajectory(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    total = raw.get("total_result", {})
    memory = total.get("memory", {}) if isinstance(total, dict) else {}
    steps = []
    for ordinal, (_, action) in enumerate(memory.items(), start=1):
        action = action if isinstance(action, dict) else {}
        planner = parse_json(total.get(f"action_predictor_{ordinal}_response"))
        verifier = parse_json(total.get(f"verifier_{ordinal}_response"))
        result = action.get("result")
        steps.append({
            "step": ordinal,
            "planner_tool_choice": planner.get("tool_name") if isinstance(planner, dict) else None,
            "planner_context": compact(planner.get("context")) if isinstance(planner, dict) else None,
            "planner_subgoal": compact(planner.get("sub_goal")) if isinstance(planner, dict) else None,
            "executor_command": compact(action.get("command")),
            "tool_result": compact(result),
            "retrieved_evidence": extract_evidence(result),
            "search_internal_telemetry": search_telemetry(result),
            "verifier": compact(verifier),
        })
    tool_sequence = [step["planner_tool_choice"] for step in steps if step["planner_tool_choice"]]
    return {
        "rollout_file": str(path),
        "reward": raw.get("reward"),
        "ground_truth": raw.get("groundtruth"),
        "final_answer": raw.get("answer_extracted"),
        "final_output": compact(total.get("direct_output") if isinstance(total, dict) else None),
        "steps": steps,
        "tool_sequence": tool_sequence,
        "distinct_tools": sorted(set(tool_sequence)),
        "used_factual_retrieval": "Wikipedia_RAG_Search_Tool" in tool_sequence or "Web_RAG_Search_Tool" in tool_sequence,
        "verifier_stop_signals": [
            step["verifier"].get("stop_signal")
            for step in steps
            if isinstance(step["verifier"], dict) and "stop_signal" in step["verifier"]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-log", type=Path)
    parser.add_argument("--train-log", type=Path)
    parser.add_argument("--rollout-log", type=Path)
    args = parser.parse_args()
    files = sorted(args.rollout_dir.rglob("*.json"))
    entries = [trajectory(path) for path in files]
    if len(entries) != 4:
        raise SystemExit(f"expected exactly four rollout files, found {len(entries)}")
    rewards = [float(entry["reward"]) for entry in entries]
    telemetry = [item for entry in entries for step in entry["steps"] for item in step["search_internal_telemetry"]]
    gpu_peak = None
    if args.gpu_log and args.gpu_log.exists():
        values = []
        for line in args.gpu_log.read_text(encoding="utf-8").splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) >= 3:
                try:
                    values.append(float(fields[2]))
                except ValueError:
                    pass
        gpu_peak = max(values) if values else None
    log_lines = [
        line
        for path in (args.train_log, args.rollout_log)
        if path and path.exists()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
    ]
    cleanup_markers = [
        line[:1000] for line in log_lines
        if any(marker in line.lower() for marker in ("vllm_cleanup", "drain", "sleep"))
    ]
    fatal_lifecycle_markers = [
        line[:1000] for line in log_lines
        if any(marker in line.lower() for marker in (
            "illegal memory access", "blocks are not freed", "outofmemoryerror", "deadlock",
        ))
    ]
    result = {
        "schema_version": 1,
        "purpose": "one-question x4 rollout-only structural tool-priority smoke; not a training result",
        "input_manifest": json.loads(args.input_manifest.read_text(encoding="utf-8")),
        "rollout_dir": str(args.rollout_dir),
        "rollout_count": len(entries),
        "reward_vector": rewards,
        "reward_mean": mean(rewards),
        "mixed_group": len(set(rewards)) > 1,
        "factual_retrieval_rollout_count": sum(entry["used_factual_retrieval"] for entry in entries),
        "two_or_more_distinct_tools_rollout_count": sum(len(entry["distinct_tools"]) >= 2 for entry in entries),
        "search_internal_telemetry": telemetry,
        "search_internal_doubao_calls": sum(int(item.get("doubao_calls", 0)) for item in telemetry),
        "search_internal_openai_calls": sum(int(item.get("openai_calls", 0)) for item in telemetry),
        "gpu_peak_memory_mib": gpu_peak,
        "cleanup_markers": cleanup_markers,
        "fatal_lifecycle_markers": fatal_lifecycle_markers,
        "trajectories": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("rollout_count", "reward_vector", "factual_retrieval_rollout_count", "two_or_more_distinct_tools_rollout_count", "search_internal_doubao_calls", "search_internal_openai_calls", "gpu_peak_memory_mib", "fatal_lifecycle_markers")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
