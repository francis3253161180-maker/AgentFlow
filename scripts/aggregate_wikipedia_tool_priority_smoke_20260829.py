#!/usr/bin/env python3
"""Summarize the bounded Wikipedia tool-priority smoke without judging content."""

from __future__ import annotations

import argparse
import json
import re
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
        if key == "evidence_chunks" and isinstance(child, list):
            for chunk in child:
                if isinstance(chunk, dict):
                    evidence.append({
                        "title": None,
                        "url": value.get("url"),
                        "excerpt": compact(chunk.get("excerpt", ""), 400),
                        "chunk_index": chunk.get("chunk_index"),
                        "lexical_score": chunk.get("lexical_score"),
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
    found = [
        value[key] for key in ("search_telemetry", "web_search_telemetry")
        if isinstance(value.get(key), dict)
    ]
    for child in value.values():
        found.extend(search_telemetry(child))
    return found


def verifier_stop_signal(value: Any) -> bool | None:
    parsed = parse_json(value)
    if isinstance(parsed, dict) and isinstance(parsed.get("stop_signal"), bool):
        return parsed["stop_signal"]
    return None


def classify_termination(
    total: dict[str, Any], steps: list[dict[str, Any]], max_steps: int, max_time: float,
) -> tuple[str, str]:
    """Classify from persisted solver state without changing production flow.

    Solver 0.5 does not serialize an explicit termination enum.  The ordering
    mirrors its loop: verifier STOP, then loop bounds on a subsequent test.
    Time classification uses the persisted whole-query execution time.
    """
    persisted = total.get("termination_reason")
    if isinstance(persisted, str) and persisted:
        return persisted, "persisted by solver"
    if steps and verifier_stop_signal(total.get(f"verifier_{len(steps)}_response")) is True:
        return "verifier_stop", "last verifier stop_signal=true"
    execution_time = total.get("execution_time")
    if isinstance(execution_time, (int, float)) and execution_time >= max_time:
        return "max_time", f"execution_time={execution_time} >= configured max_time={max_time}"
    if len(steps) >= max_steps:
        return "max_steps", f"step_count={len(steps)} >= configured max_steps={max_steps}"
    return "unknown_or_other", "no persisted verifier-stop or exhausted configured loop bound"


def extract_urls(value: Any) -> list[str]:
    return sorted(set(re.findall(r"https?://[^\s'\"<>)}\]]+", str(value))))


def active_plan_step(plan: Any) -> dict[str, Any] | None:
    parsed = parse_json(plan)
    if not isinstance(parsed, dict):
        return None
    steps = parsed.get("steps")
    if not isinstance(steps, list):
        return None
    return next((step for step in steps if isinstance(step, dict) and step.get("status") == "in_progress"), None)


def trajectory(path: Path, max_steps: int, max_time: float) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    total = raw.get("total_result", {})
    memory = total.get("memory", {}) if isinstance(total, dict) else {}
    steps = []
    for ordinal, (_, action) in enumerate(memory.items(), start=1):
        action = action if isinstance(action, dict) else {}
        planner = parse_json(total.get(f"action_predictor_{ordinal}_response"))
        verifier = parse_json(total.get(f"verifier_{ordinal}_response"))
        step_verifier = parse_json(total.get(f"step_verifier_{ordinal}_response"))
        plan_before = total.get(f"plan_before_step_{ordinal}")
        current_plan_step = active_plan_step(plan_before)
        result = action.get("result")
        steps.append({
            "step": ordinal,
            "planner_tool_choice": planner.get("tool_name") if isinstance(planner, dict) else None,
            "planner_context": compact(planner.get("context")) if isinstance(planner, dict) else None,
            "planner_subgoal": compact(planner.get("sub_goal")) if isinstance(planner, dict) else None,
            "routing_state": compact(total.get(f"action_predictor_{ordinal}_routing_state")),
            "current_plan_step": compact(current_plan_step),
            "unresolved_evidence_gaps_before_action": (
                compact(current_plan_step.get("missing_evidence", [])) if current_plan_step else []
            ),
            "executor_command": compact(action.get("command")),
            "tool_result": compact(result),
            "retrieved_evidence": extract_evidence(result),
            "search_internal_telemetry": search_telemetry(result),
            "verifier": compact(verifier),
            "step_verifier": compact(step_verifier),
        })
    known_urls: list[str] = []
    for step in steps:
        step["known_urls_before_step"] = list(known_urls)
        known_urls.extend(extract_urls(step["tool_result"]))
        known_urls = sorted(set(known_urls))
    tool_sequence = [step["planner_tool_choice"] for step in steps if step["planner_tool_choice"]]
    termination_cause, termination_evidence = classify_termination(total, steps, max_steps, max_time)
    unresolved = [
        step for step in (total.get("high_level_plan", {}) or {}).get("steps", [])
        if isinstance(step, dict) and step.get("status") != "completed"
    ]
    final_answer = raw.get("answer_extracted")
    unsupported_final_claim = bool(
        unresolved and final_answer and not str(final_answer).startswith("Insufficient verified evidence;")
    )
    continue_followed_by_action = sum(
        1
        for ordinal, step in enumerate(steps, start=1)
        if verifier_stop_signal(total.get(f"verifier_{ordinal}_response")) is False and ordinal < len(steps)
    )
    return {
        "rollout_file": str(path),
        "reward": raw.get("reward"),
        "ground_truth": raw.get("groundtruth"),
        "final_answer": final_answer,
        "final_output": compact(total.get("direct_output") if isinstance(total, dict) else None),
        "termination_reason": total.get("termination_reason"),
        "high_level_plan": compact(total.get("high_level_plan"), 1800),
        "plan_transitions": compact(total.get("plan_transitions"), 1800),
        "execution_time_seconds": total.get("execution_time"),
        "configured_max_steps": max_steps,
        "configured_max_time_seconds": max_time,
        "termination_cause": termination_cause,
        "termination_cause_evidence": termination_evidence,
        "unresolved_plan_steps": compact(unresolved),
        "unsupported_final_claim": unsupported_final_claim,
        "verifier_continue_followed_by_next_planner_action_count": continue_followed_by_action,
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
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--max-time", type=float, required=True)
    args = parser.parse_args()
    files = sorted(args.rollout_dir.rglob("*.json"))
    entries = [trajectory(path, args.max_steps, args.max_time) for path in files]
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
        "purpose": "one-question x4 rollout-only structural routing smoke; not a training result",
        "input_manifest": json.loads(args.input_manifest.read_text(encoding="utf-8")),
        "rollout_dir": str(args.rollout_dir),
        "rollout_count": len(entries),
        "reward_vector": rewards,
        "reward_mean": mean(rewards),
        "mixed_group": len(set(rewards)) > 1,
        "termination_cause_counts": {
            cause: sum(entry["termination_cause"] == cause for entry in entries)
            for cause in (
                "all_plan_steps_completed", "max_steps_with_unresolved_plan", "verifier_stop",
                "max_steps", "max_time", "unknown_or_other",
            )
        },
        "verifier_continue_followed_by_next_planner_action_count": sum(
            entry["verifier_continue_followed_by_next_planner_action_count"] for entry in entries
        ),
        "factual_retrieval_rollout_count": sum(entry["used_factual_retrieval"] for entry in entries),
        "two_or_more_distinct_tools_rollout_count": sum(len(entry["distinct_tools"]) >= 2 for entry in entries),
        "unsupported_final_claim_rollout_count": sum(entry["unsupported_final_claim"] for entry in entries),
        "search_internal_telemetry": telemetry,
        "search_telemetry_totals": {
            key: sum(int(item.get(key, 0)) for item in telemetry)
            for key in ("cache_hits", "retries", "http_429", "search_internal_llm_calls", "openai_calls", "doubao_calls")
        },
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
