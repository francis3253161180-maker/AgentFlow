#!/usr/bin/env python3
"""Summarize the bounded planner-temperature causal sanity run offline.

This script reads only persisted rollout JSON and local run metadata.  It does
not instantiate a model or call any reward/judge provider.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from agentflow.models.structured_outputs import extract_final_answer, game24_reward_decision
from train.utils import compute_score


def sha256_text(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compact(value: Any, limit: int = 280) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return re.sub(r"\s+", " ", text).strip()[:limit]


def parse_action(value: Any) -> dict[str, Any]:
    raw = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    parsed: dict[str, Any]
    try:
        candidate = json.loads(raw)
        parsed = candidate if isinstance(candidate, dict) else {}
    except (TypeError, json.JSONDecodeError):
        parsed = {}
    tool = str(parsed.get("tool_name", ""))
    sub_goal = str(parsed.get("sub_goal", ""))
    context = str(parsed.get("context", ""))
    semantic = {"tool_name": tool, "sub_goal": sub_goal, "context": context}
    return {
        "response_sha256": sha256_text(raw),
        "response_excerpt": compact(raw),
        "parsed": parsed,
        "tool_name": tool,
        "sub_goal_excerpt": compact(sub_goal, 180),
        "context_sha256": sha256_text(context),
        "context_excerpt": compact(context, 180),
        "semantic_sha256": sha256_text(json.dumps(semantic, ensure_ascii=False, sort_keys=True)),
    }


def output_record(value: Any, limit: int = 260) -> dict[str, str]:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    return {"sha256": sha256_text(text), "excerpt": compact(text, limit)}


def runtime_seconds(paths: list[Path]) -> float | None:
    stamps: list[float] = []
    pattern = re.compile(r"2026-08-29 (\d\d):(\d\d):(\d\d),(\d{3})")
    import datetime

    for path in paths:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = pattern.search(line)
            if match:
                hour, minute, second, millis = map(int, match.groups())
                stamp = datetime.datetime(2026, 8, 29, hour, minute, second, millis * 1000).timestamp()
                stamps.append(stamp)
    return max(stamps) - min(stamps) if stamps else None


def group_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_group: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_group[row["group_id"]].append(row)
    groups: list[dict[str, Any]] = []
    for group_id, members in sorted(by_group.items(), key=lambda item: int(item[0])):
        if len(members) != 4:
            raise ValueError(f"group {group_id} has {len(members)} rows, expected 4")
        members.sort(key=lambda row: row["file"])
        rewards = [int(row["reward"]) for row in members]
        action_sequences = [row["planner_action"] for row in members]
        downstream = [row["downstream_signature"] for row in members]
        tool_paths = [tuple(row["tool_path_signature"]) for row in members]
        groups.append(
            {
                "group_id": group_id,
                "source_row_index": members[0]["source_row_index"],
                "puzzle": members[0]["puzzle"],
                "reward_vector": rewards,
                "successes": sum(rewards),
                "mixed": 0 < sum(rewards) < 4,
                "unique_final_answers": len({row["answer_sha256"] for row in members}),
                "unique_planner_action_exact": len({sha256_text(json.dumps([item["response_sha256"] for item in sequence], sort_keys=True)) for sequence in action_sequences}),
                "unique_planner_action_semantic": len({sha256_text(json.dumps([item["semantic_sha256"] for item in sequence], sort_keys=True)) for sequence in action_sequences}),
                "unique_tool_paths": len(set(tool_paths)),
                "unique_downstream_signatures": len(set(downstream)),
                "actor_effectively_identical": len({sha256_text(json.dumps([item["semantic_sha256"] for item in sequence], sort_keys=True)) for sequence in action_sequences}) == 1,
                "downstream_varies": len(set(downstream)) > 1,
                "rollouts": members,
            }
        )
    bins = collections.Counter(group["successes"] for group in groups)
    reward_values = [row["reward"] for row in rows]
    return {
        "group_count": len(groups),
        "rollout_count": len(rows),
        "valid_rollouts": sum(row["valid"] for row in rows),
        "reward_mean": sum(reward_values) / len(reward_values),
        "group_bins": {f"{i}/4": bins.get(i, 0) for i in range(5)},
        "mixed_group_count": sum(group["mixed"] for group in groups),
        "mixed_group_rate": sum(group["mixed"] for group in groups) / len(groups),
        "nonzero_variance_group_rate": sum(group["mixed"] for group in groups) / len(groups),
        "mean_unique_final_answers_per_group": sum(group["unique_final_answers"] for group in groups) / len(groups),
        "mean_unique_planner_action_exact_per_group": sum(group["unique_planner_action_exact"] for group in groups) / len(groups),
        "mean_unique_planner_action_semantic_per_group": sum(group["unique_planner_action_semantic"] for group in groups) / len(groups),
        "mean_unique_tool_paths_per_group": sum(group["unique_tool_paths"] for group in groups) / len(groups),
        "mean_unique_downstream_signatures_per_group": sum(group["unique_downstream_signatures"] for group in groups) / len(groups),
        "groups_with_identical_planner_actions": sum(group["actor_effectively_identical"] for group in groups),
        "identical_actor_groups_with_mixed_rewards": sum(group["actor_effectively_identical"] and group["mixed"] for group in groups),
        "groups_with_downstream_variation": sum(group["downstream_varies"] for group in groups),
        "groups": groups,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.rollout_dir.resolve()
    paths = sorted(root.glob("**/rollout_*.json"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest_rows = {str(row["source_id"]): row for row in manifest["rows"]}
    rows: list[dict[str, Any]] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        group_id = str(data.get("idx"))
        if group_id not in manifest_rows:
            raise ValueError(f"rollout group {group_id} absent from frozen manifest")
        total = data.get("total_result") or {}
        direct_output = total.get("direct_output")
        answer = extract_final_answer(direct_output) if direct_output else "None"
        decision, details = game24_reward_decision(data.get("prompt", ""), answer)
        reward = float(compute_score(data.get("prompt", ""), data.get("groundtruth", "24"), answer))
        if decision is None or reward not in {0.0, 1.0} or bool(decision) != bool(reward):
            raise ValueError(f"unexpected Game24 scorer route/result for {path}")
        action_values = [total[key] for key in sorted(total) if key.startswith("action_predictor_") and key.endswith("_response")]
        actions = [parse_action(value) for value in action_values]
        fixed_analysis = output_record(total.get("query_analysis", ""))
        tool_results = [output_record(total[key]) for key in sorted(total) if key.startswith("tool_result_")]
        verifier = [output_record(total[key]) for key in sorted(total) if key.startswith("verifier_") and key.endswith("_response")]
        downstream_signature = sha256_text(json.dumps({"analysis": fixed_analysis["sha256"], "tools": tool_results, "verifier": verifier}, sort_keys=True))
        tool_path = [item["tool_name"] for item in actions]
        row = {
            "file": str(path),
            "file_sha256": sha256_file(path),
            "group_id": group_id,
            "rollout_id": path.stem,
            "task_slot_id": str(data.get("id", "")),
            "source_row_index": manifest_rows[group_id]["source_row_index"],
            "puzzle": manifest_rows[group_id]["puzzle"],
            "answer": answer,
            "answer_sha256": sha256_text(answer),
            "reward": reward,
            "valid": True,
            "decision_reason": details.get("reason"),
            "planner_action": actions,
            "tool_path_signature": tool_path,
            "downstream_signature": downstream_signature,
            "downstream_fixed_outputs": {"planner_fixed_analysis": fixed_analysis, "executor_tool_results": tool_results, "verifier_feedback": verifier},
            "step_count": total.get("step_count"),
        }
        rows.append(row)
    if len(paths) != 12:
        raise ValueError(f"expected 12 rollout files, found {len(paths)}")
    run_meta = json.loads(args.run_meta.read_text(encoding="utf-8"))
    gpu_values = []
    for line in args.gpu_log.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) >= 3:
            try:
                gpu_values.append(float(fields[2]))
            except ValueError:
                pass
    train_log = Path(run_meta["artifacts"]["train_log"])
    rollout_log = Path(run_meta["artifacts"]["rollout_log"])
    fatal_pattern = re.compile(
        r"CUDA out of memory|illegal memory access|blocks are not freed yet|"
        r"Failed to reset prefix cache|drained[=: ]+false|RayTaskError|"
        r"deadlock|worker died|HTTP/[0-9.]+ 5[0-9][0-9]",
        re.IGNORECASE,
    )
    forbidden: list[str] = []
    cleanup_lines: list[str] = []
    for log_path in (train_log, rollout_log):
        if not log_path.exists():
            continue
        with log_path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if fatal_pattern.search(line):
                    forbidden.append(compact(line, 320))
                if "VLLM_CLEANUP" in line:
                    cleanup_lines.append(compact(line, 300))
    cleanup_text = "\n".join(cleanup_lines)
    stats = group_stats(rows)
    route_state = Path(run_meta["artifacts"]["role_route"])
    result = {
        "schema_version": 1,
        "status": "ok",
        "mode": "rollout_only_causal_sanity_no_optimizer",
        "protocol": run_meta,
        "manifest_sha256": sha256_file(args.manifest),
        "rollout_dir": str(root),
        "rollout_dir_sha256": sha256_text("\n".join(sorted(row["file_sha256"] for row in rows))),
        "resource": {"gpu_samples": len(gpu_values), "gpu_peak_mib": max(gpu_values) if gpu_values else None, "runtime_seconds_from_logs": runtime_seconds([train_log, rollout_log]), "forbidden_error_count": len(forbidden), "forbidden_errors": forbidden[:20], "cleanup_markers": cleanup_lines[-30:], "cleanup_complete_drained_true": "complete=1 drained=1" in cleanup_text, "cleanup_normal_complete_seen": "trigger=normal_complete" in cleanup_text},
        "routing_evidence": {"role_route": json.loads(route_state.read_text(encoding="utf-8")) if route_state.exists() else None, "planner_main": "qwen-actor with trainable LoRA adapter", "planner_fixed": "doubao-seed-2-0-lite-260428", "executor": "doubao-seed-2-0-lite-260428", "verifier": "doubao-seed-2-0-lite-260428", "reward": "game24_strict_deterministic", "external_reward_judge_calls": 0},
        "aggregate": stats,
        "routing_counts": {"strict_deterministic": len(rows), "external_reward_judge": 0},
        "rows": rows,
        "comparison_note": "No complete matched planner-temperature=0.7 run exists for these same three groups under the same fixed-role configuration; the prior calibration was incomplete and its historical reward bridge was invalid. Therefore this result is not a paired causal effect estimate.",
    }
    if args.bridge_results is not None:
        result["offline_bridge_audit"] = json.loads(args.bridge_results.read_text(encoding="utf-8"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-meta", type=Path, required=True)
    parser.add_argument("--gpu-log", type=Path, required=True)
    parser.add_argument("--bridge-results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "rollouts": result["aggregate"]["rollout_count"], "mixed_groups": result["aggregate"]["mixed_group_count"], "mixed_rate": result["aggregate"]["mixed_group_rate"], "reward_mean": result["aggregate"]["reward_mean"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
