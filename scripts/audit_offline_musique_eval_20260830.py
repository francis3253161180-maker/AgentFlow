#!/usr/bin/env python3
"""Audit one frozen PRE or POST MuSiQue evaluation pack post hoc."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentflow.offline_musique import OfflineCorpus, sha256_file, terminal_reward
from agentflow.offline_musique_transition_diagnostics import annotate_trajectory, summarize_diagnostics


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def gpu_memory_mib() -> int | None:
    try:
        value = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
        ).splitlines()[0]
        return int(value.strip())
    except (FileNotFoundError, IndexError, ValueError, subprocess.CalledProcessError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("PRE", "POST"), required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--detail", type=Path, required=True)
    parser.add_argument("--runner-summary", type=Path, required=True)
    parser.add_argument("--frozen-sets", type=Path, required=True)
    parser.add_argument("--frozen-key", default="dev_eval")
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()

    corpus = OfflineCorpus.load(args.corpus)
    detail = read_json(args.detail)
    runner = read_json(args.runner_summary)
    frozen = read_json(args.frozen_sets)[args.frozen_key]
    manifest = read_json(args.artifact_manifest)
    qids = frozen["qids"]
    n = int(detail["configuration"]["n"])
    if n != 8 or detail["qids"] != qids:
        raise SystemExit("raw pack does not match frozen ordered qids at n=8")
    if len(detail["trajectories"]) != len(qids) * n:
        raise SystemExit("raw pack does not preserve qid->8 grouping cardinality")
    if detail["configuration"] != runner["configuration"]:
        raise SystemExit("runner/raw configuration mismatch")
    if detail["configuration"].get("terminal_reward", "").split(":", 1)[0] != "outcome_v2_exact_set":
        raise SystemExit("evaluation did not declare exact-set outcome v2")
    for key in ("decision_system_sha256", "evidence_system_sha256"):
        if detail["configuration"]["protocol_hashes"][key] != manifest["prompt_protocol"][key]:
            raise SystemExit(f"frozen prompt hash mismatch: {key}")

    forbidden = (
        "support_pids", "answer_aliases", "question_decomposition", "paragraph_support_idx",
        "delta_F1", "delta_F2", "new_gold_support_count", "exact_support_set",
    )
    actor_violations = []
    for prompt_hash, prompt in detail["audit_prompts"].items():
        for token in forbidden:
            if token in prompt:
                actor_violations.append([prompt_hash, token])
    if actor_violations:
        raise SystemExit(f"scorer label reached actor prompts: {actor_violations[:3]}")

    gold_by_qid = {qid: corpus.scorer_record(qid).support_pids for qid in qids}
    annotations = [
        annotate_trajectory(row, gold_by_qid[row["qid"]]) for row in detail["trajectories"]
    ]
    diagnostics = summarize_diagnostics(
        detail["trajectories"], annotations, gold_by_qid, qids, rollout_n=n
    )
    reward_mismatches = []
    for trajectory in detail["trajectories"]:
        recomputed = terminal_reward(
            corpus, trajectory["qid"], trajectory["final_answer"], trajectory["selected_pids"]
        )
        if int(recomputed["reward"]) != int(trajectory["reward_detail"]["reward"]):
            reward_mismatches.append(trajectory["trajectory_id"])
    if reward_mismatches:
        raise SystemExit(f"exact-set reward mismatch: {reward_mismatches[:3]}")

    final_rows = [row["final_support_scores"] for row in annotations]
    support_metrics = {
        key: statistics.fmean(row[key] for row in final_rows)
        for key in ("precision", "recall", "F1", "F2")
    }
    format_failures = sum(
        row["termination_reason"].startswith("format_failure") for row in detail["trajectories"]
    )
    raw_pack = dict(detail)
    raw_pack["schema_version"] = 2
    raw_pack["scorer_side_transition_diagnostics"] = {
        "training_weight": 0,
        "actor_visible": False,
        "computed_post_generation": True,
        "annotations": annotations,
    }
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_output.write_text(json.dumps(raw_pack, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    metrics = runner["metrics"]
    summary = {
        "schema_version": 1,
        "stage": args.stage,
        "generation_parent_commit": runner["parent_commit"],
        "seed": detail["configuration"]["seed"],
        "qid_count": len(qids),
        "rollout_n": n,
        "configuration": detail["configuration"],
        "freeze_checks": {
            "ordered_qids_equal_manifest": True,
            "qid_grouping_exact": True,
            "prompt_hashes_frozen": True,
            "reward_v2_recomputed_exact": True,
            "actor_prompt_scorer_label_violations": 0,
            "transition_diagnostic_training_weight": 0,
        },
        "execution": {
            "completed_rollouts": len(detail["trajectories"]),
            "format_valid_rollouts": len(detail["trajectories"]) - format_failures,
            "dropped_format_failure_rollouts": format_failures,
            "format_validity_rates": metrics["schema"],
            "provenance_invalid_count": metrics["invalid_quote_or_pid_count"],
            "provenance_invalid_rate": metrics["invalid_quote_or_pid_rate"],
            "duplicate_selection_count": metrics["duplicate_selection_count"],
            "duplicate_selection_rate": metrics["duplicate_selection_rate"],
            "generated_tokens": metrics["mean_generated_tokens"],
            "wall_seconds": metrics["wall_seconds"],
            "rollouts_per_minute": metrics["rollouts_per_minute"],
            "gpu_peak_memory_mib": metrics["gpu_peak_memory_mib"],
            "gpu_final_memory_mib": gpu_memory_mib(),
            "cleanup_errors": runner["cleanup_errors"],
        },
        "outcome": diagnostics["outcome"],
        "support": {
            "selected_support_precision_mean": support_metrics["precision"],
            "selected_support_recall_mean": support_metrics["recall"],
            "selected_support_F1_mean": support_metrics["F1"],
            "selected_support_F2_mean": support_metrics["F2"],
            "exact_set_count": diagnostics["final_support_score_diversity"]["exact_support_set_trajectory_count"],
            "exact_set_rate": diagnostics["final_support_score_diversity"]["exact_support_set_trajectory_rate"],
            "retrieval_support_recall_mean": metrics["gold_support_retrieval_recall"],
            "answer_em": metrics["answer_em"],
            "final_score_diversity": diagnostics["final_support_score_diversity"],
        },
        "transition_signal": diagnostics["question_level_transition_signal_availability"],
        "behavior_failures": {
            **diagnostics["failure_taxonomy"],
            "distractor_selection_count_rate": metrics["distractor_selection_rate"],
            "repeated_query_rate": metrics["repeated_query_rate"],
            "premature_answer_conditional_rate": metrics["premature_answer_rate"],
        },
        "artifacts": {
            "source_raw_path": str(args.detail.resolve()),
            "source_raw_sha256": sha256_file(args.detail),
            "enriched_raw_path": str(args.raw_output.resolve()),
            "enriched_raw_sha256": sha256_file(args.raw_output),
            "runner_summary_path": str(args.runner_summary.resolve()),
            "runner_summary_sha256": sha256_file(args.runner_summary),
            "corpus_path": str(args.corpus.resolve()),
            "corpus_sha256": sha256_file(args.corpus),
            "frozen_sets_path": str(args.frozen_sets.resolve()),
            "frozen_sets_sha256": sha256_file(args.frozen_sets),
        },
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
