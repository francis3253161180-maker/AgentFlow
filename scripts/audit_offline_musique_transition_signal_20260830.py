#!/usr/bin/env python3
"""Build a post-hoc scorer-only n=8 MuSiQue transition-signal audit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentflow.offline_musique import OfflineCorpus, sha256_file
from agentflow.offline_musique_transition_diagnostics import annotate_trajectory, summarize_diagnostics


FROZEN_CONFIG_KEYS = (
    "actor_base",
    "actor_lora",
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
    "subset_size",
    "seed",
    "top_k_retrieval",
    "rrf_k",
    "max_search_actions",
    "max_decision_transitions",
    "actor_context_tokens",
    "decision_max_new_tokens",
    "evidence_max_new_tokens",
    "gpu_memory_utilization",
    "external_network_or_llm_calls",
    "fixed_semantic_roles",
    "generation_constraint",
    "evidence_generation_stop",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def current_gpu_memory_mib() -> int | None:
    try:
        value = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
        ).splitlines()[0]
        return int(value.strip())
    except (FileNotFoundError, IndexError, ValueError, subprocess.CalledProcessError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--detail", type=Path, required=True)
    parser.add_argument("--runner-summary", type=Path, required=True)
    parser.add_argument("--reference-detail", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()

    corpus = OfflineCorpus.load(args.corpus)
    detail = read_json(args.detail)
    runner = read_json(args.runner_summary)
    reference_detail = read_json(args.reference_detail)
    reference = read_json(args.reference_summary)
    manifest = read_json(args.artifact_manifest)

    if detail["qids"] != reference_detail["qids"]:
        raise SystemExit("n=8 qids/order differ from the Phase-C-v4 fixed subset")
    if runner["configuration"] != detail["configuration"]:
        raise SystemExit("runner summary and raw detail configurations differ")
    if detail["configuration"]["n"] != 8 or len(detail["trajectories"]) != 32 * 8:
        raise SystemExit("diagnostic must contain exactly 32 qids x 8 rollouts")
    for key in FROZEN_CONFIG_KEYS:
        if detail["configuration"][key] != reference["configuration"][key]:
            raise SystemExit(f"frozen Phase-C-v4 configuration changed: {key}")
    prompt_protocol = detail["configuration"].get("protocol_hashes", {})
    for key in ("decision_system_sha256", "evidence_system_sha256"):
        if prompt_protocol.get(key) != manifest["prompt_protocol"][key]:
            raise SystemExit(f"passing prompt hash mismatch: {key}")

    gold_by_qid = {qid: corpus.scorer_record(qid).support_pids for qid in detail["qids"]}
    annotations = [
        annotate_trajectory(row, gold_by_qid[row["qid"]]) for row in detail["trajectories"]
    ]
    diagnostic = summarize_diagnostics(
        detail["trajectories"], annotations, gold_by_qid, detail["qids"], rollout_n=8
    )
    outcome = diagnostic["outcome"]
    support = diagnostic["final_support_score_diversity"]
    baseline_mixed_rate = reference["metrics"]["mixed_reward_groups"] / 32
    diagnostic["same_question_comparison"] = {
        "phase_c_v4_n4_terminal_mixed_group_rate": baseline_mixed_rate,
        "n8_terminal_mixed_group_rate": outcome["mixed_group_rate"],
        "n8_minus_n4_terminal_mixed_group_rate": outcome["mixed_group_rate"] - baseline_mixed_rate,
        "final_F1_mixed_minus_terminal_mixed_rate": support["final_F1_mixed_group_rate"] - outcome["mixed_group_rate"],
        "final_F2_mixed_minus_terminal_mixed_rate": support["final_F2_mixed_group_rate"] - outcome["mixed_group_rate"],
        "final_F1_mixed_rate_multiple_of_terminal": (
            support["final_F1_mixed_group_rate"] / outcome["mixed_group_rate"]
            if outcome["mixed_group_rate"]
            else None
        ),
        "final_F2_mixed_rate_multiple_of_terminal": (
            support["final_F2_mixed_group_rate"] / outcome["mixed_group_rate"]
            if outcome["mixed_group_rate"]
            else None
        ),
    }

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

    base_metrics = runner["metrics"]
    format_failures = sum(
        row["termination_reason"].startswith("format_failure") for row in detail["trajectories"]
    )
    summary = {
        "schema_version": 1,
        "experiment": "offline_musique_n8_outcome_transition_reward_diagnostic",
        "diagnostic_only": True,
        "training_weight": 0,
        "training_occurred": False,
        "grpo_occurred": False,
        "hob_occurred": False,
        "parent_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "seed": detail["configuration"]["seed"],
        "configuration": detail["configuration"],
        "freeze_checks": {
            "fixed_qids_and_order_equal_phase_c_v4": True,
            "frozen_configuration_fields_equal_phase_c_v4": True,
            "passing_prompt_hashes_equal_manifest": True,
            "terminal_reward_unchanged": True,
            "scorer_diagnostics_computed_only_after_generation": True,
            "scorer_diagnostics_actor_visible": False,
            "scorer_diagnostics_training_weight": 0,
        },
        "execution": {
            "completed_rollouts": len(detail["trajectories"]),
            "format_valid_rollouts": len(detail["trajectories"]) - format_failures,
            "dropped_format_failure_rollouts": format_failures,
            "answer_terminated_rollouts": runner["metrics"]["termination_reasons"].get("answer", 0),
            "wall_seconds": base_metrics["wall_seconds"],
            "rollouts_per_minute": base_metrics["rollouts_per_minute"],
            "gpu_peak_memory_mib": base_metrics["gpu_peak_memory_mib"],
            "gpu_final_memory_mib": current_gpu_memory_mib(),
            "format_validity_rates": base_metrics["schema"],
            "overall_schema_valid_rate": base_metrics["overall_schema_valid_rate"],
            "answer_em": base_metrics["answer_em"],
            "gold_support_retrieval_recall": base_metrics["gold_support_retrieval_recall"],
            "gold_support_selection_recall": base_metrics["gold_support_selection_recall"],
            "distractor_selection_rate": base_metrics["distractor_selection_rate"],
            "repeated_query_rate": base_metrics["repeated_query_rate"],
            "premature_answer_rate": base_metrics["premature_answer_rate"],
            "grounded_positive_count": outcome["grounded_positive_count"],
            "exact_support_set_rate": support["exact_support_set_trajectory_rate"],
            "cleanup_errors": runner["cleanup_errors"],
        },
        "diagnostics": diagnostic,
        "artifacts": {
            "corpus_path": str(args.corpus.resolve()),
            "corpus_sha256": sha256_file(args.corpus),
            "source_raw_detail_path": str(args.detail.resolve()),
            "source_raw_detail_sha256": sha256_file(args.detail),
            "runner_summary_path": str(args.runner_summary.resolve()),
            "runner_summary_sha256": sha256_file(args.runner_summary),
            "enriched_raw_pack_path": str(args.raw_output.resolve()),
            "enriched_raw_pack_sha256": sha256_file(args.raw_output),
            "reference_phase_c_v4_detail_sha256": sha256_file(args.reference_detail),
            "reference_phase_c_v4_summary_sha256": sha256_file(args.reference_summary),
        },
        "training_metrics": {
            "advantages": "not recorded; no training batch or GRPO normalization was materialized",
            "pg_loss": "not recorded; no training",
            "grad_norm": "not recorded; no training",
            "entropy": "not recorded; no training",
            "old_log_prob": "not recorded; generation token logprobs exist, but no training metric was computed",
            "optimizer_step": False,
            "update_actor": False,
            "global_step": "not recorded; no training",
        },
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
