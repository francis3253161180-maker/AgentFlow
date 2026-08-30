#!/usr/bin/env python3
"""Append the strictly matched causal-split C result to the PRE/A/B audit."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path


BASE = Path(__file__).with_name("audit_offline_musique_transition_ablation_20260830.py")
SPEC = importlib.util.spec_from_file_location("transition_ablation_audit", BASE)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_from_log(path: Path, name: str) -> float | None:
    match = re.search(rf"'{re.escape(name)}': np\.float64\(([-+0-9.eE]+)\)", path.read_text(encoding="utf-8"))
    return float(match.group(1)) if match else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ab-audit", type=Path, required=True)
    parser.add_argument("--post-c-enriched", type=Path, required=True)
    parser.add_argument("--post-c-runner", type=Path, required=True)
    parser.add_argument("--dev-corpus", type=Path, required=True)
    parser.add_argument("--c0-audit", type=Path, required=True)
    parser.add_argument("--c-checksum", type=Path, required=True)
    parser.add_argument("--c-log", type=Path, required=True)
    parser.add_argument("--c-post-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = read(args.ab_audit)
    c = MODULE.stage_summary(read(args.post_c_enriched), read(args.post_c_runner), read(args.dev_corpus))
    checksum, c0 = read(args.c_checksum), read(args.c0_audit)
    import torch
    state = torch.load(args.c_post_snapshot, map_location="cpu", weights_only=False)["lora_state"]
    finite = all(bool(torch.isfinite(tensor).all().item()) for tensor in state.values())
    c0_compact = {
        "audit_path": str(args.c0_audit),
        "output_pack_sha256": c0["source"]["output_pack_sha256"],
        "hard_boundaries": c0["hard_boundaries"],
        "identity_and_preservation": {
            "exact_trajectory_transition_identity": c0["identity_and_preservation"]["exact_trajectory_transition_identity"],
            "old_logprob_response_mask_preserved": c0["identity_and_preservation"]["old_logprob_response_mask_preserved"],
        },
        "mapping": c0["mapping"],
        "normalization": {
            "finite": c0["normalization"]["finite"],
            "zero_mean_tolerance": c0["normalization"]["zero_mean_tolerance"],
            "search": {key: c0["normalization"]["search"][key] for key in ("group_count", "nondegenerate_group_count", "max_abs_group_mean", "by_hop")},
            "evidence": {key: c0["normalization"]["evidence"][key] for key in ("group_count", "nondegenerate_group_count", "max_abs_group_mean", "by_hop")},
        },
        "qid_signal_coverage": c0["qid_signal_coverage"],
        "token_credit": c0["token_credit"],
    }
    log = args.c_log
    output = {
        "schema_version": 1,
        "kind": "offline_musique_matched_PRE_A_B_C_causal_split_audit",
        "comparison": {**base["comparison"], "POST_C": c},
        "initial_and_branch_hashes": {
            "initial_and_A": base["B_optimizer"]["pre_lora_hash"],
            "B": base["B_optimizer"]["post_lora_hash"],
            "C": checksum["post"]["hash"],
            "C_pre_verified": checksum["pre"]["hash"],
        },
        "C0_causal_split": c0_compact,
        "C1_optimizer": {
            "optimizer_step_count": 1,
            "pre_lora_hash": checksum["pre"]["hash"],
            "post_lora_hash": checksum["post"]["hash"],
            "hash_changed": checksum["hash_changed"],
            "changed_tensor_count": checksum["changed_tensor_count"],
            "grad_norm": checksum["grad_norm"],
            "pg_loss": metric_from_log(log, "actor/pg_loss"),
            "kl_loss": metric_from_log(log, "actor/kl_loss"),
            "ppo_kl": metric_from_log(log, "actor/ppo_kl"),
            "lr": metric_from_log(log, "actor/lr"),
            "official_kl": {"beta": 0.001, "type": "low_var_kl", "reference": "same_actor_adapter_disabled", "entropy": 0.0},
            "post_state_finite": finite,
        },
        "boundaries": {"same_initial_snapshot": True, "same_train_pack": True, "new_train_rollouts": 0, "HOB": False, "variance_aware_sampling": False, "reranker": False, "lambda_sweep": False, "scorer_labels_actor_visible": False},
        "interpretation": {
            "preregistered_3_4_hop_retrieval_expectation_met": False,
            "terminal_outcome_improved_over_PRE": False,
            "terminal_outcome_improved_over_B": False,
            "conclusion": "This bounded C pilot does not support causal split credit as an improvement over B; no further training is authorized by this plan.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "C_reward": c["overall"]["terminal"]["reward_mean"], "C_hash": checksum["post"]["hash"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
