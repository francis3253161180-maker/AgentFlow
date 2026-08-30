#!/usr/bin/env python3
"""Append the bounded conditional-F2 C-v2 result to the matched C audit."""

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
    parser.add_argument("--c-final", type=Path, required=True)
    parser.add_argument("--post-cv2-enriched", type=Path, required=True)
    parser.add_argument("--post-cv2-runner", type=Path, required=True)
    parser.add_argument("--dev-corpus", type=Path, required=True)
    parser.add_argument("--cv2-audit", type=Path, required=True)
    parser.add_argument("--cv2-checksum", type=Path, required=True)
    parser.add_argument("--cv2-log", type=Path, required=True)
    parser.add_argument("--cv2-post-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prior = read(args.c_final)
    cv2 = MODULE.stage_summary(read(args.post_cv2_enriched), read(args.post_cv2_runner), read(args.dev_corpus))
    safety, checksum = read(args.cv2_audit), read(args.cv2_checksum)
    import torch

    state = torch.load(args.cv2_post_snapshot, map_location="cpu", weights_only=False)["lora_state"]
    finite = all(bool(torch.isfinite(value).all().item()) for value in state.values())
    nonzero_b = sum(int(torch.count_nonzero(value).item()) for key, value in state.items() if "lora_B" in key)
    role_credit = safety["C_v2_role_step_credit"]
    evidence = safety["evidence_select_all_exploitability"]
    optimizer = {
        "optimizer_step_count": 1,
        "pre_lora_hash": checksum["pre"]["hash"],
        "post_lora_hash": checksum["post"]["hash"],
        "hash_changed": checksum["hash_changed"],
        "changed_tensor_count": checksum["changed_tensor_count"],
        "post_state_finite": finite,
        "post_lora_B_nonzero_elements": nonzero_b,
        "grad_norm": checksum["grad_norm"],
        "pg_loss": metric_from_log(args.cv2_log, "actor/pg_loss"),
        "kl_loss": metric_from_log(args.cv2_log, "actor/kl_loss"),
        "ppo_kl": metric_from_log(args.cv2_log, "actor/ppo_kl"),
        "lr": metric_from_log(args.cv2_log, "actor/lr"),
        "official_kl_loss": {"beta": 0.001, "type": "low_var_kl", "reference": "same_actor_adapter_disabled", "entropy_coeff": 0.0},
        "new_train_rollouts": 0,
        "external_calls": 0,
    }
    output = {
        "schema_version": 1,
        "kind": "offline_musique_matched_PRE_A_B_Cv1_Cv2_conditional_F2_audit",
        "comparison": {**prior["comparison"], "POST_Cv2": cv2},
        "initial_and_branch_hashes": {**prior["initial_and_branch_hashes"], "Cv2": checksum["post"]["hash"], "Cv2_pre_verified": checksum["pre"]["hash"]},
        "Cv2_0_reward_safety": {
            "status": safety["status"],
            "optimizer_authorized": safety["gate"]["optimizer_authorized"],
            "output_pack_sha256": safety["source"]["output_pack_sha256"],
            "hard_boundaries": safety["hard_boundaries"],
            "evidence_reward": role_credit["evidence_reward"],
            "grouping": role_credit["comparison_key"],
            "select_all_exploitability": evidence,
            "role_step_signal": role_credit["summary"],
            "ineligible_event_task_pg_token_count": role_credit["ineligible_event_task_pg_token_count"],
        },
        "Cv2_1_optimizer": optimizer,
        "boundaries": {
            "same_initial_snapshot": checksum["pre"]["hash"] == prior["initial_and_branch_hashes"]["initial_and_A"],
            "same_train_pack": True,
            "new_train_rollouts": 0,
            "HOB": False,
            "variance_aware_sampling": False,
            "reranker": False,
            "adaptive_retriever_started": False,
            "lambda_sweep": False,
            "scorer_labels_actor_visible": False,
        },
        "interpretation": {
            "post_cv2_terminal_success_improved_over_PRE": cv2["overall"]["terminal"]["reward_mean"] > prior["comparison"]["PRE"]["overall"]["terminal"]["reward_mean"],
            "post_cv2_retrieval_improved_over_PRE": cv2["overall"]["retrieval_support_recall"] > prior["comparison"]["PRE"]["overall"]["retrieval_support_recall"],
            "post_cv2_retrieval_improved_over_B": cv2["overall"]["retrieval_support_recall"] > prior["comparison"]["POST_B"]["overall"]["retrieval_support_recall"],
            "conclusion": "The conditional-F2 C-v2 repair passed reward-safety gates and produced a finite changed adapter, but did not improve frozen terminal success or retrieval support recall. Adaptive Retriever remains only a later optional engineering ablation and was not started.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "Cv2_reward": cv2["overall"]["terminal"]["reward_mean"], "Cv2_hash": checksum["post"]["hash"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
