#!/usr/bin/env python3
"""Build and audit C's causal-split offline MuSiQue replay pack.

SEARCH decisions receive only scorer-side retrieval-coverage deltas.  Their
following EVIDENCE_UPDATE transitions receive only selected-support delta-F2.
All final-answer and unmapped transitions retain the (here zero) terminal term.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from agentflow.verl.unified_smoke_capture import _field_digest


FORBIDDEN_ACTOR_FIELDS = (
    "support_pids", "answer_aliases", "question_decomposition",
    "paragraph_support_idx", "delta_F1", "delta_F2", "new_gold_support_count",
    "exact_support_set",
)


def normalize_group(values: list[float], epsilon: float = 1e-6) -> list[float]:
    if len(values) < 2:
        return [0.0] * len(values)
    tensor = torch.tensor(values, dtype=torch.float32)
    if bool(torch.all(tensor == tensor[0]).item()):
        return [0.0] * len(values)
    std = tensor.std(unbiased=True)
    if not bool(torch.isfinite(std).item()) or float(std.item()) == 0.0:
        return [0.0] * len(values)
    result = (tensor - tensor.mean()) / (std + epsilon)
    if not bool(torch.isfinite(result).all().item()):
        raise ValueError("non-finite normalized local credit")
    return [float(value) for value in result.tolist()]


def hop_of(qid: str) -> str:
    for hop in ("2hop", "3hop", "4hop"):
        if qid.startswith(hop):
            return hop
    raise ValueError(f"unrecognized MuSiQue qid hop: {qid}")


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def add_group_credit(
    entries_by_group: dict[tuple[str, int], list[tuple[str, int, float]]],
    *,
    name: str,
) -> tuple[dict[tuple[str, int], float], list[dict]]:
    credit_by_transition: dict[tuple[str, int], float] = {}
    audit = []
    for (qid, ordinal), entries in sorted(entries_by_group.items()):
        normalized = normalize_group([entry[2] for entry in entries])
        for (trajectory_id, transition_index, _delta), credit in zip(entries, normalized):
            key = (trajectory_id, transition_index)
            if key in credit_by_transition:
                raise ValueError(f"duplicate {name} annotation for {key}")
            credit_by_transition[key] = credit
        audit.append({
            "qid": qid,
            "hop": hop_of(qid),
            "ordinal": ordinal,
            "valid_rollout_count": len(entries),
            "delta_values": [entry[2] for entry in entries],
            "credit_values": normalized,
            "nondegenerate": any(value != 0.0 for value in normalized),
            "credit_mean": float(np.mean(normalized)) if normalized else 0.0,
        })
    return credit_by_transition, audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enriched", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--terminal-pack", type=Path, required=True)
    parser.add_argument("--output-pack", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()

    enriched = json.loads(args.enriched.read_text(encoding="utf-8"))
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    diagnostics = enriched["scorer_side_transition_diagnostics"]
    if diagnostics.get("actor_visible") or diagnostics.get("training_weight") != 0:
        raise ValueError("source scorer diagnostics must be actor-hidden and weight-zero")
    actor_leaks = [
        (prompt_hash, field)
        for prompt_hash, prompt in enriched["audit_prompts"].items()
        for field in FORBIDDEN_ACTOR_FIELDS if field in prompt
    ]
    if actor_leaks:
        raise ValueError(f"scorer leakage in actor prompts: {actor_leaks[:3]}")

    pack = torch.load(args.terminal_pack, map_location="cpu", weights_only=False)
    tensors, non_tensor = pack["tensor_fields"], pack["non_tensor_batch"]
    mask = tensors["response_mask"].bool()
    if not bool(torch.all(tensors["advantages"].float() == 0).item()):
        raise ValueError("C must start from the all-zero terminal advantage pack")
    row_by_transition = {
        (str(trajectory_id), int(turn_index)): row
        for row, (trajectory_id, turn_index) in enumerate(
            zip(non_tensor["trajectory_id_list"], non_tensor["turn_index_list"])
        )
    }
    if len(row_by_transition) != len(mask):
        raise ValueError("duplicate or missing replay transition identity")
    trajectories = {row["trajectory_id"]: row for row in enriched["trajectories"]}
    if set(trajectories) != {key[0] for key in row_by_transition}:
        raise ValueError("persisted replay and enriched trajectory identity differ")

    search_entries: dict[tuple[str, int], list[tuple[str, int, float]]] = defaultdict(list)
    for trajectory_id, trajectory in trajectories.items():
        qid = trajectory["qid"]
        gold = set(corpus["scorer_only"][qid]["support_pids"])
        if not gold:
            raise ValueError(f"empty scorer-only gold support set: {qid}")
        retrieved: set[str] = set()
        ordinal = 0
        for transition_index, transition in enumerate(trajectory["transitions"]):
            output = transition.get("semantic_output") or {}
            if transition["mode"] != "DECISION" or output.get("action") != "search":
                continue
            refs = transition.get("observation_refs", [])
            if len(refs) != 1 or refs[0] not in enriched["audit_observations"]:
                raise ValueError(f"search observation missing for {trajectory_id}/{transition_index}")
            ordinal += 1
            before = len(retrieved & gold) / len(gold)
            observation = enriched["audit_observations"][refs[0]]
            if not isinstance(observation, list) or not all("pid" in item for item in observation):
                raise ValueError("retrieval observation is malformed")
            retrieved.update(str(item["pid"]) for item in observation)
            after = len(retrieved & gold) / len(gold)
            search_entries[(qid, ordinal)].append((trajectory_id, transition_index, after - before))

    evidence_entries: dict[tuple[str, int], list[tuple[str, int, float]]] = defaultdict(list)
    for annotation in diagnostics["annotations"]:
        trajectory_id = str(annotation["trajectory_id"])
        for score in annotation["transition_scores"]:
            evidence_entries[(str(annotation["qid"]), int(score["evidence_update_ordinal"]))].append(
                (trajectory_id, int(score["transition_index"]), float(score["delta_F2"]))
            )

    search_credit, search_audit = add_group_credit(search_entries, name="search")
    evidence_credit, evidence_audit = add_group_credit(evidence_entries, name="evidence")
    row_search = torch.zeros(len(mask), dtype=torch.float32)
    row_evidence = torch.zeros(len(mask), dtype=torch.float32)
    for (trajectory_id, transition_index), credit in search_credit.items():
        row = row_by_transition.get((trajectory_id, transition_index))
        if row is None or str(non_tensor["policy_mode_list"][row]) != "DECISION":
            raise ValueError("search credit does not map to a DECISION replay row")
        if (trajectories[trajectory_id]["transitions"][transition_index].get("semantic_output") or {}).get("action") != "search":
            raise ValueError("search credit maps to a non-search decision")
        row_search[row] = credit
    for (trajectory_id, transition_index), credit in evidence_credit.items():
        row = row_by_transition.get((trajectory_id, transition_index))
        if row is None or str(non_tensor["policy_mode_list"][row]) != "EVIDENCE_UPDATE":
            raise ValueError("evidence credit does not map to an EVIDENCE_UPDATE replay row")
        row_evidence[row] = credit
    if bool(torch.any((row_search != 0) & (row_evidence != 0)).item()):
        raise ValueError("a transition received both causal local credits")
    local_rows = row_search + row_evidence
    for trajectory_id, trajectory in trajectories.items():
        for transition_index, transition in enumerate(trajectory["transitions"]):
            if (transition.get("semantic_output") or {}).get("action") == "answer":
                row = row_by_transition[(trajectory_id, transition_index)]
                if float(local_rows[row].item()) != 0.0:
                    raise ValueError("final answer DECISION received local credit")

    advantages = local_rows.unsqueeze(-1) * tensors["response_mask"].float()
    if not bool(torch.isfinite(advantages).all().item()):
        raise ValueError("non-finite C token advantages")
    def credited_tokens(rows: torch.Tensor) -> int:
        return sum(int(mask[index].sum().item()) for index, value in enumerate(rows) if float(value.item()) != 0.0)
    search_tokens, evidence_tokens = credited_tokens(row_search), credited_tokens(row_evidence)
    either_tokens = int((advantages[mask] != 0).sum().item())
    if either_tokens != search_tokens + evidence_tokens:
        raise ValueError("local credit did not broadcast exactly over response masks")

    output = copy.deepcopy(pack)
    output["metadata"] = dict(output["metadata"])
    output["metadata"].update({
        "source_run_id": "offline-musique-causal-split-credit-C-n8-20260830",
        "scorer": "outcome_v2 terminal + split scorer-side retrieval-coverage/search and delta_F2/evidence credit",
        "transition_diagnostic_training_weight": 1,
        "transition_advantage_definition": "search: same qid+search ordinal delta retrieval coverage; evidence: same qid+evidence ordinal delta F2",
        "terminal_advantage_weight": 1,
        "search_progress_advantage_weight": 1,
        "evidence_progress_advantage_weight": 1,
    })
    output["tensor_fields"] = dict(output["tensor_fields"])
    output["tensor_fields"]["advantages"] = advantages
    output["tensor_fields"]["returns"] = advantages.clone()
    output["captured_field_digest"] = _field_digest(output["tensor_fields"], output["non_tensor_batch"], output["meta_info"])
    args.output_pack.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_pack.with_name(f".{args.output_pack.name}.tmp")
    torch.save(output, temporary)
    temporary.replace(args.output_pack)

    def group_summary(items: list[dict], label: str) -> dict:
        by_hop = {}
        for hop in ("2hop", "3hop", "4hop"):
            group = [item for item in items if item["hop"] == hop]
            by_hop[hop] = {"group_count": len(group), "nondegenerate_group_count": sum(item["nondegenerate"] for item in group), "qids_with_signal": len({item["qid"] for item in group if item["nondegenerate"]})}
        return {
            "credit": label,
            "group_count": len(items),
            "nondegenerate_group_count": sum(item["nondegenerate"] for item in items),
            "max_abs_group_mean": max((abs(item["credit_mean"]) for item in items), default=0.0),
            "qids_with_signal": sorted({item["qid"] for item in items if item["nondegenerate"]}),
            "by_hop": by_hop,
        }

    search_summary, evidence_summary = group_summary(search_audit, "search_retrieval_coverage"), group_summary(evidence_audit, "evidence_delta_F2")
    search_qids, evidence_qids = set(search_summary["qids_with_signal"]), set(evidence_summary["qids_with_signal"])
    invariant_hashes = {name: tensor_sha256(tensors[name]) for name in ("input_ids", "attention_mask", "response_mask", "old_log_probs")}
    audit = {
        "schema_version": 1,
        "kind": "offline_musique_causal_split_credit_C0_audit",
        "source": {
            "enriched": str(args.enriched), "terminal_pack": str(args.terminal_pack), "output_pack": str(args.output_pack),
            "terminal_pack_sha256": hashlib.sha256(args.terminal_pack.read_bytes()).hexdigest(),
            "output_pack_sha256": hashlib.sha256(args.output_pack.read_bytes()).hexdigest(),
        },
        "hard_boundaries": {"actor_visible_scorer_fields": False, "new_train_rollouts": 0, "terminal_advantage_all_zero": True, "final_answer_local_credit_zero": True},
        "identity_and_preservation": {"exact_trajectory_transition_identity": True, "old_logprob_response_mask_preserved": True, "invariant_tensor_sha256": invariant_hashes},
        "mapping": {"search_credit_only_on_search_decisions": True, "evidence_credit_only_on_evidence_updates": True, "search_mapped_transition_count": len(search_credit), "evidence_mapped_transition_count": len(evidence_credit)},
        "normalization": {"finite": True, "zero_mean_tolerance": 2e-6, "search": search_summary, "evidence": evidence_summary},
        "qid_signal_coverage": {"search": len(search_qids), "evidence": len(evidence_qids), "union": len(search_qids | evidence_qids), "intersection": len(search_qids & evidence_qids)},
        "token_credit": {"response_token_count": int(mask.sum().item()), "nonzero_search_token_count": search_tokens, "nonzero_evidence_token_count": evidence_tokens, "nonzero_local_credit_token_count": either_tokens, "nonzero_local_credit_fraction": either_tokens / int(mask.sum().item()), "final_answer_tokens_with_local_credit": 0},
        "expected_estimate_comparison": {"search_nondegenerate_expected_approx": 182, "evidence_nondegenerate_expected_approx": 145, "union_qids_expected_approx": 98, "nonzero_local_tokens_expected_approx": 59255},
    }
    if search_summary["max_abs_group_mean"] > 2e-6 or evidence_summary["max_abs_group_mean"] > 2e-6:
        raise ValueError("C local groups failed zero-mean gate")
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
