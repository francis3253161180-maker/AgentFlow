#!/usr/bin/env python3
"""Materialize the fixed delta-F2 transition-credit replay pack for branch B."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from agentflow.verl.unified_smoke_capture import _field_digest


def normalize_group(values: list[float], epsilon: float = 1e-6) -> list[float]:
    """Match the repo GRPO sample-standard-deviation normalization."""
    if len(values) < 2:
        return [0.0] * len(values)
    tensor = torch.tensor(values, dtype=torch.float32)
    # torch.std can report a tiny positive round-off residual for a vector of
    # exactly equal float32 values; the specified zero-variance rule is about
    # the values, not that reduction artifact.
    if bool(torch.all(tensor == tensor[0]).item()):
        return [0.0] * len(values)
    std = tensor.std(unbiased=True)
    if not bool(torch.isfinite(std).item()) or float(std.item()) == 0.0:
        return [0.0] * len(values)
    normalized = (tensor - tensor.mean()) / (std + epsilon)
    if not bool(torch.isfinite(normalized).all().item()):
        raise ValueError("non-finite local progress advantage")
    return [float(value) for value in normalized.tolist()]


def hop_of(qid: str) -> str:
    for hop in ("2hop", "3hop", "4hop"):
        if qid.startswith(hop):
            return hop
    raise ValueError(f"unrecognized MuSiQue qid hop: {qid}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enriched", type=Path, required=True)
    parser.add_argument("--terminal-pack", type=Path, required=True)
    parser.add_argument("--output-pack", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()

    enriched = json.loads(args.enriched.read_text(encoding="utf-8"))
    annotations = enriched["scorer_side_transition_diagnostics"]
    if annotations.get("actor_visible") or annotations.get("training_weight") != 0:
        raise ValueError("source diagnostics are not scorer-only weight-zero annotations")
    pack = torch.load(args.terminal_pack, map_location="cpu", weights_only=False)
    tensors = pack["tensor_fields"]
    non_tensor = pack["non_tensor_batch"]
    mask = tensors["response_mask"].bool()
    terminal_advantages = tensors["advantages"].float()
    if not bool(torch.all(terminal_advantages == 0).item()):
        raise ValueError("branch B must start from an all-zero terminal advantage pack")

    row_by_transition = {
        (str(trajectory_id), int(turn_index)): row
        for row, (trajectory_id, turn_index) in enumerate(
            zip(non_tensor["trajectory_id_list"], non_tensor["turn_index_list"])
        )
    }
    if len(row_by_transition) != len(mask):
        raise ValueError("replay has duplicate trajectory/turn rows")
    trajectory_rows = {row["trajectory_id"]: row for row in enriched["trajectories"]}

    local_entries: dict[tuple[str, int], list[tuple[str, int, float]]] = defaultdict(list)
    for annotation in annotations["annotations"]:
        trajectory_id = str(annotation["trajectory_id"])
        for score in annotation["transition_scores"]:
            ordinal = int(score["evidence_update_ordinal"])
            transition_index = int(score["transition_index"])
            local_entries[(str(annotation["qid"]), ordinal)].append(
                (trajectory_id, transition_index, float(score["delta_F2"]))
            )

    progress_by_transition: dict[tuple[str, int], float] = {}
    group_audit = []
    for (qid, ordinal), entries in sorted(local_entries.items()):
        normalized = normalize_group([entry[2] for entry in entries])
        if len(normalized) != len(entries):
            raise AssertionError("normalization cardinality changed")
        for (trajectory_id, transition_index, _delta), advantage in zip(entries, normalized):
            key = (trajectory_id, transition_index)
            if key in progress_by_transition:
                raise ValueError(f"duplicate scorer transition annotation: {key}")
            progress_by_transition[key] = advantage
        group_audit.append(
            {
                "qid": qid,
                "hop": hop_of(qid),
                "evidence_update_ordinal": ordinal,
                "valid_rollout_count": len(entries),
                "delta_F2_values": [entry[2] for entry in entries],
                "A_prog_values": normalized,
                "nondegenerate": any(value != 0.0 for value in normalized),
                "A_prog_mean": float(np.mean(normalized)) if normalized else 0.0,
            }
        )

    row_progress = torch.zeros(len(mask), dtype=torch.float32)
    mapped_evidence_rows = set()
    mapped_causal_decision_rows = set()
    for (trajectory_id, evidence_index), advantage in progress_by_transition.items():
        evidence_row = row_by_transition.get((trajectory_id, evidence_index))
        decision_row = row_by_transition.get((trajectory_id, evidence_index - 1))
        if evidence_row is None or decision_row is None:
            raise ValueError(f"missing causal transition pair for {trajectory_id}/{evidence_index}")
        if str(non_tensor["policy_mode_list"][evidence_row]) != "EVIDENCE_UPDATE":
            raise ValueError("scorer evidence annotation does not map to EVIDENCE_UPDATE row")
        if str(non_tensor["policy_mode_list"][decision_row]) != "DECISION":
            raise ValueError("evidence update predecessor is not a DECISION row")
        source = trajectory_rows[trajectory_id]["transitions"]
        prior = source[evidence_index - 1].get("semantic_output", {})
        if prior.get("action") != "search":
            raise ValueError("causal DECISION predecessor is not a search action")
        row_progress[evidence_row] = advantage
        row_progress[decision_row] = advantage
        mapped_evidence_rows.add(evidence_row)
        mapped_causal_decision_rows.add(decision_row)

    advantages = row_progress.unsqueeze(-1) * tensors["response_mask"].float()
    if not bool(torch.isfinite(advantages).all().item()):
        raise ValueError("non-finite token advantages")
    expected_nonzero = sum(
        int(mask[row].sum().item())
        for row, value in enumerate(row_progress)
        if float(value.item()) != 0.0
    )
    nonzero = int((advantages[mask] != 0).sum().item())
    if expected_nonzero != nonzero:
        raise ValueError("progress advantage was not broadcast to every response token")

    output = copy.deepcopy(pack)
    output["metadata"] = dict(output["metadata"])
    output["metadata"].update(
        {
            "source_run_id": "offline-musique-transition-aware-grpo-n8-20260830",
            "scorer": "outcome_v2 terminal + scorer-side delta_F2 local advantage; lambda=1.0",
            "transition_diagnostic_training_weight": 1,
            "transition_advantage_definition": "same qid + evidence ordinal sample-std normalized delta_F2",
            "terminal_advantage_weight": 1,
            "progress_advantage_weight": 1,
        }
    )
    output["tensor_fields"] = dict(output["tensor_fields"])
    output["tensor_fields"]["advantages"] = advantages
    output["tensor_fields"]["returns"] = advantages.clone()
    output["captured_field_digest"] = _field_digest(
        output["tensor_fields"], output["non_tensor_batch"], output["meta_info"]
    )
    args.output_pack.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_pack.with_name(f".{args.output_pack.name}.tmp")
    torch.save(output, temporary)
    temporary.replace(args.output_pack)

    qids_with_signal = {
        item["qid"] for item in group_audit if item["nondegenerate"]
    }
    per_hop = {}
    for hop in ("2hop", "3hop", "4hop"):
        groups = [item for item in group_audit if item["hop"] == hop]
        per_hop[hop] = {
            "local_group_count": len(groups),
            "nondegenerate_local_group_count": sum(item["nondegenerate"] for item in groups),
            "qids_with_nonzero_local_progress": len({item["qid"] for item in groups if item["nondegenerate"]}),
        }
    ordinal_summary = {}
    for ordinal in range(1, 7):
        groups = [item for item in group_audit if item["evidence_update_ordinal"] == ordinal]
        ordinal_summary[str(ordinal)] = {
            "local_group_count": len(groups),
            "nondegenerate_local_group_count": sum(item["nondegenerate"] for item in groups),
            "valid_evidence_update_count": sum(item["valid_rollout_count"] for item in groups),
        }
    group_digest = hashlib.sha256(
        json.dumps(group_audit, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    audit = {
        "schema_version": 1,
        "kind": "offline_musique_transition_aware_advantage_audit",
        "source": {
            "enriched_trajectory_path": str(args.enriched),
            "terminal_pack_path": str(args.terminal_pack),
            "terminal_pack_sha256": hashlib.sha256(args.terminal_pack.read_bytes()).hexdigest(),
            "output_pack_path": str(args.output_pack),
            "output_pack_sha256": hashlib.sha256(args.output_pack.read_bytes()).hexdigest(),
        },
        "hard_boundaries": {
            "terminal_exact_set_reward_preserved": True,
            "actor_visible_scorer_fields": False,
            "lambda": 1.0,
            "normalization": "within same qid and evidence_update_ordinal only",
        },
        "mapping": {
            "annotated_valid_evidence_updates": len(progress_by_transition),
            "mapped_evidence_transition_count": len(mapped_evidence_rows),
            "mapped_causal_decision_transition_count": len(mapped_causal_decision_rows),
            "final_answer_and_unmapped_transition_terminal_only": True,
            "all_causal_predecessors_are_search_decisions": True,
        },
        "local_progress": {
            "group_count": len(group_audit),
            "nondegenerate_group_count": sum(item["nondegenerate"] for item in group_audit),
            "zero_mean_tolerance": 2e-6,
            "max_abs_group_mean": max((abs(item["A_prog_mean"]) for item in group_audit), default=0.0),
            "all_zero_terminal_qids_rescued_by_nonzero_local_progress": len(qids_with_signal),
            "per_hop": per_hop,
            "per_evidence_ordinal": ordinal_summary,
            "full_scorer_group_audit_sha256": group_digest,
            "representative_nondegenerate_groups": [
                item for item in group_audit if item["nondegenerate"]
            ][:8],
        },
        "token_credit": {
            "response_token_count": int(mask.sum().item()),
            "nonzero_terminal_advantage_token_count": 0,
            "nonzero_progress_advantage_token_count": nonzero,
            "nonzero_either_advantage_token_count": nonzero,
            "nonzero_progress_advantage_token_fraction": nonzero / int(mask.sum().item()),
            "progress_is_broadcast_over_response_masks": True,
            "finite": True,
        },
    }
    if audit["local_progress"]["max_abs_group_mean"] > audit["local_progress"]["zero_mean_tolerance"]:
        raise ValueError("local progress groups are not zero mean")
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
