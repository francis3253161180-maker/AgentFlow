#!/usr/bin/env python3
"""Fail-closed, advantage-only audit for role-step C-v2 MuSiQue credit.

This script deliberately does not write a replay pack.  It constructs the
would-be task advantages in memory solely to audit eligibility, group-local
normalization, masking, and exploitability before any optimizer can run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import torch


FORBIDDEN_ACTOR_FIELDS = (
    "support_pids", "answer_aliases", "question_decomposition",
    "paragraph_support_idx", "delta_F1", "delta_F2", "new_gold_support_count",
    "exact_support_set",
)
ROLE_SEARCH = "SEARCH"
ROLE_EVIDENCE = "EVIDENCE_UPDATE"
ROLE_FINAL = "FINAL_ANSWER"
PID_RE = re.compile(r"^\[([^\]]+)\]")


def hop_of(qid: str) -> str:
    for hop in ("2hop", "3hop", "4hop"):
        if qid.startswith(hop):
            return hop
    raise ValueError(f"unrecognized MuSiQue qid hop: {qid}")


def normalized(values: list[float], epsilon: float = 1e-6) -> tuple[list[float], bool]:
    """Return sample-std normalized values and whether the group has signal."""
    if len(values) < 2:
        return [0.0] * len(values), False
    tensor = torch.tensor(values, dtype=torch.float32)
    if bool(torch.all(tensor == tensor[0]).item()):
        return [0.0] * len(values), False
    std = tensor.std(unbiased=True)
    if not bool(torch.isfinite(std).item()) or float(std.item()) <= 1e-12:
        return [0.0] * len(values), False
    result = (tensor - tensor.mean()) / (std + epsilon)
    if not bool(torch.isfinite(result).all().item()):
        raise ValueError("non-finite local advantage")
    return [float(value) for value in result.tolist()], True


def validated_pids(memory: dict) -> set[str]:
    result = set()
    for entry in memory.get("validated_evidence", []):
        match = PID_RE.match(str(entry))
        if match:
            result.add(match.group(1))
    return result


def event_opportunity(observation: list[dict], gold: set[str], consumed: set[str], remaining_capacity: int) -> set[str]:
    """Scorer-side selectable gold supports in the immediate observation."""
    if remaining_capacity <= 0:
        return set()
    observed = {str(row["pid"]) for row in observation}
    return (observed & gold) - consumed


def is_valid_search(transition: dict) -> bool:
    semantic = transition.get("semantic_output") or {}
    validation = transition.get("validation_result") or {}
    return (
        transition.get("mode") == "DECISION"
        and semantic.get("action") == "search"
        and not validation.get("format_failure", False)
        and bool(validation.get("schema_valid", False))
    )


def is_final_answer(transition: dict) -> bool:
    semantic = transition.get("semantic_output") or {}
    validation = transition.get("validation_result") or {}
    return (
        transition.get("mode") == "DECISION"
        and semantic.get("action") == "answer"
        and not validation.get("format_failure", False)
        and bool(validation.get("schema_valid", False))
    )


def sha256_tensor(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def group_key(event: dict) -> tuple[str, str, int]:
    return str(event["qid"]), str(event["role"]), int(event["ordinal"])


def summarize(events: list[dict], *, row_token_counts: dict[tuple[str, int], int]) -> dict:
    """Summarize all role-step events; only eligible values enter normalization."""
    by_key: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for event in events:
        by_key[group_key(event)].append(event)
    signal_rows: set[tuple[str, int]] = set()
    groups = []
    for key, members in sorted(by_key.items()):
        eligible = [member for member in members if member["eligible"]]
        rewards = [float(member["reward"]) for member in eligible]
        credits, nondegenerate = normalized(rewards)
        if len(credits) != len(eligible):
            raise AssertionError("normalization cardinality changed")
        for member, credit in zip(eligible, credits):
            member["task_advantage"] = credit
            if credit != 0.0:
                signal_rows.add((member["trajectory_id"], member["transition_index"]))
        for member in members:
            if not member["eligible"]:
                member["task_advantage"] = 0.0
        group_signal_rows = {
            (member["trajectory_id"], member["transition_index"])
            for member in eligible if member["task_advantage"] != 0.0
        }
        groups.append({
            "qid": key[0], "hop": hop_of(key[0]), "role": key[1], "ordinal": key[2],
            "event_count": len(members), "eligible_event_count": len(eligible),
            "ineligible_event_count": len(members) - len(eligible),
            "eligible_group_size_at_least_two": len(eligible) >= 2,
            "zero_std_or_insufficient": bool(eligible) and not nondegenerate,
            "nondegenerate": nondegenerate,
            "eligible_group_mean_after_normalization": (
                sum(credits) / len(credits) if credits else 0.0
            ),
            "effective_task_pg_token_count": sum(row_token_counts[row] for row in group_signal_rows),
        })
    all_response_tokens = sum(row_token_counts.values())
    def aggregate(selected: list[dict]) -> dict:
        task_tokens = sum(group["effective_task_pg_token_count"] for group in selected)
        return {
            "total_groups": len(selected),
            "eligible_event_count": sum(group["eligible_event_count"] for group in selected),
            "ineligible_event_count": sum(group["ineligible_event_count"] for group in selected),
            "groups_with_at_least_two_eligible": sum(group["eligible_group_size_at_least_two"] for group in selected),
            "nondegenerate_groups": sum(group["nondegenerate"] for group in selected),
            "zero_std_or_insufficient_groups": sum(group["zero_std_or_insufficient"] for group in selected),
            "qids_with_signal": len({group["qid"] for group in selected if group["nondegenerate"]}),
            "max_abs_eligible_group_mean": max((abs(group["eligible_group_mean_after_normalization"]) for group in selected), default=0.0),
            "effective_task_pg_token_count": task_tokens,
            "effective_task_pg_token_fraction_of_all_response_tokens": task_tokens / all_response_tokens,
        }
    role_summary = {}
    for role in (ROLE_SEARCH, ROLE_EVIDENCE, ROLE_FINAL):
        selected = [group for group in groups if group["role"] == role]
        events_for_role = [event for event in events if event["role"] == role]
        role_summary[role] = {**aggregate(selected), "event_count": len(events_for_role)}
    by_hop, by_ordinal, by_role_ordinal = {}, {}, {}
    for hop in ("2hop", "3hop", "4hop"):
        selected = [group for group in groups if group["hop"] == hop]
        by_hop[hop] = aggregate(selected)
    for ordinal in sorted({group["ordinal"] for group in groups}):
        selected = [group for group in groups if group["ordinal"] == ordinal]
        by_ordinal[str(ordinal)] = aggregate(selected)
    for role in (ROLE_SEARCH, ROLE_EVIDENCE, ROLE_FINAL):
        by_role_ordinal[role] = {
            str(ordinal): aggregate([group for group in groups if group["role"] == role and group["ordinal"] == ordinal])
            for ordinal in sorted({group["ordinal"] for group in groups if group["role"] == role})
        }
    token_count = sum(row_token_counts[row] for row in signal_rows)
    return {
        "by_role": role_summary, "by_hop": by_hop, "by_ordinal": by_ordinal, "by_role_ordinal": by_role_ordinal,
        "group_count": len(groups), "event_count": len(events),
        "effective_task_pg_token_count": token_count,
        "signal_rows": signal_rows, "groups": groups,
    }


def delta_f2_signal_summary(diagnostics: dict) -> dict:
    """Recompute B/C-v1 scorer-side delta-F2 group density from raw annotations."""
    groups: dict[tuple[str, int], list[float]] = defaultdict(list)
    for annotation in diagnostics["annotations"]:
        for score in annotation["transition_scores"]:
            groups[(str(annotation["qid"]), int(score["evidence_update_ordinal"]))].append(float(score["delta_F2"]))
    records = []
    for (qid, ordinal), values in sorted(groups.items()):
        _credit, has_signal = normalized(values)
        records.append({"qid": qid, "hop": hop_of(qid), "ordinal": ordinal, "nondegenerate": has_signal})
    return {
        "group_count": len(records),
        "nondegenerate_group_count": sum(row["nondegenerate"] for row in records),
        "qids_with_signal": len({row["qid"] for row in records if row["nondegenerate"]}),
        "by_hop": {
            hop: {
                "group_count": sum(row["hop"] == hop for row in records),
                "nondegenerate_group_count": sum(row["hop"] == hop and row["nondegenerate"] for row in records),
            } for hop in ("2hop", "3hop", "4hop")
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enriched", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--terminal-pack", type=Path, required=True)
    parser.add_argument("--b-pack", type=Path, required=True)
    parser.add_argument("--cv1-pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    enriched = json.loads(args.enriched.read_text(encoding="utf-8"))
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    diagnostics = enriched["scorer_side_transition_diagnostics"]
    if diagnostics.get("actor_visible") or diagnostics.get("training_weight") != 0:
        raise ValueError("source scorer diagnostics must be actor-hidden and weight-zero")
    leaks = [
        (prompt_hash, field) for prompt_hash, prompt in enriched["audit_prompts"].items()
        for field in FORBIDDEN_ACTOR_FIELDS if field in prompt
    ]
    if leaks:
        raise ValueError(f"scorer fields leaked to actor prompt: {leaks[:3]}")
    observation_label_leaks = [
        (ref, field) for ref, observation in enriched["audit_observations"].items()
        for row in observation for field in FORBIDDEN_ACTOR_FIELDS
        if field in row
    ]
    if observation_label_leaks:
        raise ValueError(f"scorer fields leaked to runtime observation: {observation_label_leaks[:3]}")

    packs = {name: torch.load(path, map_location="cpu", weights_only=False) for name, path in {
        "terminal": args.terminal_pack, "B": args.b_pack, "C_v1": args.cv1_pack,
    }.items()}
    tensors, non_tensor = packs["terminal"]["tensor_fields"], packs["terminal"]["non_tensor_batch"]
    mask = tensors["response_mask"].bool()
    if not bool(torch.all(tensors["advantages"].float() == 0).item()):
        raise ValueError("terminal replay pack must retain all-zero terminal task advantage")
    invariants = ("input_ids", "attention_mask", "response_mask", "old_log_probs")
    invariant_hashes = {name: sha256_tensor(tensors[name]) for name in invariants}
    for pack_name in ("B", "C_v1"):
        candidate = packs[pack_name]["tensor_fields"]
        if any(sha256_tensor(candidate[name]) != invariant_hashes[name] for name in invariants):
            raise ValueError(f"{pack_name} changed a frozen replay tensor")
    row_by_transition = {
        (str(tid), int(turn)): index for index, (tid, turn) in enumerate(
            zip(non_tensor["trajectory_id_list"], non_tensor["turn_index_list"])
        )
    }
    if len(row_by_transition) != len(mask):
        raise ValueError("replay has non-unique transition identities")
    trajectories = {str(row["trajectory_id"]): row for row in enriched["trajectories"]}
    if set(trajectories) != {key[0] for key in row_by_transition}:
        raise ValueError("trajectory identity differs from persisted replay")
    if len(trajectories) != 1024 or len(mask) != 6007 or len({row["qid"] for row in trajectories.values()}) != 128:
        raise ValueError("C-v2 must use exactly the frozen 128x8 / 6007 replay")
    row_token_counts = {
        key: int(mask[index].sum().item()) for key, index in row_by_transition.items()
    }

    events, evidence_exploit = [], []
    unmapped = 0
    for trajectory_id, trajectory in trajectories.items():
        qid, gold = str(trajectory["qid"]), set(corpus["scorer_only"][trajectory["qid"]]["support_pids"])
        search_ordinal = evidence_ordinal = final_ordinal = 0
        for transition_index, transition in enumerate(trajectory["transitions"]):
            key = (trajectory_id, transition_index)
            if key not in row_by_transition:
                raise ValueError(f"missing replay row for {key}")
            if is_valid_search(transition):
                search_ordinal += 1
                observation = enriched["audit_observations"][transition["observation_refs"][0]]
                previous = set()
                for previous_transition in trajectory["transitions"][:transition_index]:
                    if is_valid_search(previous_transition):
                        previous.update(row["pid"] for row in enriched["audit_observations"][previous_transition["observation_refs"][0]])
                reward = len(({str(row["pid"]) for row in observation} & gold) - previous) / len(gold)
                events.append({"trajectory_id": trajectory_id, "transition_index": transition_index, "qid": qid, "role": ROLE_SEARCH, "ordinal": search_ordinal, "eligible": True, "reward": reward})
            elif transition.get("mode") == "EVIDENCE_UPDATE":
                evidence_ordinal += 1
                observation = enriched["audit_observations"][transition["observation_refs"][0]]
                consumed = validated_pids(transition.get("compact_memory_before") or {})
                remaining_capacity = 6 - len(consumed)
                opportunity = event_opportunity(observation, gold, consumed, remaining_capacity)
                accepted = {str(row["pid"]) for row in (transition.get("validation_result") or {}).get("accepted", [])}
                selectable = {str(row["pid"]) for row in observation} - consumed
                reward = len(accepted & opportunity) / len(opportunity) if opportunity else 0.0
                eligible = bool(opportunity)
                events.append({"trajectory_id": trajectory_id, "transition_index": transition_index, "qid": qid, "role": ROLE_EVIDENCE, "ordinal": evidence_ordinal, "eligible": eligible, "reward": reward})
                if eligible:
                    evidence_exploit.append({
                        "candidate_count": len(observation), "selectable_candidate_count": len(selectable),
                        "opportunity_count": len(opportunity), "remaining_capacity": remaining_capacity,
                        "select_all_selectable_reward": len(selectable & opportunity) / len(opportunity),
                        "accepted_distractor_count": len(accepted - gold),
                        "rejected_selection_count": len((transition.get("validation_result") or {}).get("rejected", [])),
                    })
            elif is_final_answer(transition):
                final_ordinal += 1
                consumed = validated_pids(transition.get("compact_memory_before") or {})
                eligible = gold.issubset(consumed)
                events.append({"trajectory_id": trajectory_id, "transition_index": transition_index, "qid": qid, "role": ROLE_FINAL, "ordinal": final_ordinal, "eligible": eligible, "reward": float(trajectory["reward_detail"].get("answer_em", False))})
            else:
                unmapped += 1
    summary = summarize(events, row_token_counts=row_token_counts)
    if any(event["task_advantage"] != 0.0 for event in events if not event["eligible"]):
        raise AssertionError("ineligible event received task advantage")
    if not all(torch.isfinite(torch.tensor(event["task_advantage"])).item() for event in events):
        raise ValueError("non-finite C-v2 task advantage")
    eligible_exploit = len(evidence_exploit)
    maximal_by_select_all = sum(row["select_all_selectable_reward"] == 1.0 for row in evidence_exploit)
    exploitability = {
        "eligible_evidence_event_count": eligible_exploit,
        "select_all_selectable_maximizes_reward_count": maximal_by_select_all,
        "select_all_selectable_maximizes_reward_rate": maximal_by_select_all / eligible_exploit if eligible_exploit else 0.0,
        "accepted_distractor_selection_count": sum(row["accepted_distractor_count"] for row in evidence_exploit),
        "rejected_false_positive_selection_count": sum(row["rejected_selection_count"] for row in evidence_exploit),
        "reward_trivially_exploitable": eligible_exploit > 0 and maximal_by_select_all == eligible_exploit,
        "reason": "Selecting every currently selectable candidate includes every opportunity PID and receives recall 1.0 without a false-positive penalty.",
    }
    b_active = int((packs["B"]["tensor_fields"]["advantages"][mask] != 0).sum().item())
    cv1_advantages = packs["C_v1"]["tensor_fields"]["advantages"]
    cv1_active = int((cv1_advantages[mask] != 0).sum().item())
    delta_f2 = delta_f2_signal_summary(diagnostics)
    audit = {
        "schema_version": 1,
        "kind": "offline_musique_causal_role_step_Cv2_advantage_only_audit",
        "status": "FAILED_CLOSED_EXPLOITABLE_EVIDENCE_REWARD" if exploitability["reward_trivially_exploitable"] else "PASS",
        "source": {
            "enriched": str(args.enriched), "terminal_pack": str(args.terminal_pack), "B_pack": str(args.b_pack), "C_v1_pack": str(args.cv1_pack),
            "terminal_pack_sha256": hashlib.sha256(args.terminal_pack.read_bytes()).hexdigest(),
        },
        "hard_boundaries": {
            "new_train_rollouts": 0, "optimizer_steps": 0, "actor_visible_scorer_fields": False,
            "frozen_trajectory_transition_identity": True, "old_logprobs_unchanged": True,
            "input_ids_response_masks_unchanged": True, "ineligible_events_excluded_from_group_stats_and_task_pg": True,
            "search_credit_depends_only_on_query_retrieval_delta": True,
            "final_answer_uses_own_role_step_group": True,
            "runtime_observation_scorer_label_violations": 0,
            "finite_task_advantages": True,
            "KL_mask": "original_response_mask", "official_KL": {"beta": 0.001, "type": "low_var_kl", "entropy": 0.0, "reference": "same_actor_adapter_disabled"},
        },
        "invariant_tensor_sha256": invariant_hashes,
        "C_v2_role_step_credit": {
            "comparison_key": "qid + role/credit_channel + same_role_ordinal",
            "summary": {key: value for key, value in summary.items() if key not in {"signal_rows", "groups"}},
            "effective_task_pg_token_fraction": summary["effective_task_pg_token_count"] / int(mask.sum().item()),
            "ineligible_event_task_pg_token_count": 0,
            "unmapped_transition_count": unmapped,
        },
        "evidence_select_all_exploitability": exploitability,
        "signal_density_recomputed_from_persisted_artifacts": {
            "B": {"delta_F2": delta_f2, "nonzero_task_tokens": b_active},
            "C_v1": {
                "search": {
                    key: summary["by_role"][ROLE_SEARCH][key]
                    for key in ("total_groups", "nondegenerate_groups", "qids_with_signal")
                },
                "evidence_delta_F2": delta_f2,
                "nonzero_task_tokens": cv1_active,
            },
            "C_v2_would_be": {
                "role_step": summary["by_role"],
                "nonzero_task_tokens": summary["effective_task_pg_token_count"],
            },
        },
        "gate": {
            "all_technical_integrity_checks_pass": True,
            "evidence_reward_not_trivially_exploitable": not exploitability["reward_trivially_exploitable"],
            "optimizer_authorized": not exploitability["reward_trivially_exploitable"],
            "stop_condition": "Evidence useful-selection-recall is maximized by selecting every selectable candidate; C-v2 may not proceed without an explicitly authorized reward redesign.",
        },
    }
    if max(group["max_abs_eligible_group_mean"] for group in summary["by_role"].values()) > 2e-6:
        raise ValueError("eligible local groups are not zero mean")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": audit["status"], "optimizer_authorized": audit["gate"]["optimizer_authorized"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
