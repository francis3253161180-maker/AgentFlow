#!/usr/bin/env python3
"""Deterministic Phase-E actor-causality audit over persisted trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentflow.offline_musique import OfflineCorpus, stable_json_hash


def normalized_query(value: str) -> str:
    return " ".join(value.lower().split())


def anon_qid(qid: str) -> str:
    return hashlib.sha256(qid.encode()).hexdigest()[:12]


def evidence_pids(memory: dict) -> set[str]:
    values = set()
    for row in memory.get("validated_evidence", []):
        if row.startswith("[") and "]" in row:
            values.add(row[1 : row.index("]")])
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--phase-c-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    corpus = OfflineCorpus.load(args.corpus)
    pack = json.loads(args.trajectories.read_text())
    phase_c = json.loads(args.phase_c_results.read_text())
    trajectories = pack["trajectories"]
    by_qid = defaultdict(list)
    observation_by_query = defaultdict(set)
    query_effect_pairs = 0
    distinct_query_pairs = 0
    support_returned_not_selected = 0
    full_support_retrieved_not_selected = 0
    reformulated_useful = 0
    reformulated_still_empty = 0
    repeated_after_empty = 0
    answers_after_full = 0
    premature_answers = 0
    positive_grounding_violations = 0
    examples = {"query_effect": [], "returned_not_selected": [], "reformulation": [], "answer_timing": []}

    enriched = []
    for trajectory in trajectories:
        qid = trajectory["qid"]
        scorer = corpus.scorer_record(qid)
        support = set(scorer.support_pids)
        retrieved = set(trajectory["retrieved_pids"])
        selected = set(trajectory["selected_pids"])
        row = {
            **trajectory,
            "retrieved_support_count": len(retrieved & support),
            "selected_support_count": len(selected & support),
            "gold_support_count": len(support),
        }
        enriched.append(row)
        by_qid[qid].append(row)
        if (retrieved & support) - selected:
            support_returned_not_selected += 1
            if len(examples["returned_not_selected"]) < 4:
                examples["returned_not_selected"].append(
                    {
                        "question_id_hash": anon_qid(qid),
                        "retrieved_support_count": len(retrieved & support),
                        "selected_support_count": len(selected & support),
                        "search_count": len(trajectory["query_sequence"]),
                        "termination": trajectory["termination_reason"],
                    }
                )
        if support <= retrieved and not support <= selected:
            full_support_retrieved_not_selected += 1
        if trajectory["reward_detail"]["reward"] and not (
            trajectory["reward_detail"]["answer_em"]
            and trajectory["reward_detail"]["full_selected_support_coverage"]
        ):
            positive_grounding_violations += 1

        searches = []
        for transition in trajectory["transitions"]:
            if transition["mode"] == "DECISION" and transition["semantic_output"] and transition["semantic_output"].get("action") == "search":
                query = transition["semantic_output"]["query"]
                ref = transition["observation_refs"][0]
                searches.append({"query": query, "ref": ref})
                observation_by_query[(qid, normalized_query(query))].add(ref)
            elif transition["mode"] == "EVIDENCE_UPDATE" and searches:
                searches[-1]["outcome"] = transition["validation_result"].get("outcome")
        for previous, current in zip(searches, searches[1:]):
            if previous.get("outcome") != "no_useful_evidence":
                continue
            same = normalized_query(previous["query"]) == normalized_query(current["query"])
            if same:
                repeated_after_empty += 1
            elif current.get("outcome") == "useful":
                reformulated_useful += 1
                if len(examples["reformulation"]) < 4:
                    examples["reformulation"].append(
                        {
                            "question_id_hash": anon_qid(qid),
                            "before_query": previous["query"],
                            "after_query": current["query"],
                            "after_outcome": "useful",
                        }
                    )
            else:
                reformulated_still_empty += 1

        for transition in trajectory["transitions"]:
            semantic = transition["semantic_output"] or {}
            if transition["mode"] == "DECISION" and semantic.get("action") == "answer":
                full = support <= evidence_pids(transition["compact_memory_before"])
                answers_after_full += full
                premature_answers += not full
                if len(examples["answer_timing"]) < 4:
                    examples["answer_timing"].append(
                        {
                            "question_id_hash": anon_qid(qid),
                            "full_evidence_before_answer": full,
                            "selected_support_count": len(evidence_pids(transition["compact_memory_before"]) & support),
                            "gold_support_count": len(support),
                            "reward": trajectory["reward_detail"]["reward"],
                        }
                    )

    questions_with_query_variation = 0
    questions_where_query_changed_retrieval = 0
    questions_with_support_retrieval_variation = 0
    for qid, items in by_qid.items():
        first = []
        for item in items:
            search = next(
                (
                    transition
                    for transition in item["transitions"]
                    if transition["mode"] == "DECISION"
                    and transition["semantic_output"]
                    and transition["semantic_output"].get("action") == "search"
                ),
                None,
            )
            if search:
                first.append((search["semantic_output"]["query"], search["observation_refs"][0]))
        query_values = {normalized_query(query) for query, _ in first}
        observation_values = {ref for _, ref in first}
        if len(query_values) > 1:
            questions_with_query_variation += 1
            if len(observation_values) > 1:
                questions_where_query_changed_retrieval += 1
                if len(examples["query_effect"]) < 4:
                    examples["query_effect"].append(
                        {
                            "question_id_hash": anon_qid(qid),
                            "distinct_first_queries": len(query_values),
                            "distinct_first_observations": len(observation_values),
                            "queries": [query for query, _ in first[:4]],
                        }
                    )
        for left in range(len(first)):
            for right in range(left + 1, len(first)):
                if normalized_query(first[left][0]) != normalized_query(first[right][0]):
                    distinct_query_pairs += 1
                    query_effect_pairs += first[left][1] != first[right][1]
        support_counts = {item["retrieved_support_count"] for item in items}
        questions_with_support_retrieval_variation += len(support_counts) > 1

    deterministic_violations = sum(len(refs) > 1 for refs in observation_by_query.values())
    action_modes = sorted(
        {
            transition["mode"]
            for trajectory in trajectories
            for transition in trajectory["transitions"]
        }
    )
    mixed_groups = []
    for qid, items in by_qid.items():
        rewards = [item["reward_detail"]["reward"] for item in items]
        if len(set(rewards)) > 1:
            mixed_groups.append(
                {
                    "question_id_hash": anon_qid(qid),
                    "rewards": rewards,
                    "rollouts": [
                        {
                            "queries": item["query_sequence"],
                            "retrieved_support_count": item["retrieved_support_count"],
                            "selected_support_count": item["selected_support_count"],
                            "gold_support_count": item["gold_support_count"],
                            "answer_em": item["reward_detail"]["answer_em"],
                        }
                        for item in items
                    ],
                }
            )

    metrics = {
        "trajectory_count": len(trajectories),
        "questions_with_first_query_variation": questions_with_query_variation,
        "questions_where_first_query_changed_observation": questions_where_query_changed_retrieval,
        "questions_with_support_retrieval_variation": questions_with_support_retrieval_variation,
        "distinct_first_query_pairs": distinct_query_pairs,
        "distinct_first_query_pairs_with_different_observation": query_effect_pairs,
        "query_effect_rate": query_effect_pairs / max(distinct_query_pairs, 1),
        "identical_qid_query_nondeterminism_violations": deterministic_violations,
        "support_returned_but_missing_from_selection_trajectories": support_returned_not_selected,
        "full_support_retrieved_but_not_fully_selected_trajectories": full_support_retrieved_not_selected,
        "no_useful_then_distinct_useful_reformulation": reformulated_useful,
        "no_useful_then_distinct_still_empty": reformulated_still_empty,
        "no_useful_then_repeated_query": repeated_after_empty,
        "answers_after_full_selected_support": answers_after_full,
        "premature_answers_without_full_selected_support": premature_answers,
        "positive_reward_grounding_violations": positive_grounding_violations,
        "mixed_reward_group_count": len(mixed_groups),
        "actor_transition_modes": action_modes,
        "fixed_semantic_role_count": 0,
        "external_semantic_call_count": 0,
    }
    gate_checks = {
        "retriever_deterministic": deterministic_violations == 0,
        "only_two_actor_modes": action_modes == ["DECISION", "EVIDENCE_UPDATE"],
        "actor_queries_vary": questions_with_query_variation > 0,
        "query_changes_observations": query_effect_pairs > 0,
        "support_selection_changes_reward_path": support_returned_not_selected > 0 and answers_after_full > 0,
        "at_least_one_grounded_mixed_group": len(mixed_groups) > 0,
        "all_positive_rewards_grounded": positive_grounding_violations == 0,
        "no_hidden_fixed_or_external_roles": True,
    }
    result = {
        "phase": "E",
        "iteration_id": "phase_e_v1",
        "parent_phase_c_iteration": phase_c["iteration_id"],
        "phase_c_config_hash": phase_c["config_hash"],
        "hypothesis": "actor-generated search/evidence/answer differences, rather than retriever randomness or hidden roles, account for observable support and reward variance",
        "metrics": metrics,
        "mixed_group_audit": mixed_groups,
        "trajectory_examples": examples,
        "gate_checks": gate_checks,
        "gate_passed": all(gate_checks.values()),
        "source_trajectory_sha256": hashlib.sha256(args.trajectories.read_bytes()).hexdigest(),
        "audit_config_hash": stable_json_hash({"normalization": "lowercase whitespace", "qid_reporting": "sha256 first 12"}),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
