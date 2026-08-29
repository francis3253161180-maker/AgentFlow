#!/usr/bin/env python3
"""Offline manual audit of the persisted all-Qwen7B MuSiQue/2Wiki rollouts.

This script never loads a model or calls an API.  The labels in MANUAL_REVIEW
were written after directly reading every persisted trajectory listed below.
They are audit annotations, not scorer rules and are intentionally isolated
from production code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOTS = {
    "musique": Path(
        "rollout_data/46.38.243.197/"
        "multihop-allqwen7b-musique-20260829_20260829-165503/"
        "Qwen2.5-7B-Instruct_20260829-165504/train"
    ),
    "2wiki": Path(
        "rollout_data/46.38.243.197/"
        "multihop-allqwen7b-2wiki-20260829_20260829-171907/"
        "Qwen2.5-7B-Instruct_20260829-171908/train"
    ),
}
MANIFEST = Path("log/2026-08-29_multihop_allqwen20_probe_sample_manifest.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text(value: Any, limit: int = 1_200) -> str:
    if isinstance(value, list):
        value = "\n".join(map(str, value))
    elif isinstance(value, dict):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        value = str(value or "")
    value = " ".join(value.split())
    return value if len(value) <= limit else value[:limit] + "…"


def _json_object(value: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {"raw": _text(value)}
    return parsed if isinstance(parsed, dict) else {"raw": _text(parsed)}


def _review(dataset: str, group_id: int, candidate_index: int) -> dict[str, str]:
    """Return the direct, manually written semantic label for one trajectory."""
    # The two explicitly unresolved Charles-University answers contain the
    # expected entity but also make an unscoped, plural-employer claim.  They
    # are intentionally not forced into a binary label.
    if dataset == "musique" and group_id == 0:
        labels = [
            ("clearly_correct", "Names Charles University exactly; the parenthetical University of Prague is the same institution."),
            ("ambiguous_needs_semantic_judgment", "Mentions University of Prague/Charles University but also gives Leiden as another employer; the singular, time-unspecified target cannot be safely reduced to binary."),
            ("clearly_correct", "Names Charles University in Prague, which is the gold institution."),
            ("ambiguous_needs_semantic_judgment", "Mentions University of Prague/Charles University but also gives Leiden as another employer; the singular, time-unspecified target cannot be safely reduced to binary."),
        ]
        label, reason = labels[candidate_index]
        return {"manual_label": label, "manual_reason": reason}
    if dataset == "musique" and group_id == 4:
        labels = [
            ("clearly_correct", "Exact numeric answer 38."),
            ("clearly_correct", "States 38 games per team, semantically equivalent to the gold answer 38."),
            ("clearly_wrong", "Answers 34, not the gold value 38."),
            ("clearly_wrong", "Answers 306, the league-wide total implied by an incorrect interpretation, not the gold per-team value 38."),
        ]
        label, reason = labels[candidate_index]
        return {"manual_label": label, "manual_reason": reason}

    group_reasons = {
        ("musique", 1): "Names Anna Murray Douglass or Susan B. Anthony, not gold Helen Pitts Douglass.",
        ("musique", 2): "Gives 100388.2 or no area value, not gold 17.037 square miles.",
        ("musique", 3): "Does not supply the gold coalition: anti-slavery activists, modernizers, ex Whigs and ex Free Soilers.",
        ("musique", 5): "Names Torrie Wilson or Tori Spelling, not gold Diana DeGarmo.",
        ("musique", 6): "Answers Bleach, not gold OVA.",
        ("musique", 7): "Answers Puerto Rico/San Juan, not gold Minas Gerais.",
        ("musique", 8): "Names another castle or abstains, not gold Casa Loma.",
        ("musique", 9): "Does not answer gold cause blackmail.",
        ("2wiki", 0): "Answers American, not gold British.",
        ("2wiki", 1): "Names a film director/career history rather than gold workplace San Diego State University.",
        ("2wiki", 2): "Answers Berlin or Potsdam, not gold Zürich.",
        ("2wiki", 3): "Answers Driving Miss Wealthy/Daisy or makes a self-contradictory comparison, not gold Payaso.",
        ("2wiki", 4): "Answers The Fog Of War, not gold Jaśnie Pan Szofer.",
        ("2wiki", 5): "Answers Princess Helene of Prussia, not gold Infanta Maria Antonia of Portugal.",
        ("2wiki", 6): "Answers Egypt/Palestine, not gold India.",
        ("2wiki", 7): "Abstains or discusses another film instead of gold Mumbai University.",
        ("2wiki", 8): "Answers French/Hong Kong, not gold German.",
        ("2wiki", 9): "Abstains or gives an unrelated character, not gold Lydia Echevarría.",
    }
    reason = group_reasons[(dataset, group_id)]
    return {"manual_label": "clearly_wrong", "manual_reason": reason}


def _step_records(total: dict[str, Any], steps: int) -> list[dict[str, Any]]:
    memory = total.get("memory") or {}
    records: list[dict[str, Any]] = []
    for step in range(1, steps + 1):
        action = _json_object(total.get(f"action_predictor_{step}_response", ""))
        verifier = _json_object(total.get(f"verifier_{step}_response", ""))
        action_memory = memory.get(f"Action Step {step}") or {}
        records.append(
            {
                "step": step,
                "tool_name": action_memory.get("tool_name") or action.get("tool_name"),
                "planner_main_subgoal": _text(action.get("sub_goal"), 900),
                "planner_main_context": _text(action.get("context"), 900),
                "planner_main_justification": _text(action.get("justification"), 900),
                "key_tool_result_excerpt": _text(total.get(f"tool_result_{step}"), 1_200),
                "tool_execution_error": "Error in execute_tool_command" in str(total.get(f"tool_result_{step}", "")),
                "verifier_stop_signal": verifier.get("stop_signal"),
                "verifier_analysis_excerpt": _text(verifier.get("analysis") or verifier.get("raw"), 1_200),
            }
        )
    return records


def _stop_assessment(label: str, steps: list[dict[str, Any]]) -> tuple[str, str, bool | None]:
    final_signal = steps[-1]["verifier_stop_signal"] if steps else None
    if final_signal is not True:
        return (
            "step_budget_exhausted_after_verifier_continue",
            "The final verifier did not declare the memory complete; the two-step harness ended the trajectory.",
            False,
        )
    if label == "clearly_correct":
        return (
            "verifier_stop_true_with_gold_support",
            "The final tool result directly states the gold answer; no independent source was available, but its answer supports the final response.",
            False,
        )
    if label == "ambiguous_needs_semantic_judgment":
        return (
            "verifier_stop_true_with_ambiguous_support",
            "The tool result includes the gold entity but combines it with an unscoped alternate relation; it needs a semantic/time-scope judgment.",
            None,
        )
    return (
        "verifier_stop_true_after_non_gold_claim",
        "The verifier declared completion after a broad generator asserted a non-gold answer; the raw tool result did not contain evidence for the gold answer.",
        True,
    )


def _source_rows(manifest: dict[str, Any], dataset: str) -> dict[int, dict[str, Any]]:
    return {
        int(row["source_row_index"]): row
        for row in manifest["datasets"][dataset]["selected_rows"]
    }


def _record(dataset: str, path: Path, raw: dict[str, Any], candidate_index: int, source: dict[str, Any]) -> dict[str, Any]:
    total = raw["total_result"]
    steps = int(total.get("step_count") or 0)
    step_records = _step_records(total, steps)
    review = _review(dataset, int(raw["id"]), candidate_index)
    stop_reason, stop_evidence, early_stop = _stop_assessment(review["manual_label"], step_records)
    reward = float(raw["reward"])
    if review["manual_label"] == "clearly_correct":
        scorer_relation = "true_positive" if reward == 1.0 else "deterministic_false_negative"
    elif review["manual_label"] == "clearly_wrong":
        scorer_relation = "true_negative" if reward == 0.0 else "false_positive"
    else:
        scorer_relation = "unresolved_ambiguous"
    return {
        "dataset": dataset,
        "group_id": int(raw["id"]),
        "candidate_index": candidate_index,
        "source_row_index": int(raw["idx"]),
        "source_id": source.get("source_id"),
        "source_record_sha256": source.get("record_sha256"),
        "trajectory_path": str(path),
        "trajectory_sha256": _sha256(path),
        "question": raw["prompt"],
        "ground_truth": raw["groundtruth"],
        "candidate_answer": raw["answer_extracted"],
        "raw_reward": reward,
        "manual_label": review["manual_label"],
        "manual_reason": review["manual_reason"],
        "scorer_relation": scorer_relation,
        "steps": step_records,
        "step_count": steps,
        "ordered_tool_sequence": [step["tool_name"] for step in step_records],
        "stop_reason": stop_reason,
        "evidence_at_stop_assessment": stop_evidence,
        "verifier_likely_stopped_too_early": early_stop,
        "direct_output_excerpt": _text(total.get("direct_output"), 1_200),
    }


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(row["manual_label"] for row in records)
    relations = Counter(row["scorer_relation"] for row in records)
    first_tools = Counter((row["ordered_tool_sequence"] or ["none"])[0] for row in records)
    sequences = Counter(" -> ".join(row["ordered_tool_sequence"]) for row in records)
    stop_reasons = Counter(row["stop_reason"] for row in records)
    second = sum(row["step_count"] >= 2 for row in records)
    repeated = sum(
        len(row["ordered_tool_sequence"]) >= 2
        and len(set(row["ordered_tool_sequence"])) == 1
        for row in records
    )
    distinct = sum(len(set(row["ordered_tool_sequence"])) >= 2 for row in records)
    early = sum(row["verifier_likely_stopped_too_early"] is True for row in records)
    resolved = labels["clearly_correct"] + labels["clearly_wrong"]
    return {
        "rollout_count": len(records),
        "manual_labels": dict(labels),
        "scorer_relations": dict(relations),
        "clear_scoring_confusion": {
            "true_positive": relations["true_positive"],
            "true_negative": relations["true_negative"],
            "false_positive": relations["false_positive"],
            "false_negative": relations["deterministic_false_negative"],
            "ambiguous_excluded": labels["ambiguous_needs_semantic_judgment"],
            "agreement_rate_clear_only": (relations["true_positive"] + relations["true_negative"]) / resolved if resolved else None,
            "false_negative_rate_clear_correct_only": relations["deterministic_false_negative"] / labels["clearly_correct"] if labels["clearly_correct"] else None,
            "false_positive_rate_clear_wrong_only": relations["false_positive"] / labels["clearly_wrong"] if labels["clearly_wrong"] else None,
        },
        "first_tool_distribution": dict(first_tools),
        "second_step_count": second,
        "second_step_rate": second / len(records),
        "repeated_same_tool_count": repeated,
        "repeated_same_tool_rate": repeated / len(records),
        "two_or_more_distinct_tools_count": distinct,
        "two_or_more_distinct_tools_rate": distinct / len(records),
        "tool_sequence_distribution": dict(sequences),
        "stop_reason_distribution": dict(stop_reasons),
        "verifier_likely_stopped_too_early_count": early,
        "verifier_likely_stopped_too_early_rate": early / len(records),
        "tool_execution_error_count": sum(
            step["tool_execution_error"] for row in records for step in row["steps"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("log/2026-08-29_multihop_manual_audit_results.json"),
    )
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for dataset, root in ROOTS.items():
        assert root.is_dir(), root
        sources = _source_rows(manifest, dataset)
        groups: dict[int, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
        for path in root.rglob("*.json"):
            raw = json.loads(path.read_text())
            groups[int(raw["id"])].append((path, raw))
        assert len(groups) == 10, (dataset, len(groups))
        rows: list[dict[str, Any]] = []
        for group_id, trajectories in sorted(groups.items()):
            ordered = sorted(trajectories, key=lambda item: (item[1].get("timestamp", ""), str(item[0])))
            assert len(ordered) == 4, (dataset, group_id, len(ordered))
            for candidate_index, (path, raw) in enumerate(ordered):
                source = sources[int(raw["idx"])]
                rows.append(_record(dataset, path, raw, candidate_index, source))
        by_dataset[dataset] = rows

    all_records = [row for rows in by_dataset.values() for row in rows]
    payload = {
        "schema_version": 1,
        "audit_mode": "offline_manual_observational_no_model_no_api",
        "input_manifest": str(MANIFEST),
        "input_manifest_sha256": _sha256(MANIFEST),
        "raw_rollout_roots": {name: str(path) for name, path in ROOTS.items()},
        "manual_label_standard": {
            "clearly_correct": "Semantic answer clearly matches the supplied gold answer.",
            "clearly_wrong": "Semantic answer clearly fails to match the supplied gold answer.",
            "ambiguous_needs_semantic_judgment": "Answer contains relevant material but time scope, multiplicity, or reference semantics prevent a safe binary label without an independent semantic judgment.",
        },
        "datasets": {
            name: {"summary": _summarize(rows), "rollouts": rows}
            for name, rows in by_dataset.items()
        },
        "overall_summary": _summarize(all_records),
        "audit_limitations": [
            "This is a direct human/Codex reading of the persisted trajectories against their supplied gold answers; it does not independently verify every benchmark fact.",
            "The two Charles-University plural-employer answers remain ambiguous rather than being force-labeled.",
            "Tool-result sufficiency is judged against the gold answer and recorded raw text, not as a causal intervention on planner or verifier.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {args.output} ({len(all_records)} rollouts)")


if __name__ == "__main__":
    main()
