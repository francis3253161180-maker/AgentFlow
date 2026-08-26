#!/usr/bin/env python3
"""Offline reward audit for the 20260825_231408 mini20 training rollouts.

The script reads only saved rollout JSON, the source parquet, and (optionally)
the training log.  It never calls an LLM scorer.  The manual labels below are
the audit's explicit semantic review of this fixed 20-example dataset; they
are not presented as a replacement reward implementation.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.pop("AGENTFLOW_USE_LLM_SCORER", None)
from train.utils import deterministic_fallback_score


def fallback_candidates(value: Any) -> set[str]:
    """Mirror train/utils.py's current local fallback scorer exactly."""
    text = str(value).lower()
    text = re.sub(r"<answer>|</answer>|\\boxed\s*", "", text)
    text = text.replace("\\left", "").replace("\\right", "")
    text = re.sub(r"\\(?:d?frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"\1/\2", text)
    text = re.sub(r"\\text\s*\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"\s+", "", text)
    forms = {text}
    if "=" in text:
        forms.add(text.rsplit("=", 1)[-1])
    return {re.sub(r"[^a-z0-9./+-]", "", form) for form in forms}


def legacy_fallback_score(groundtruth: Any, answer_extracted: Any) -> bool:
    """Mirror the pre-fix train/utils.py fallback without network access."""
    answer_forms = fallback_candidates(answer_extracted)
    truth_forms = fallback_candidates(groundtruth)
    if answer_forms & truth_forms:
        return True
    answer_tokens = set(re.findall(r"[-+]?\d+(?:\.\d+)?(?:/[-+]?\d+)?", str(answer_extracted)))
    truth_tokens = set(re.findall(r"[-+]?\d+(?:\.\d+)?(?:/[-+]?\d+)?", str(groundtruth)))
    return bool(answer_tokens & truth_tokens)


# Keep the old name for callers of the first audit script version.
fallback_score = legacy_fallback_score


# Explicit semantic review for the 20 source examples.  IDs in the first set
# are correct in every saved answer; IDs in the second set are incorrect in
# every saved answer.  The two conditional IDs have visibly different saved
# answers and are split below.
CORRECT_ALL = {
    66972,   # Daimler-Benz
    34128,   # 22 episodes
    80293,   # Yes
    116769,  # residue = 6
    108190,  # curvature = 1/2
    121873,  # John McCrae
    129667,  # bust
    68772,   # x + 1
    90379,   # Oscar the Grouch
    147614,  # Chicago's Grant Park
    132919,  # Yes
    138626,  # Yes
    5991,    # residue = 1/6
    67873,   # x0 = 1/3
    51978,   # 1 + sqrt(2), including the rendered form
    35924,   # Democratic-Republicans / Jeffersonian democracy
}
INCORRECT_ALL = {
    98332,   # Sophie Sumner / unidentified, not Louise Glover
    6418,    # March 31, not the dataset answer April 7
}


def semantic_correct(record: dict[str, Any]) -> bool:
    """Return the fixed, documented semantic review label independent of reward."""
    sample_id = int(record["id"])
    answer = str(record["answer_extracted"])

    if sample_id == 61526:
        # The first answer gives the album date (wrong for this benchmark's
        # target); the second also gives the title-track single date (right).
        correct = "october 3, 2017" in answer.lower()
    elif sample_id == 101323:
        correct = bool(re.search(r"\b13\b", answer)) and not bool(re.search(r"\b20\b", answer))
    elif sample_id in CORRECT_ALL:
        correct = True
    elif sample_id in INCORRECT_ALL:
        correct = False
    else:
        raise AssertionError(f"No semantic review label for dataset id {sample_id}")

    return correct


def semantic_verdict(record: dict[str, Any], reward: float | None = None) -> str:
    """Return TP/TN/FP/FN for a supplied or saved binary reward."""
    correct = semantic_correct(record)
    if reward is None:
        reward = float(record["reward"])

    if correct and reward == 1.0:
        return "TP"
    if correct and reward == 0.0:
        return "FN"
    if not correct and reward == 0.0:
        return "TN"
    if not correct and reward == 1.0:
        return "FP"
    raise AssertionError(f"Unexpected reward {reward!r} for id {sample_id}")


def load_source_map(dataset_path: Path) -> dict[int, dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise SystemExit("Run with the AgentFlow environment (pandas/pyarrow required)") from exc

    frame = pd.read_parquet(dataset_path)
    return {int(row.id): {"source": row.source, "question": row.question} for row in frame.itertuples()}


def parse_metric_lines(log_path: Path | None) -> list[dict[str, Any]]:
    if log_path is None or not log_path.exists():
        return []
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    key_patterns = {
        "actor/entropy_loss": r"actor/entropy_loss:([^ ]+)",
        "actor/pg_loss": r"actor/pg_loss:(?:np\.float64\()?([^\) ]+)",
        "actor/grad_norm": r"actor/grad_norm:(?:np\.float64\()?([^\) ]+)",
        "critic/rewards/mean": r"critic/rewards/mean:([^ ]+)",
        "critic/rewards/min": r"critic/rewards/min:([^ ]+)",
        "critic/rewards/max": r"critic/rewards/max:([^ ]+)",
        "critic/advantages/mean": r"critic/advantages/mean:([^ ]+)",
        "critic/advantages/min": r"critic/advantages/min:([^ ]+)",
        "critic/advantages/max": r"critic/advantages/max:([^ ]+)",
        "perf/max_memory_allocated_gb": r"perf/max_memory_allocated_gb:(?:np\.float64\()?([^\) ]+)",
        "timing_s/old_log_prob": r"timing_s/old_log_prob:([^ ]+)",
        "timing_s/update_actor": r"timing_s/update_actor:([^ ]+)",
        "training/global_step": r"training/global_step:([^ ]+)",
    }
    metrics: list[dict[str, Any]] = []
    for line in log_path.read_text(errors="replace").splitlines():
        line = ansi.sub("", line)
        step_match = re.search(r"\bstep:(\d+)\s+-", line)
        if not step_match:
            continue
        row: dict[str, Any] = {"step": int(step_match.group(1))}
        for key, pattern in key_patterns.items():
            match = re.search(pattern, line)
            if match:
                raw = match.group(1)
                try:
                    row[key] = float(raw)
                except ValueError:
                    row[key] = raw
        metrics.append(row)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--train-log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_map = load_source_map(args.dataset)
    json_paths = sorted((args.run_root / "train").glob("**/rollout_*.json"))
    records: list[dict[str, Any]] = []
    for path in json_paths:
        raw = json.loads(path.read_text())
        sample = source_map[int(raw["id"])]
        answer = str(raw["answer_extracted"])
        legacy_fallback = legacy_fallback_score(raw["groundtruth"], answer)
        fixed_fallback = deterministic_fallback_score(raw["groundtruth"], answer)
        # train/rollout.py calls compute_score(question, groundtruth, answer)
        # directly.  train/utils.py::eval has a swapped call, so record both
        # to distinguish that bug from the scorer's matching limitation.
        swapped_fallback = legacy_fallback_score(answer, raw["groundtruth"])
        verdict = semantic_verdict(raw, float(raw["reward"]))
        try:
            rel_path = path.relative_to(Path.cwd())
        except ValueError:
            rel_path = path
        records.append(
            {
                "path": str(rel_path),
                "step_dir": int(re.search(r"/train/step_(\d+)/", str(path)).group(1)),
                "idx": raw.get("idx"),
                "id": int(raw["id"]),
                "source": str(sample["source"]),
                "question": str(raw["prompt"]),
                "groundtruth": str(raw["groundtruth"]),
                "answer_extracted": answer,
                "reward": float(raw["reward"]),
                "legacy_fallback": bool(legacy_fallback),
                "current_fallback": bool(legacy_fallback),
                "fixed_fallback": bool(fixed_fallback),
                "swapped_fallback": bool(swapped_fallback),
                "scorer_matches_record": bool(float(raw["reward"]) == float(legacy_fallback)),
                "fixed_reward_matches_semantic": bool(float(fixed_fallback) == semantic_correct(raw)),
                "semantic_verdict": verdict,
                "fixed_semantic_verdict": semantic_verdict(raw, float(fixed_fallback)),
                "answer_excerpt": re.sub(r"\s+", " ", answer).strip()[:320],
            }
        )

    counts = Counter(r["semantic_verdict"] for r in records)
    fixed_counts = Counter(r["fixed_semantic_verdict"] for r in records)
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    fixed_by_source: dict[str, Counter[str]] = defaultdict(Counter)
    for row in records:
        by_source[row["source"]][row["semantic_verdict"]] += 1
        fixed_by_source[row["source"]][row["fixed_semantic_verdict"]] += 1
    reward_counts = Counter(int(r["reward"]) for r in records)
    fixed_reward_counts = Counter(int(r["fixed_fallback"]) for r in records)
    newly_positive = [
        {
            "path": r["path"],
            "source": r["source"],
            "id": r["id"],
            "groundtruth": r["groundtruth"],
            "answer_extracted": r["answer_extracted"],
            "pre_fix_reward": r["reward"],
            "post_fix_reward": float(r["fixed_fallback"]),
            "post_fix_verdict": r["fixed_semantic_verdict"],
        }
        for r in records
        if not r["legacy_fallback"] and r["fixed_fallback"]
    ]
    output = {
        "audit": "2026-08-26 reward audit",
        "scope": "train rollouts only; validation is intentionally excluded",
        "run_root": str(args.run_root),
        "dataset": str(args.dataset),
        "record_count": len(records),
        "unique_dataset_ids": len({r["id"] for r in records}),
        "reward_counts": {str(k): v for k, v in sorted(reward_counts.items())},
        "semantic_counts": dict(sorted(counts.items())),
        "fixed_semantic_counts": dict(sorted(fixed_counts.items())),
        "by_source": {source: dict(sorted(counter.items())) for source, counter in sorted(by_source.items())},
        "fixed_by_source": {source: dict(sorted(counter.items())) for source, counter in sorted(fixed_by_source.items())},
        "fixed_reward_counts": {str(k): v for k, v in sorted(fixed_reward_counts.items())},
        "scorer_matches_saved_reward": all(r["scorer_matches_record"] for r in records),
        "scorer_mismatch_count": sum(not r["scorer_matches_record"] for r in records),
        "swapped_fallback_diff_count": sum(r["current_fallback"] != r["swapped_fallback"] for r in records),
        "fixed_matches_semantic_count": sum(r["fixed_reward_matches_semantic"] for r in records),
        "fixed_mismatch_semantic_count": sum(not r["fixed_reward_matches_semantic"] for r in records),
        "newly_positive_count": len(newly_positive),
        "newly_positive": newly_positive,
        "metric_lines": parse_metric_lines(args.train_log),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: output[k] for k in output if k != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
