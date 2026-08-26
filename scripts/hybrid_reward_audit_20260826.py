#!/usr/bin/env python3
"""Offline regression/audit for the hybrid reward scorer.

The semantic judge used here is a deterministic mock whose verdicts come from
the already-reviewed labels in the committed seen audit and the independent
synthetic fixture.  No network call is made unless ``--live-count`` is given.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from train.reward_judge import (
    DeepSeekSemanticJudge,
    HybridRewardScorer,
    RewardJudgeCache,
)

from test.hybrid_reward_cases import SYNTHETIC_CASES


class MockSemanticJudge:
    def __init__(self, rows: list[dict[str, Any]]):
        self.verdicts = {
            (str(row["question"]), str(row["groundtruth"]), str(row["answer"])): bool(row["correct"])
            for row in rows
        }
        self.calls: list[dict[str, str]] = []

    def __call__(self, question: str, groundtruth: str, answer: str) -> dict[str, Any]:
        key = (str(question), str(groundtruth), str(answer))
        if key not in self.verdicts:
            raise KeyError("mock input not found")
        verdict = self.verdicts[key]
        self.calls.append(
            {
                "question_sha256": RewardJudgeCache.key(question, "", "")[:16],
                "groundtruth_sha256": RewardJudgeCache.key("", groundtruth, "")[:16],
                "answer_sha256": RewardJudgeCache.key("", "", answer)[:16],
            }
        )
        return {"true_false": verdict, "analysis": "offline mock"}


def confusion(correct: list[bool], predicted: list[bool]) -> dict[str, int]:
    counts = Counter()
    for expected, actual in zip(correct, predicted):
        if expected and actual:
            counts["TP"] += 1
        elif expected and not actual:
            counts["FN"] += 1
        elif not expected and actual:
            counts["FP"] += 1
        else:
            counts["TN"] += 1
    return {key: counts[key] for key in ("TP", "TN", "FP", "FN")}


def rates(counts: dict[str, int]) -> dict[str, float]:
    positives = counts["TP"] + counts["FN"]
    negatives = counts["TN"] + counts["FP"]
    return {
        "fn_rate": counts["FN"] / positives if positives else 0.0,
        "fp_rate": counts["FP"] / negatives if negatives else 0.0,
        "reward_positive": counts["TP"] + counts["FP"],
        "reward_negative": counts["TN"] + counts["FN"],
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = [bool(row["correct"] ) for row in rows]
    predicted = [bool(row["predicted"] ) for row in rows]
    counts = confusion(correct, predicted)
    return {"count": len(rows), "confusion": counts, **rates(counts)}


def seen_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for record in results["records"]:
        # The previous audit's fixed_semantic_verdict is the independent
        # semantic review label, not the old reward.  It is used only as the
        # mock judge oracle for this offline routing/regression run.
        fixed_verdict = str(record["fixed_semantic_verdict"])
        correct = fixed_verdict in {"TP", "FN"}
        rows.append(
            {
                "id": int(record["id"]),
                "source": str(record["source"]),
                "question": str(record["question"]),
                "groundtruth": str(record["groundtruth"]),
                "answer": str(record["answer_extracted"]),
                "correct": correct,
            }
        )
    return rows


def run_mock(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="agentflow_hybrid_audit_") as cache_dir:
        mock = MockSemanticJudge(rows)
        scorer = HybridRewardScorer(
            judge=mock,
            cache=RewardJudgeCache(cache_dir),
            enabled=True,
            judge_name="offline-mock",
        )
        scored = []
        for row in rows:
            result = scorer.score_with_metadata(row["question"], row["groundtruth"], row["answer"])
            scored.append(
                {
                    "name": row.get("name", str(row.get("id", "synthetic"))),
                    "source": row.get("source", "synthetic"),
                    "correct": row["correct"],
                    "predicted": result.score,
                    "route": result.route,
                    "reason": result.reason,
                    "judge_error": result.judge_error,
                }
            )
        metadata = {"scorer": scorer.stats.snapshot(), "mock_calls": len(mock.calls)}
        judge_rows = [row for row in scored if row["route"] == "judge"]
        metadata["judge_route_input_characters"] = sum(
            len(str(row["question"])) + len(str(row["groundtruth"])) + len(str(row["answer"]))
            for row in rows
            if any(
                scored_row["name"] == row.get("name", str(row.get("id", "synthetic")))
                and scored_row["route"] == "judge"
                for scored_row in judge_rows
            )
        )
        metadata["judge_route_input_token_estimate_at_4_chars"] = (
            metadata["judge_route_input_characters"] / 4.0
        )
        return scored, metadata


def run_live(seen: list[dict[str, Any]], count: int) -> dict[str, Any]:
    if count <= 0:
        return {"status": "not_requested", "calls": 0}
    if not os.getenv("DEEPSEEK_API_KEY"):
        return {"status": "not_available", "reason": "DEEPSEEK_API_KEY_absent", "calls": 0}

    try:
        judge = DeepSeekSemanticJudge.from_environment()
    except Exception as exc:
        return {
            "status": "not_available",
            "reason": type(exc).__name__,
            "calls": 0,
        }

    # Keep this deliberately tiny and use two reviewed NQ records plus two
    # independent adversarial records when available.
    samples = [row for row in seen if row.get("source") == "nq"][: max(0, count // 2)]
    samples.extend(
        {
            "question": f"synthetic-{case.name}",
            "groundtruth": case.groundtruth,
            "answer": case.answer,
        }
        for case in SYNTHETIC_CASES[: max(0, count - len(samples))]
    )
    samples = samples[:count]
    latencies = []
    verdicts = []
    errors = []
    for row in samples:
        started = time.perf_counter()
        try:
            verdict = judge(row["question"], row["groundtruth"], row["answer"])
            verdicts.append(bool(verdict.true_false))
        except Exception as exc:
            errors.append(type(exc).__name__)
        latencies.append(time.perf_counter() - started)
    return {
        "status": "completed" if not errors else "completed_with_errors",
        "model": judge.model,
        "calls": len(samples),
        "successful_verdicts": len(verdicts),
        "errors": errors,
        "average_latency_seconds": sum(latencies) / len(latencies) if latencies else None,
        "latencies_seconds": latencies,
        "verdicts": verdicts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seen-results",
        type=Path,
        default=Path("log/2026-08-26_reward_scorer_fix_results.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("log/2026-08-26_hybrid_reward_audit_results.json"),
    )
    parser.add_argument("--live-count", type=int, default=0)
    args = parser.parse_args()

    previous = json.loads(args.seen_results.read_text(encoding="utf-8"))
    seen = seen_rows(previous)
    synthetic = [
        {
            "name": case.name,
            "source": "synthetic",
            "question": f"synthetic question: {case.name}",
            "groundtruth": case.groundtruth,
            "answer": case.answer,
            "correct": case.expected,
            "expected_route": case.expected_route,
        }
        for case in SYNTHETIC_CASES
    ]

    seen_scored, seen_meta = run_mock(seen)
    synthetic_scored, synthetic_meta = run_mock(synthetic)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in seen_scored:
        by_source[row["source"]].append(row)

    output = {
        "audit": "2026-08-26 hybrid reward scorer offline regression",
        "seen_source": str(args.seen_results),
        "seen": {
            "overall": summarize(seen_scored),
            "by_source": {source: summarize(rows) for source, rows in sorted(by_source.items())},
            "pre_hybrid_saved_reward": {
                "confusion": previous["semantic_counts"],
                "reward_positive": previous["reward_counts"].get("1", 0),
                "reward_negative": previous["reward_counts"].get("0", 0),
            },
            "routing": seen_meta,
        },
        "synthetic": {
            "case_count": len(synthetic_scored),
            "overall": summarize(synthetic_scored),
            "routing": synthetic_meta,
            "expected_route_mismatches": [
                row["name"]
                for row, case in zip(synthetic_scored, SYNTHETIC_CASES)
                if row["route"] != case.expected_route
            ],
            "cases": synthetic_scored,
        },
        "live": run_live(seen, args.live_count),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
