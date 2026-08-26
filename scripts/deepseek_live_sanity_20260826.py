#!/usr/bin/env python3
"""Small real-DeepSeek sanity check for the production hybrid scorer.

This script intentionally emits only case metadata, booleans, timings, and
exception type names.  It never prints prompts, responses, cache contents, or
credentials.  The cache is temporary and is removed at process exit.
"""

from __future__ import annotations

import json
import os
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

from train.reward_judge import (
    DeepSeekSemanticJudge,
    HybridRewardScorer,
    RewardJudgeCache,
    parse_judge_response,
)
from train.utils import deterministic_decision
from test.hybrid_reward_cases import SYNTHETIC_CASES


REPO_ROOT = Path(__file__).resolve().parents[1]
REPEAT_NAMES = {
    "date_corrected_to_wrong",
    "date_wrong_corrected_to_right",
    "yes_to_no_self_correction",
}
UNIQUE_COMPLEX_NAMES = {
    "no_to_yes_self_correction",
    "multiple_candidate_entities",
    "entity_mentioned_then_rejected",
    "thought_x_actually_y",
    "near_but_not_in_entity",
}
DETERMINISTIC_NAMES = {
    "final_marker_overrides_earlier_reasoning",
    "fraction_local_proof",
    "integer_local_mismatch",
}


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def latency_summary(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "average_seconds": statistics.mean(values) if values else None,
        "median_seconds": statistics.median(values) if values else None,
        "p95_seconds": percentile(values, 0.95),
    }


def safe_call(judge, question: str, groundtruth: str, answer: str) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        verdict = parse_judge_response(judge(question, groundtruth, answer))
        return {
            "actual": bool(verdict.true_false),
            "error": None,
            "latency_seconds": time.perf_counter() - started,
        }
    except Exception as exc:
        return {
            "actual": None,
            "error": type(exc).__name__,
            "latency_seconds": time.perf_counter() - started,
        }


def load_real_rollout() -> dict[str, Any] | None:
    path = REPO_ROOT / "log/2026-08-26_reward_scorer_fix_results.json"
    if not path.exists():
        return None
    records = json.loads(path.read_text(encoding="utf-8"))["records"]
    for record in records:
        if record.get("source") != "nq":
            continue
        expected = str(record.get("fixed_semantic_verdict")) in {"TP", "FN"}
        decision = deterministic_decision(record["groundtruth"], record["answer_extracted"])
        if decision.value is None:
            return {
                "case_id": "real_rollout_nq_open_answer",
                "case_type": "real_rollout_nq",
                "question": str(record["question"]),
                "groundtruth": str(record["groundtruth"]),
                "answer": str(record["answer_extracted"]),
                "expected": expected,
                "expected_route": "judge",
            }
    return None


def main() -> None:
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise SystemExit("DEEPSEEK_API_KEY is missing; live check not attempted")

    judge = DeepSeekSemanticJudge.from_environment()
    selected = [
        case
        for case in SYNTHETIC_CASES
        if case.name in REPEAT_NAMES | UNIQUE_COMPLEX_NAMES | DETERMINISTIC_NAMES
    ]
    real = load_real_rollout()
    cases: list[dict[str, Any]] = [
        {
            "case_id": case.name,
            "case_type": "synthetic_adversarial",
            "question": f"Synthetic live sanity question for case {case.name}.",
            "groundtruth": case.groundtruth,
            "answer": case.answer,
            "expected": case.expected,
            "expected_route": case.expected_route,
        }
        for case in selected
    ]
    if real is not None:
        cases.append(real)

    raw_repeat_results: dict[str, list[dict[str, Any]]] = {}
    uncached_latencies: list[float] = []
    raw_consistency: dict[str, Any] = {}

    # Three direct uncached calls per complex case measure raw provider
    # stability.  They intentionally bypass the production cache.
    for case in cases:
        if case["case_id"] not in REPEAT_NAMES:
            continue
        results = [
            safe_call(judge, case["question"], case["groundtruth"], case["answer"])
            for _ in range(3)
        ]
        raw_repeat_results[case["case_id"]] = results
        uncached_latencies.extend(
            result["latency_seconds"] for result in results if result["error"] is None
        )
        verdicts = [result["actual"] for result in results if result["error"] is None]
        raw_consistency[case["case_id"]] = {
            "verdicts": verdicts,
            "all_successful_verdicts_equal": len(set(verdicts)) <= 1 if verdicts else False,
            "errors": [result["error"] for result in results if result["error"]],
        }

    with tempfile.TemporaryDirectory(prefix="agentflow_deepseek_live_") as cache_dir:
        cache = RewardJudgeCache(cache_dir)
        scorer = HybridRewardScorer(judge=judge, cache=cache, enabled=True, judge_name=judge.model)
        case_results = []
        cached_latencies: list[float] = []
        for case in cases:
            started = time.perf_counter()
            result = scorer.score_with_metadata(
                case["question"], case["groundtruth"], case["answer"]
            )
            elapsed = time.perf_counter() - started
            if result.route == "judge" and not result.cache_hit:
                uncached_latencies.append(elapsed)
            case_results.append(
                {
                    "case_id": case["case_id"],
                    "case_type": case["case_type"],
                    "expected": case["expected"],
                    "actual": result.score,
                    "correct": result.score == case["expected"],
                    "route": result.route,
                    "judge_called": result.route == "judge",
                    "cache_hit": result.cache_hit,
                    "latency_seconds": elapsed,
                    "error": result.judge_error,
                    "expected_route": case["expected_route"],
                }
            )

            # Explicit second call for every repeated case demonstrates the
            # stable cache hit and its latency separately from raw probes.
            if case["case_id"] in REPEAT_NAMES:
                started = time.perf_counter()
                cached_result = scorer.score_with_metadata(
                    case["question"], case["groundtruth"], case["answer"]
                )
                cached_elapsed = time.perf_counter() - started
                cached_latencies.append(cached_elapsed)
                case_results.append(
                    {
                        "case_id": case["case_id"] + "__cached_repeat",
                        "case_type": case["case_type"],
                        "expected": case["expected"],
                        "actual": cached_result.score,
                        "correct": cached_result.score == case["expected"],
                        "route": cached_result.route,
                        "judge_called": cached_result.route == "judge",
                        "cache_hit": cached_result.cache_hit,
                        "latency_seconds": cached_elapsed,
                        "error": cached_result.judge_error,
                        "expected_route": "judge_cache",
                    }
                )

        cache_files = list(Path(cache_dir).iterdir())
        cache_json = [path for path in cache_files if path.suffix == ".json"]
        cache_records = [json.loads(path.read_text(encoding="utf-8")) for path in cache_json]
        cache_schema_only = all(
            set(record) == {"schema_version", "key", "true_false"}
            and type(record.get("schema_version")) is int
            and isinstance(record.get("key"), str)
            and type(record.get("true_false")) is bool
            for record in cache_records
        )
        # The cache contract has no fields capable of storing the input
        # strings. This schema check avoids false positives for short values
        # such as "47" appearing naturally in a hash or boolean field.
        cache_leak = not cache_schema_only
        stats = scorer.stats.snapshot()

    primary = [row for row in case_results if not row["case_id"].endswith("__cached_repeat")]
    successful_primary = [row for row in primary if row["error"] is None]
    accuracy = (
        sum(row["correct"] for row in successful_primary) / len(successful_primary)
        if successful_primary
        else None
    )
    output = {
        "status": "completed",
        "model": judge.model,
        "temperature": 0,
        "case_count": len(cases),
        "api_calls": stats["judge_calls"] + sum(
            len(results) for results in raw_repeat_results.values()
        ),
        "primary_accuracy": accuracy,
        "primary_successes": len(successful_primary),
        "primary_errors": len(primary) - len(successful_primary),
        "case_results": case_results,
        "raw_repeat_results": raw_consistency,
        "raw_repeat_consistency_all_cases": all(
            item["all_successful_verdicts_equal"] for item in raw_consistency.values()
        ),
        "uncached_latency": latency_summary(uncached_latencies),
        "cached_latency": latency_summary(cached_latencies),
        "scorer_stats": stats,
        "cache_validation": {
            "cache_json_files": len(cache_json),
            "cache_contains_raw_input": cache_leak,
            "cache_key_format_sha256": all(len(path.stem) == 64 for path in cache_json),
            "cache_directory_temporary": True,
        },
        "api_error_types": sorted(
            {
                result["error"]
                for result in case_results
                if result["error"]
            }
            | {
                error
                for results in raw_repeat_results.values()
                for result in results
                for error in [result["error"]]
                if error
            }
        ),
    }
    output_path = Path(os.getenv(
        "AGENTFLOW_DEEPSEEK_LIVE_OUTPUT",
        str(REPO_ROOT / "log/2026-08-26_deepseek_live_sanity_results.json"),
    ))
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
