#!/usr/bin/env python3
"""Aggregate the saved 2048/8192 rollout-only evidence without new requests."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path
from statistics import mean, median


def percentile(values: list[int], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q
    low, high = int(position), min(int(position) + 1, len(values) - 1)
    fraction = position - low
    return values[low] + (values[high] - values[low]) * fraction


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence-dir", type=Path, required=True)
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-response", type=int, default=2048)
    ap.add_argument("--max-model-len", type=int, default=8192)
    args = ap.parse_args()
    files = sorted(args.evidence_dir.glob("rollout_*.json"))
    response_lengths: list[int] = []
    prompt_lengths: list[int] = []
    finish_reasons: collections.Counter[str] = collections.Counter()
    eos_or_stop = 0
    context_errors = 0
    rewards: collections.Counter[str] = collections.Counter()
    groups: dict[str, list[dict]] = collections.defaultdict(list)
    final_response_hashes: list[str] = []
    for path in files:
        evidence = json.loads(path.read_text(encoding="utf-8"))
        rollout = evidence["rollout"]
        group_id = str(evidence["original_sample"].get("data_id"))
        rewards[str(float(rollout.get("final_reward", 0.0) or 0.0))] += 1
        serialized = json.dumps(evidence, ensure_ascii=False)
        context_errors += int("maximum context length" in serialized or "BadRequestError" in serialized)
        triplets = rollout.get("triplets") or []
        for span in rollout.get("trace") or []:
            reason = (span.get("attributes") or {}).get("gen_ai.completion.0.finish_reason")
            if reason:
                finish_reasons[str(reason)] += 1
        for triplet in triplets:
            response = triplet.get("response") or {}
            prompt = triplet.get("prompt") or {}
            response_ids = response.get("response_token_ids") or response.get("token_ids") or []
            prompt_ids = prompt.get("token_ids") or []
            response_lengths.append(len(response_ids))
            prompt_lengths.append(len(prompt_ids))
            final_response_hashes.append(hashlib.sha256(json.dumps(response_ids).encode()).hexdigest())
        groups[group_id].append({"path": path.name, "reward": rollout.get("final_reward", 0.0)})
        # The current Qwen/vLLM trace exposes stop as finish_reason; EOS token
        # IDs are not reliably serialized in this evidence schema.
    eos_or_stop = finish_reasons.get("stop", 0)
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    stats = {
        "count": len(response_lengths),
        "mean": mean(response_lengths) if response_lengths else None,
        "p50": percentile(response_lengths, 0.50),
        "p75": percentile(response_lengths, 0.75),
        "p90": percentile(response_lengths, 0.90),
        "p95": percentile(response_lengths, 0.95),
        "p99": percentile(response_lengths, 0.99),
        "max": max(response_lengths) if response_lengths else None,
        "equal_cap_count": sum(value == args.max_response for value in response_lengths),
        "equal_cap_rate": (sum(value == args.max_response for value in response_lengths) / len(response_lengths)) if response_lengths else None,
    }
    prompt_stats = {
        "count": len(prompt_lengths),
        "mean": mean(prompt_lengths) if prompt_lengths else None,
        "p50": percentile(prompt_lengths, 0.50),
        "p95": percentile(prompt_lengths, 0.95),
        "max": max(prompt_lengths) if prompt_lengths else None,
    }
    result = {
        "schema_version": 1,
        "status": "ok",
        "evidence_dir": str(args.evidence_dir),
        "evidence_file_count": len(files),
        "group_count": len(groups),
        "selection_sha256": sha256(args.selection),
        "selected_group_count": selection.get("selected_group_count"),
        "requested_max_response_tokens": args.max_response,
        "requested_max_model_len": args.max_model_len,
        "response_tokens": stats,
        "prompt_tokens": prompt_stats,
        "finish_reason_counts": dict(finish_reasons),
        "eos_or_stop_count": eos_or_stop,
        "eos_or_stop_note": "finish_reason=stop is available; explicit EOS token IDs are not serialized reliably",
        "context_overflow_or_400_events": context_errors,
        "reward_counts_observed": dict(rewards),
        "scorer_routing": {
            "deterministic": len(files),
            "judge_fallback": 0,
            "cache_hit": 0,
            "routing_basis": "run configuration explicitly disabled external judge and selected local deterministic scorer; per-event route telemetry is not embedded in Rollout evidence",
        },
        "answer_exact_duplicate_rate": 1.0 - len(set(final_response_hashes)) / len(final_response_hashes) if final_response_hashes else None,
        "raw_artifacts_not_embedded": True,
        "evidence_file_sha256": {path.name: sha256(path) for path in files},
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "evidence_file_count", "group_count", "response_tokens", "prompt_tokens", "finish_reason_counts", "context_overflow_or_400_events")}, sort_keys=True))


if __name__ == "__main__":
    main()
