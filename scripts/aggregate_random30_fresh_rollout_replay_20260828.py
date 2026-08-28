#!/usr/bin/env python3
"""Aggregate the local random30 fresh rollout evidence without generating text."""
from __future__ import annotations
import argparse, hashlib, json, statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def percentile(values: list[int], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return float(values[0])
    pos = (len(values) - 1) * p
    lo, hi = int(pos), min(len(values) - 1, int(pos) + 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)

def find_keys(value: Any, names: set[str], found: list[Any]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in names:
                found.append(item)
            find_keys(item, names, found)
    elif isinstance(value, list):
        for item in value:
            find_keys(item, names, found)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-response", type=int, default=1024)
    ap.add_argument("--max-model-len", type=int, default=4096)
    args = ap.parse_args()
    files = sorted(args.evidence_dir.glob("rollout_*.json"))
    records = []
    for path in files:
        item = json.loads(path.read_text(encoding="utf-8"))
        rollout = item.get("rollout", {})
        sample = item.get("original_sample", {})
        triplets = rollout.get("triplets") or []
        records.append((path, rollout, sample, triplets))
    groups: dict[str, list[tuple[Path, dict, dict, list]]] = defaultdict(list)
    for item in records:
        groups[str(item[2].get("data_id", ""))].append(item)
    rewards = []
    response_lengths = []
    prompt_lengths = []
    context_lengths = []
    finish_reasons = []
    eos_flags = []
    answer_texts = []
    for path, rollout, sample, triplets in records:
        rewards.append(float(rollout.get("final_reward", 0.0)))
        for triplet in triplets:
            pids = triplet.get("prompt", {}).get("token_ids", []) or []
            rids = triplet.get("response", {}).get("token_ids", []) or []
            prompt_lengths.append(len(pids))
            response_lengths.append(len(rids))
            context_lengths.append(len(pids) + len(rids))
            found = []
            find_keys(triplet, {"finish_reason", "finish_reasons"}, found)
            finish_reasons.extend(str(x) for x in found)
            found_eos = []
            find_keys(triplet, {"eos", "eos_reached", "eos_token"}, found_eos)
            eos_flags.extend(bool(x) for x in found_eos)
        # Some future writers may include final answer text in metadata.
        for key in ("answer_extracted", "final_answer", "direct_output"):
            if key in rollout:
                answer_texts.append(str(rollout[key]))
    cap_hits = sum(x >= args.max_response for x in response_lengths)
    contexts_over = sum(x > args.max_model_len for x in context_lengths)
    exact_dup = None
    if answer_texts:
        exact_dup = 1.0 - len(set(answer_texts)) / len(answer_texts)
    group_rows = []
    for gid, items in sorted(groups.items()):
        group_rewards = [float(x[1].get("final_reward", 0.0)) for x in items]
        group_rows.append({
            "group_id": gid,
            "count": len(items),
            "reward_vector": group_rewards,
            "unique_rewards": len(set(group_rewards)),
            "valid": all(bool(x[3]) and all(bool(t.get("prompt", {}).get("token_ids")) and bool(t.get("response", {}).get("token_ids")) for t in x[3]) for x in items),
            "source_row_indices": sorted({str(x[2].get("extra_info", {}).get("source_row_index", x[2].get("extra_info", {}).get("idx", ""))) for x in items}),
        })
    route_counts = Counter()
    for item in records:
        for value in (item[1].get("metadata", {}), item[0].name):
            text = json.dumps(value, ensure_ascii=False).lower()
            if "judge" in text or "deepseek" in text:
                route_counts["judge_or_external_marker"] += 1
            else:
                route_counts["deterministic_or_unreported"] += 1
                break
    payload = {
        "schema_version": 1,
        "evidence_dir": str(args.evidence_dir),
        "evidence_file_count": len(files),
        "evidence_files_sha256": {path.name: sha256(path) for path, *_ in records},
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "trajectory_count": len(records),
        "group_count": len(groups),
        "valid_trajectory_count": sum(1 for x in records if bool(x[3]) and all(bool(t.get("prompt", {}).get("token_ids")) and bool(t.get("response", {}).get("token_ids")) for t in x[3])),
        "valid_group_count": sum(int(x["valid"]) for x in group_rows),
        "reward_mean": statistics.mean(rewards) if rewards else None,
        "reward_counts": dict(sorted(Counter(rewards).items())),
        "group_rows": group_rows,
        "length_context": {
            "response_token_count": len(response_lengths),
            "response": {"mean": statistics.mean(response_lengths), **{k: percentile(response_lengths, v) for k, v in {"p50": 0.50, "p75": 0.75, "p90": 0.90, "p95": 0.95, "p99": 0.99, "max": 1.0}.items()}} if response_lengths else {},
            "prompt": {"mean": statistics.mean(prompt_lengths), **{k: percentile(prompt_lengths, v) for k, v in {"p50": 0.50, "p75": 0.75, "p90": 0.90, "p95": 0.95, "p99": 0.99, "max": 1.0}.items()}} if prompt_lengths else {},
            "context": {"mean": statistics.mean(context_lengths), **{k: percentile(context_lengths, v) for k, v in {"p50": 0.50, "p75": 0.75, "p90": 0.90, "p95": 0.95, "p99": 0.99, "max": 1.0}.items()}} if context_lengths else {},
            "finish_reason_length_count": sum(x.lower() == "length" for x in finish_reasons),
            "finish_reason_values": dict(Counter(finish_reasons)),
            "response_at_cap_count": cap_hits,
            "response_at_cap_rate": cap_hits / len(response_lengths) if response_lengths else None,
            "context_over_max_model_len_count": contexts_over,
            "eos_flags_observed": len(eos_flags),
        },
        "answer_texts_available": bool(answer_texts),
        "answer_exact_duplicate_rate": exact_dup,
        "scorer_routing_observed": dict(route_counts),
        "raw_artifacts_not_embedded": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "trajectory_count": len(records), "group_count": len(groups), "output": str(args.output)}, sort_keys=True))

if __name__ == "__main__":
    main()
