#!/usr/bin/env python3
"""Prepare fixed MuSiQue-Ans/2Wiki dev probe rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def read_rows(path: Path) -> tuple[list[dict[str, Any]], str]:
    raw = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        encoding = "jsonl"
    else:
        rows = json.loads(raw)
        encoding = "json"
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise SystemExit(f"{path}: expected a list of JSON objects")
    return rows, encoding


def fields(dataset: str, row: dict[str, Any]) -> tuple[str, str, str]:
    if dataset == "musique":
        keys = ("id", "question", "answer")
    elif dataset == "2wiki":
        keys = ("_id", "question", "answer")
    else:
        raise SystemExit(f"unsupported dataset: {dataset}")
    return tuple(str(row.get(key, "")).strip() for key in keys)  # type: ignore[return-value]


def prepare(dataset: str, source: Path, output: Path, seed: int, count: int, source_commit: str) -> dict[str, Any]:
    rows, encoding = read_rows(source)
    valid: list[int] = []
    invalid: dict[str, int] = {}
    for index, row in enumerate(rows):
        identifier, question, answer = fields(dataset, row)
        reason = None
        if not identifier:
            reason = "missing_identifier"
        elif not question:
            reason = "missing_question"
        elif not answer:
            reason = "missing_answer"
        if reason is None:
            valid.append(index)
        else:
            invalid[reason] = invalid.get(reason, 0) + 1

    chosen = sorted(random.Random(seed).sample(valid, min(count, len(valid))))
    parquet_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for sample_order, source_row_index in enumerate(chosen):
        row = rows[source_row_index]
        identifier, question, answer = fields(dataset, row)
        parquet_rows.append({
            "id": sample_order,
            "question": question,
            "chain": "",
            "result": answer,
            "source": dataset,
            "extra_info": {
                "ground_truth": answer,
                "groundtruth": answer,
                "idx": source_row_index,
                "benchmark": dataset,
                "benchmark_split": "dev",
                "benchmark_id": identifier,
                "probe_sample_order": sample_order,
            },
        })
        selected_rows.append({
            "sample_order": sample_order,
            "source_row_index": source_row_index,
            "source_id": identifier,
            "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
            "ground_truth_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
            "source_record_sha256": sha256_json(row),
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(parquet_rows).to_parquet(output, index=False)
    return {
        "dataset": dataset,
        "source_path": str(source),
        "source_commit": source_commit,
        "source_sha256": sha256_path(source),
        "source_encoding": encoding,
        "source_split": "dev",
        "source_row_count": len(rows),
        "structurally_valid_row_count": len(valid),
        "invalid_row_counts": invalid,
        "selection_seed": seed,
        "selection_rule": "random.sample over structurally valid dev rows only; sorted source indices; no answer-content criterion",
        "requested_sample_count": count,
        "selected_sample_count": len(chosen),
        "output_parquet": str(output),
        "output_parquet_sha256": sha256_path(output),
        "selected_rows": selected_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--musique-source", type=Path, required=True)
    parser.add_argument("--twowiki-source", type=Path, required=True)
    parser.add_argument("--musique-output", type=Path, required=True)
    parser.add_argument("--twowiki-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--musique-commit", required=True)
    parser.add_argument("--twowiki-commit", required=True)
    args = parser.parse_args()
    if args.sample_count <= 0:
        raise SystemExit("--sample-count must be positive")
    datasets = {
        "musique": prepare("musique", args.musique_source, args.musique_output, args.seed, args.sample_count, args.musique_commit),
        "2wiki": prepare("2wiki", args.twowiki_source, args.twowiki_output, args.seed, args.sample_count, args.twowiki_commit),
    }
    manifest = {
        "schema_version": 1,
        "purpose": "bounded rollout-only AgentFlow difficulty probe; dev examples are probe-only and forbidden as future formal training data",
        "selection_seed": args.seed,
        "requested_sample_count_per_dataset": args.sample_count,
        "datasets": datasets,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({name: {k: v for k, v in value.items() if k != "selected_rows"} for name, value in datasets.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
