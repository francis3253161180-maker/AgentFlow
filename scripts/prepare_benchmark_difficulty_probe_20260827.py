#!/usr/bin/env python3
"""Prepare fixed, answer-content-independent benchmark probe samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def parse_mapping(values: list[str], label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"{label} must use DATASET=PATH: {value}")
        dataset, path = value.split("=", 1)
        if not dataset or not path or dataset in result:
            raise SystemExit(f"invalid or duplicate {label}: {value}")
        result[dataset] = Path(path)
    return result


def ground_truth_text(value: Any) -> str:
    # Bamboogle stores answers as a JSON list; preserve that native value in
    # text because AgentFlow's result column is textual.
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def prepare_dataset(
    dataset: str,
    input_path: Path,
    output_path: Path,
    seed: int,
    requested_count: int,
) -> dict[str, Any]:
    import pandas as pd

    input_bytes = input_path.read_bytes()
    rows = json.loads(input_bytes)
    if not isinstance(rows, list):
        raise SystemExit(f"{dataset}: expected a JSON list")

    valid_indices: list[int] = []
    invalid_reasons: dict[str, int] = {}
    for index, row in enumerate(rows):
        reason = None
        if not isinstance(row, dict):
            reason = "row_not_object"
        else:
            question = row.get("query") or row.get("question")
            if not str(question or "").strip():
                reason = "missing_question_or_query"
            elif "answer" not in row or not str(row.get("answer", "")).strip():
                reason = "missing_answer"
            elif row.get("pid") is None or not str(row.get("pid")).strip():
                reason = "missing_pid"
        if reason is None:
            valid_indices.append(index)
        else:
            invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1

    count = min(requested_count, len(valid_indices))
    rng = random.Random(seed)
    selected_indices = sorted(rng.sample(valid_indices, count))
    selected_rows: list[dict[str, Any]] = []
    parquet_rows: list[dict[str, Any]] = []
    for sample_order, source_row_index in enumerate(selected_indices):
        row = rows[source_row_index]
        question = str(row.get("query") or row.get("question"))
        answer = ground_truth_text(row["answer"])
        parquet_rows.append(
            {
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
                    "benchmark_pid": str(row["pid"]),
                    "probe_sample_order": sample_order,
                },
            }
        )
        selected_rows.append(
            {
                "sample_order": sample_order,
                "source_row_index": source_row_index,
                "pid": str(row["pid"]),
                "question_sha256": sha256_bytes(question.encode("utf-8")),
                "ground_truth_sha256": sha256_bytes(answer.encode("utf-8")),
                "source_record_sha256": sha256_json(row),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(parquet_rows).to_parquet(output_path, index=False)
    return {
        "dataset": dataset,
        "source_path": str(input_path),
        "source_sha256": sha256_bytes(input_bytes),
        "source_row_count": len(rows),
        "structurally_valid_row_count": len(valid_indices),
        "invalid_row_counts": invalid_reasons,
        "requested_sample_count": requested_count,
        "selected_sample_count": count,
        "output_parquet": str(output_path),
        "output_parquet_sha256": sha256_bytes(output_path.read_bytes()),
        "answer_encoding": "native scalar string; compact JSON text for list/dict answers",
        "selected_rows": selected_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", required=True, help="DATASET=JSON_PATH")
    parser.add_argument("--output-parquet", action="append", required=True, help="DATASET=PARQUET_PATH")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--sample-count", type=int, default=20)
    args = parser.parse_args()
    if args.sample_count <= 0:
        raise SystemExit("--sample-count must be positive")
    inputs = parse_mapping(args.dataset, "--dataset")
    outputs = parse_mapping(args.output_parquet, "--output-parquet")
    if set(inputs) != set(outputs):
        raise SystemExit("dataset input/output names differ")

    datasets = {
        name: prepare_dataset(name, inputs[name], outputs[name], args.seed, args.sample_count)
        for name in sorted(inputs)
    }
    manifest = {
        "schema_version": 1,
        "purpose": "rollout-only difficulty probe; benchmark examples are evaluation/probe-only and forbidden as future formal training data",
        "selection_seed": args.seed,
        "selection_rule": "random.sample over structurally valid source rows only, then sort selected source row indices; no answer-content criterion",
        "requested_sample_count_per_dataset": args.sample_count,
        "datasets": datasets,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {name: {key: value for key, value in data.items() if key != "selected_rows"} for name, data in datasets.items()},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
