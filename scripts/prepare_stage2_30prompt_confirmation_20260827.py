#!/usr/bin/env python3
"""Prepare answer-content-independent, non-overlapping stage-2 additions."""

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
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def ground_truth_text(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def parse_mapping(values: list[str], label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"{label} must use DATASET=PATH: {value}")
        name, path = value.split("=", 1)
        if not name or not path or name in result:
            raise SystemExit(f"invalid or duplicate {label}: {value}")
        result[name] = Path(path)
    return result


def parse_int_mapping(values: list[str], label: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"{label} must use DATASET=INTEGER: {value}")
        name, raw = value.split("=", 1)
        if not name or not raw or name in result:
            raise SystemExit(f"invalid or duplicate {label}: {value}")
        try:
            result[name] = int(raw)
        except ValueError as exc:
            raise SystemExit(f"invalid {label} integer: {value}") from exc
    return result


def load_historical_rows(manifests: list[Path], dataset: str, source_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    source_resolved = source_path.resolve()
    records: list[dict[str, Any]] = []
    provenance: list[str] = []
    for manifest_path in manifests:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = data.get("datasets", {}).get(dataset)
        if not entry:
            continue
        entry_source = Path(entry["source_path"])
        if not entry_source.is_absolute():
            entry_source = (Path.cwd() / entry_source).resolve()
        if entry_source != source_resolved:
            raise SystemExit(f"{dataset}: historical source mismatch in {manifest_path}")
        if entry.get("source_sha256") != sha256_bytes(source_path.read_bytes()):
            raise SystemExit(f"{dataset}: source hash mismatch in {manifest_path}")
        provenance.append(str(manifest_path))
        for row in entry.get("selected_rows", []):
            records.append({"manifest": str(manifest_path), **row})
    return records, provenance


def prepare_dataset(
    dataset: str,
    source_path: Path,
    output_path: Path,
    target_count: int,
    seed: int,
    historical_manifests: list[Path],
) -> dict[str, Any]:
    import pandas as pd

    source_bytes = source_path.read_bytes()
    rows = json.loads(source_bytes)
    if not isinstance(rows, list):
        raise SystemExit(f"{dataset}: expected JSON list")
    historical, provenance = load_historical_rows(historical_manifests, dataset, source_path)
    historical_indices = [int(row["source_row_index"]) for row in historical]
    historical_pids = [str(row["pid"]) for row in historical]
    if len(set(historical_indices)) != len(historical_indices):
        raise SystemExit(f"{dataset}: duplicate historical source row index")
    if len(historical) >= target_count:
        raise SystemExit(f"{dataset}: historical rows already reach target; no incremental probe needed")

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
            elif row.get("pid") is None or not str(row["pid"]).strip():
                reason = "missing_pid"
        if reason is None:
            valid_indices.append(index)
        else:
            invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1

    available = [index for index in valid_indices if index not in set(historical_indices)]
    additional_count = target_count - len(historical)
    if len(available) < additional_count:
        raise SystemExit(f"{dataset}: only {len(available)} non-overlapping valid rows for {additional_count} additions")
    selected_indices = sorted(random.Random(seed).sample(available, additional_count))
    selected_set = set(selected_indices)
    if selected_set.intersection(historical_indices):
        raise SystemExit(f"{dataset}: incremental/index overlap detected")

    incremental_rows: list[dict[str, Any]] = []
    parquet_rows: list[dict[str, Any]] = []
    for sample_order, source_row_index in enumerate(selected_indices):
        row = rows[source_row_index]
        question = str(row.get("query") or row.get("question"))
        answer = ground_truth_text(row["answer"])
        pid = str(row["pid"])
        if pid in set(historical_pids):
            raise SystemExit(f"{dataset}: incremental/pid overlap detected")
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
                "benchmark_pid": pid,
                "probe_sample_order": sample_order,
                "stage2_phase": "incremental",
            },
        })
        incremental_rows.append({
            "sample_order": sample_order,
            "source_row_index": source_row_index,
            "pid": pid,
            "question_sha256": sha256_bytes(question.encode("utf-8")),
            "ground_truth_sha256": sha256_bytes(answer.encode("utf-8")),
            "source_record_sha256": sha256_json(row),
            "phase": "incremental",
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(parquet_rows).to_parquet(output_path, index=False)
    historical_rows = [{"phase": "historical", **row} for row in historical]
    all_indices = historical_indices + selected_indices
    all_pids = historical_pids + [row["pid"] for row in incremental_rows]
    return {
        "dataset": dataset,
        "source_path": str(source_path),
        "source_sha256": sha256_bytes(source_bytes),
        "source_row_count": len(rows),
        "structurally_valid_row_count": len(valid_indices),
        "invalid_row_counts": invalid_reasons,
        "target_sample_count": target_count,
        "historical_sample_count": len(historical),
        "incremental_sample_count": len(incremental_rows),
        "historical_manifests": provenance,
        "historical_source_row_indices": historical_indices,
        "selected_source_row_indices": selected_indices,
        "output_parquet": str(output_path),
        "output_parquet_sha256": sha256_bytes(output_path.read_bytes()),
        "answer_encoding": "native scalar string; compact JSON text for list/dict answers",
        "historical_rows": historical_rows,
        "incremental_rows": incremental_rows,
        "all_rows": historical_rows + incremental_rows,
        "overlap_check": {
            "historical_duplicate_index_count": len(historical_indices) - len(set(historical_indices)),
            "historical_duplicate_pid_count": len(historical_pids) - len(set(historical_pids)),
            "incremental_historical_index_overlap_count": len(set(selected_indices).intersection(historical_indices)),
            "incremental_historical_pid_overlap_count": len(set(row["pid"] for row in incremental_rows).intersection(historical_pids)),
            "combined_unique_index_count": len(set(all_indices)),
            "combined_unique_pid_count": len(set(all_pids)),
            "combined_row_count": len(all_indices),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", required=True, help="DATASET=JSON_PATH")
    parser.add_argument("--output-parquet", action="append", required=True, help="DATASET=PARQUET_PATH")
    parser.add_argument("--target-count", action="append", required=True, help="DATASET=30")
    parser.add_argument("--historical-manifest", action="append", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()
    inputs = parse_mapping(args.dataset, "--dataset")
    outputs = parse_mapping(args.output_parquet, "--output-parquet")
    targets = parse_int_mapping(args.target_count, "--target-count")
    if set(inputs) != set(outputs) or set(inputs) != set(targets):
        raise SystemExit("dataset names differ between --dataset, --output-parquet, and --target-count")
    manifests = [Path(value) for value in args.historical_manifest]
    if len(set(manifests)) != len(manifests):
        raise SystemExit("duplicate --historical-manifest")
    datasets = {}
    for name in sorted(inputs):
        datasets[name] = prepare_dataset(name, inputs[name], outputs[name], targets[name], args.seed, manifests)
    manifest = {
        "schema_version": 1,
        "purpose": "stage-2 rollout-only confirmation; benchmark examples are probe-only and forbidden in future formal training",
        "selection_seed": args.seed,
        "selection_rule": "for each dataset, random.sample over structurally valid source rows after excluding every historical manifest row, then sort source indices; no answer-content criterion",
        "target_sample_count_per_dataset": targets,
        "historical_manifest_paths": [str(path) for path in manifests],
        "datasets": datasets,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({name: {key: value for key, value in data.items() if key not in {"historical_rows", "incremental_rows", "all_rows"}} for name, data in datasets.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
