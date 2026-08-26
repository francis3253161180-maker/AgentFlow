#!/usr/bin/env python3
"""Select a reproducible NQ/mathhard prompt set for the rollout-only audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--per-source", type=int, default=50)
    return parser.parse_args()


def extra_info(row: Any) -> dict[str, Any]:
    if isinstance(row, dict):
        return row
    if isinstance(row, str):
        parsed = json.loads(row)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"extra_info is not a mapping: {type(row)!r}")


def main() -> None:
    args = parse_args()
    import pandas as pd

    frame = pd.read_parquet(args.data)
    required = {"id", "question", "result", "source", "extra_info"}
    missing = required.difference(frame.columns)
    if missing:
        raise SystemExit(f"missing required columns: {sorted(missing)}")

    rng = random.Random(args.seed)
    chosen: list[tuple[int, dict[str, Any]]] = []
    used_idx: set[int] = set()
    for source in ("nq", "mathhard"):
        candidates = [
            (position, row)
            for position, row in frame.iterrows()
            if str(row["source"]) == source
        ]
        rng.shuffle(candidates)
        source_count = 0
        for position, row in candidates:
            info = extra_info(row["extra_info"])
            idx = info.get("idx")
            if idx is None:
                continue
            idx = int(idx)
            # Upstream idx values are source-local.  Avoid path-key collisions
            # while leaving each selected data row otherwise unchanged.
            if idx in used_idx:
                continue
            used_idx.add(idx)
            chosen.append((int(position), {"source": source, "idx": idx, "id": int(row["id"])}))
            source_count += 1
            if source_count == args.per_source:
                break
        if source_count != args.per_source:
            raise SystemExit(f"could not select {args.per_source} rows for {source}")

    selected = frame.loc[[position for position, _ in chosen]].reset_index(drop=True)
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(args.output_parquet, index=False)

    rows = []
    for order, (position, metadata) in enumerate(chosen):
        rows.append({"order": order, "source_row_position": position, **metadata})
    parquet_sha = hashlib.sha256(args.output_parquet.read_bytes()).hexdigest()
    manifest = {
        "script": "scripts/prepare_rollout_difficulty_audit_20260826.py",
        "source_data": str(args.data),
        "selection_seed": args.seed,
        "requested_per_source": args.per_source,
        "selected_count": len(rows),
        "source_counts": {
            source: sum(row["source"] == source for row in rows)
            for source in ("nq", "mathhard")
        },
        "selected_parquet_sha256": parquet_sha,
        "idx_collision_avoidance": True,
        "rows": rows,
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: manifest[key] for key in manifest if key != "rows"}, sort_keys=True))


if __name__ == "__main__":
    main()
