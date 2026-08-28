#!/usr/bin/env python3
"""Materialize the length-only selected groups without changing their rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--meta-output", type=Path, required=True)
    args = ap.parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    selected_indices = {int(row["source_row_index"]) for row in selection["selected_groups"]}
    table = pq.read_table(args.source)
    rows = table.to_pylist()
    chosen = [row for row in rows if int(row["extra_info"]["source_row_index"]) in selected_indices]
    if len(chosen) != len(selected_indices):
        raise SystemExit(f"selection/source mismatch: selected={len(selected_indices)} rows={len(chosen)}")
    chosen.sort(key=lambda row: int(row["extra_info"]["source_row_index"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(chosen, schema=table.schema), args.output)
    meta = {
        "schema_version": 1,
        "source": str(args.source),
        "source_sha256": sha256(args.source),
        "selection": str(args.selection),
        "selection_sha256": sha256(args.selection),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "selected_group_count": len(chosen),
        "selected_source_row_indices": [int(row["extra_info"]["source_row_index"]) for row in chosen],
        "answer_content_selection": False,
    }
    args.meta_output.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, sort_keys=True))


if __name__ == "__main__":
    main()
