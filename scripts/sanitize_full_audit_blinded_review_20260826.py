#!/usr/bin/env python3
"""Strip all non-review metadata from the finalized blind review file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    source = json.loads(a.input.read_text(encoding="utf-8"))
    clean = {
        "audit": "2026-08-26 full outcome-reward audit blind phase",
        "selection_seed": source.get("selection_seed", 20260826),
        "selected_count": len(source["groups"]),
        "groups": [
            {
                "opaque_id": group["opaque_id"],
                "question": group["question"],
                "ground_truth": group["ground_truth"],
                "candidates": [
                    {"candidate_id": candidate["candidate_id"], "candidate_answer": candidate["candidate_answer"]}
                    for candidate in group["candidates"]
                ],
            }
            for group in source["groups"]
        ],
    }
    a.output.write_text(json.dumps(clean, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"groups": len(clean["groups"]), "rows": sum(len(group["candidates"]) for group in clean["groups"])}))


if __name__ == "__main__":
    main()
