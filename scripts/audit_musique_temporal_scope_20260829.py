#!/usr/bin/env python3
"""Offline-only MuSiQue decomposition audit with no runtime-data export.

This utility is deliberately not imported by AgentFlow.  It reads an official
record solely to test whether a question decomposition separates a
time-qualified intermediate relation from an unqualified requested property.
The output contains only hashes, counts, and boolean lexical predicates: never
the answer, support titles, paragraph text, or decomposition text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


YEAR = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2})\b")
PROPERTY = re.compile(r"\b(?:game|games|match|matches|matchday|matchdays|season|standings)\b", re.I)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-jsonl", type=Path, required=True)
    parser.add_argument("--record-id", required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    record = None
    with args.official_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            candidate = json.loads(line)
            if candidate.get("id") == args.record_id:
                record = candidate
                break
    if record is None:
        raise SystemExit("requested official record was not found")

    decompositions = record.get("question_decomposition", [])
    items = []
    for index, decomposition in enumerate(decompositions):
        question = str(decomposition.get("question", ""))
        support_index = decomposition.get("paragraph_support_idx")
        paragraph = None
        if isinstance(support_index, int) and 0 <= support_index < len(record.get("paragraphs", [])):
            paragraph = record["paragraphs"][support_index]
        support_text = str((paragraph or {}).get("paragraph_text", ""))
        items.append({
            "ordinal": index,
            "question_sha256": digest(question),
            "question_has_year_like_token": bool(YEAR.search(question)),
            "question_has_requested_property_marker": bool(PROPERTY.search(question)),
            "support_index": support_index,
            "support_is_marked_supporting": bool((paragraph or {}).get("is_supporting")),
            "support_sha256": digest(support_text) if support_text else None,
            "support_has_year_like_token": bool(YEAR.search(support_text)),
            "support_has_requested_property_marker": bool(PROPERTY.search(support_text)),
        })
    try:
        repo_commit = subprocess.check_output(
            ["git", "-C", str(args.official_repo), "rev-parse", "HEAD"], text=True,
        ).strip()
    except subprocess.CalledProcessError:
        repo_commit = None
    temporal_relation_then_property = (
        len(items) >= 2
        and items[0]["question_has_year_like_token"]
        and not items[0]["question_has_requested_property_marker"]
        and not items[1]["question_has_year_like_token"]
        and items[1]["question_has_requested_property_marker"]
    )
    result = {
        "schema_version": 1,
        "purpose": "offline diagnostic only; no field here is consumed by AgentFlow runtime",
        "record_id": args.record_id,
        "official_jsonl_sha256": hashlib.sha256(args.official_jsonl.read_bytes()).hexdigest(),
        "official_repo_commit": repo_commit,
        "record_question_sha256": digest(str(record.get("question", ""))),
        "decomposition_count": len(items),
        "decomposition_metadata": items,
        "temporal_relation_then_unqualified_property": temporal_relation_then_property,
        "contains_answer_or_support_text": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
