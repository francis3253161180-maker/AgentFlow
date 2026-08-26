#!/usr/bin/env python3
"""Recover the first full-blind artifact after a set-order reproducibility bug.

The first preparation run used a set before seeded shuffling.  This utility
uses the question order recorded during the completed manual review and the
same RNG stream to reconstruct the original opaque/candidate mapping from the
still-available raw members.  It is an audit-artifact repair utility only; it
does not touch scorer or rollout data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


QUESTION_PREFIXES = [
    "Let $L$ be a finite-dimensional complex semisimple Lie algebra.",
    "who played sara lance in arrow season 2?",
    "Consider the lines",
    "Given that $x in",  # normalized matching fallback below handles LaTeX punctuation.
    "when is magnus chase book 3 coming out?",
    "Determine the number of zeros of the function",
    "what is the age limit to buy alcohol in uk?",
    "Determine if the function",
    "Suppose",
    "Determine the number of distinct terms",
    "who develops the first computer language called cobol?",
    "the smiths there is a light",
    "Inside triangle",
    "what is target disk mode on a mac?",
    "who was china fighting in world war 2?",
    "where do purple martins go",
    "Determine whether there exist two matrices",
    "Determine the limit superior of the sequence",
    "Evaluate the limit:",
    "Find the Laurent series",
    "Let $f: \\mathbb{N}",
    "who plays elizabeth swann",
    "when does fear of the walking dead season 3 start?",
    "Determine whether the following statement is true or false",
    "who was the first chief of naval staff in nigeria?",
    "Determine the minimum number of vertices",
    "nba record for most blocks in a game?",
    "Given the system of equations:",
    "Given the sequence",
    "where do salmon spend most",
    "PQR",
    "Determine the order of the generalized quaternion group",
    "who plays arnie on im dying up here?",
    "when does greenhouse academy season 2 come out?",
    "when did they start building the sydney opera house?",
    "who is known as the nepolian of iran?",
    "who starred in saturday night and sunday morning?",
    "Given that for all",
    "In triangle $ABC$, where $AB = AC",
    "when did the united states join united nations?",
    "Evaluate the limit:",
    "Given a bounded open set",
    "Find the norm of the operator",
    "AD = BE = AC",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--current-blinded", type=Path, required=True)
    p.add_argument("--current-sealed", type=Path, required=True)
    p.add_argument("--output-blinded", type=Path, required=True)
    p.add_argument("--output-sealed", type=Path, required=True)
    p.add_argument("--exposure-manifest", type=Path)
    p.add_argument("--seed", type=int, default=20260826)
    return p.parse_args()


def norm(value: str) -> str:
    return " ".join(value.casefold().replace("\\", "").split())


def main() -> None:
    a = parse_args()
    current = json.loads(a.current_blinded.read_text(encoding="utf-8"))
    current_sealed = json.loads(a.current_sealed.read_text(encoding="utf-8"))
    groups = current["groups"]
    if len(groups) != 44 or len(QUESTION_PREFIXES) != 44:
        raise SystemExit("expected exactly 44 groups and recovery prefixes")
    current_by_opaque = {group["opaque_id"]: group for group in current_sealed["groups"]}
    matched = []
    used = set()
    for number, prefix in enumerate(QUESTION_PREFIXES, 1):
        candidates = [
            (position, group) for position, group in enumerate(groups)
            if position not in used and (
                norm(group["question"]).startswith(norm(prefix))
                or norm(prefix) in norm(group["question"])
            )
        ]
        if len(candidates) != 1:
            # Two questions begin with “Evaluate the limit”; use the expected
            # ordinal split from the manual-review record.
            if number == 19:
                candidates = [(position, group) for position, group in enumerate(groups) if position not in used and "x\\rightarrow \\infty" in group["question"]]
            elif number == 41:
                candidates = [(position, group) for position, group in enumerate(groups) if position not in used and "150" in group["question"]]
        if len(candidates) != 1:
            raise SystemExit(f"question prefix {number} matched {len(candidates)} groups: {prefix}")
        position, group = candidates[0]
        used.add(position)
        matched.append((position, group))

    # Consume exactly the same seeded RNG stream as preparation: one group
    # shuffle, then one 4-member shuffle per group position.
    rng = random.Random(a.seed)
    dummy = list(range(44))
    rng.shuffle(dummy)
    permutations = []
    for _ in range(44):
        member_order = list(range(4))
        rng.shuffle(member_order)
        permutations.append(member_order)

    recovered_blind_groups = []
    recovered_sealed_groups = []
    for old_position, (current_position, group) in enumerate(matched):
        current_sealed_group = current_by_opaque[group["opaque_id"]]
        current_members = {member["candidate_id"]: member for member in current_sealed_group["members"]}
        current_candidates = {candidate["candidate_id"]: candidate for candidate in group["candidates"]}
        current_ordered_members = [current_members[f"candidate-{i}"] for i in range(1, 5)]
        current_ordered_candidates = [current_candidates[f"candidate-{i}"] for i in range(1, 5)]
        current_perm = permutations[current_position]
        canonical_members = [None] * 4
        canonical_candidates = [None] * 4
        for current_index, canonical_index in enumerate(current_perm):
            canonical_members[canonical_index] = current_ordered_members[current_index]
            canonical_candidates[canonical_index] = current_ordered_candidates[current_index]
        old_perm = permutations[old_position]
        old_members = [canonical_members[index] for index in old_perm]
        old_candidates = [canonical_candidates[index] for index in old_perm]
        opaque = f"full-blind-{old_position + 1:03d}"
        recovered_blind_groups.append({
            "opaque_id": opaque,
            "question": group["question"],
            "ground_truth": group["ground_truth"],
            "candidates": [
                {"candidate_id": f"candidate-{i}", "candidate_answer": candidate["candidate_answer"]}
                for i, candidate in enumerate(old_candidates, 1)
            ],
        })
        recovered_sealed_groups.append({
            **current_sealed_group,
            "opaque_id": opaque,
            "members": [
                {**member, "candidate_id": f"candidate-{i}"}
                for i, member in enumerate(old_members, 1)
            ],
        })

    recovered_blind = {
        "audit": "2026-08-26 full outcome-reward audit blind phase",
        "selection_seed": a.seed,
        "selected_count": 44,
        "groups": recovered_blind_groups,
    }
    recovered_sealed = {
        "audit": "SEALED full-audit mapping recovered from first blind preparation run",
        "selection_seed": a.seed,
        "blinded_sha256": hashlib.sha256(json.dumps(recovered_blind, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        "groups": recovered_sealed_groups,
    }
    a.output_blinded.write_text(json.dumps(recovered_blind, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    a.output_sealed.write_text(json.dumps(recovered_sealed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if a.exposure_manifest:
        exposure = json.loads(a.exposure_manifest.read_text(encoding="utf-8"))
        key_by_group = {entry["group_key"]: entry for entry in exposure["groups"]}
        exposure["canonical_blind_order"] = [
            {
                "opaque_id": sealed_group["opaque_id"],
                "group_key": sealed_group["group_key"],
                "candidate_ids": [member["candidate_id"] for member in sealed_group["members"]],
            }
            for sealed_group in recovered_sealed_groups
        ]
        exposure["blind_mapping_recovery"] = {
            "used_after_manual_labels": True,
            "reason": "The first run used set iteration before seeded shuffle; canonical order is recorded here so the final artifact is auditable.",
            "group_keys_resolved": len(key_by_group),
        }
        a.exposure_manifest.write_text(json.dumps(exposure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"recovered_groups": 44, "output_blinded": str(a.output_blinded), "output_sealed": str(a.output_sealed)}))


if __name__ == "__main__":
    main()
