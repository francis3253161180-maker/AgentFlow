#!/usr/bin/env python3
"""Offline Game24 reward audit and length-only probe preparation.

This script never contacts a model provider.  It decodes only the persisted
random30 evidence, applies an independent Fraction oracle, and writes a
metadata-only selection manifest for the optional long-context probe.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

os.environ.setdefault("AGENTFLOW_DISABLE_EXTERNAL_LLM", "1")
os.environ.setdefault("AGENTFLOW_REWARD_JUDGE_ENABLED", "0")

from transformers import AutoTokenizer

from train.utils import deterministic_decision


TOKEN_RE = re.compile(r"\d+|[()+\-*/×÷]")
NUMBERS_RE = re.compile(r"numbers\s*\[([^]]+)\]", re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
EXPR_RE = re.compile(r"[0-9][0-9\s()+\-*/×÷.]*")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_puzzle(question: str) -> tuple[int, ...]:
    match = NUMBERS_RE.search(question)
    if not match:
        raise ValueError(f"cannot find Game24 numbers in {question[:120]!r}")
    return tuple(int(x.strip()) for x in match.group(1).split(","))


class ExprParser:
    def __init__(self, expression: str):
        self.tokens = TOKEN_RE.findall(expression.replace("\\times", "×"))
        compact = re.sub(r"\s+", "", expression)
        reconstructed = "".join(self.tokens).replace("×", "*").replace("÷", "/")
        if reconstructed != compact.replace("×", "*").replace("÷", "/"):
            raise ValueError("unsupported expression characters")
        self.pos = 0
        self.numbers: list[int] = []

    def peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def take(self) -> str:
        token = self.peek()
        if token is None:
            raise ValueError("unexpected end")
        self.pos += 1
        return token

    def expression(self) -> Fraction:
        value = self.term()
        while self.peek() in {"+", "-"}:
            op = self.take()
            rhs = self.term()
            value = value + rhs if op == "+" else value - rhs
        return value

    def term(self) -> Fraction:
        value = self.factor()
        while self.peek() in {"*", "/", "×", "÷"}:
            op = self.take()
            rhs = self.factor()
            if op in {"*", "×"}:
                value *= rhs
            else:
                if rhs == 0:
                    raise ValueError("division by zero")
                value /= rhs
        return value

    def factor(self) -> Fraction:
        token = self.peek()
        if token == "(":
            self.take()
            value = self.expression()
            if self.take() != ")":
                raise ValueError("missing closing parenthesis")
            return value
        if token == "-":
            self.take()
            return -self.factor()
        if token is None or not token.isdigit():
            raise ValueError("expected integer")
        self.numbers.append(int(self.take()))
        return Fraction(self.numbers[-1])

    def parse(self) -> tuple[Fraction, tuple[int, ...]]:
        value = self.expression()
        if self.peek() is not None:
            raise ValueError("trailing tokens")
        return value, tuple(self.numbers)


def parse_expression(expression: str) -> tuple[Fraction, tuple[int, ...]]:
    return ExprParser(expression).parse()


def find_expression_candidates(text: str) -> list[str]:
    text = re.sub(r"\\boxed\s*", "", text)
    text = text.replace("\\(", "").replace("\\)", "")
    candidates = []
    for match in EXPR_RE.finditer(text):
        value = match.group(0).strip(" .,:;=")
        if any(op in value for op in "+-*/×÷"):
            candidates.append(value)
    return candidates


def solve_game24(numbers: tuple[int, ...]) -> str | None:
    """Return one exact expression using the input multiset once, if any."""
    items = [(Fraction(value), str(value)) for value in numbers]

    def search(values: list[tuple[Fraction, str]]) -> str | None:
        if len(values) == 1:
            return values[0][1] if values[0][0] == 24 else None
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                left, right = values[i], values[j]
                rest = [v for k, v in enumerate(values) if k not in {i, j}]
                candidates = [
                    (left[0] + right[0], f"({left[1]}+{right[1]})"),
                    (left[0] - right[0], f"({left[1]}-{right[1]})"),
                    (right[0] - left[0], f"({right[1]}-{left[1]})"),
                    (left[0] * right[0], f"({left[1]}*{right[1]})"),
                ]
                if right[0] != 0:
                    candidates.append((left[0] / right[0], f"({left[1]}/{right[1]})"))
                if left[0] != 0:
                    candidates.append((right[0] / left[0], f"({right[1]}/{left[1]})"))
                for candidate in candidates:
                    result = search(rest + [candidate])
                    if result:
                        return result
        return None

    return search(items)


def decode_response(tokenizer: Any, triplet: dict[str, Any]) -> str:
    response = triplet.get("response") or {}
    ids = response.get("response_token_ids") or response.get("token_ids") or []
    return tokenizer.decode(ids, skip_special_tokens=False) if ids else ""


def final_answer(tokenizer: Any, evidence: dict[str, Any]) -> tuple[str, str, str]:
    triplets = evidence["rollout"].get("triplets") or []
    if not triplets:
        return "None", "no_candidate", ""
    final_text = decode_response(tokenizer, triplets[-1])
    matches = ANSWER_RE.findall(final_text)
    if matches:
        return matches[-1].strip(), "answer_tag", final_text
    if final_text.strip():
        return final_text.strip(), "full_final_response", final_text
    return "None", "no_candidate", final_text


def oracle_evaluation(answer: str, puzzle: tuple[int, ...]) -> dict[str, Any]:
    expressions = find_expression_candidates(answer)
    parsed = []
    for expression in expressions:
        try:
            value, used = parse_expression(expression)
        except ValueError:
            continue
        parsed.append({"expression": expression, "value": str(value), "numbers": list(used)})
    target = sorted(puzzle)
    valid = [item for item in parsed if sorted(item["numbers"]) == target and item["value"] == "24"]
    same_numbers = [item for item in parsed if sorted(item["numbers"]) == target]
    if valid:
        category = "oracle_valid"
    elif not expressions:
        category = "invalid_or_no_expression"
    elif not parsed:
        category = "invalid_or_no_expression"
    elif not same_numbers:
        category = "wrong_number_multiset"
    else:
        category = "arithmetic_not_24"
    return {
        "category": category,
        "expression_candidates": expressions[:12],
        "parsed_expressions": parsed[:12],
        "oracle_valid": bool(valid),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    files = sorted(Path(args.evidence_dir).glob("rollout_*.json"))
    rows = []
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for path in files:
        evidence = json.loads(path.read_text(encoding="utf-8"))
        sample = evidence["original_sample"]
        rollout = evidence["rollout"]
        answer, answer_source, final_text = final_answer(tokenizer, evidence)
        puzzle = parse_puzzle(sample["question"])
        gt = str(sample.get("extra_info", {}).get("ground_truth", sample.get("result", "24")))
        decision = deterministic_decision(gt, answer)
        oracle = oracle_evaluation(answer, puzzle)
        serialized = json.dumps(evidence, ensure_ascii=False)
        context_error = "maximum context length" in serialized or "BadRequestError" in serialized
        stored = float(rollout.get("final_reward", 0.0) or 0.0)
        if answer_source == "no_candidate":
            cause = "no_candidate_final_answer"
        elif oracle["oracle_valid"] and not decision.value:
            cause = "production_scorer_disagreement"
        elif context_error:
            cause = "context_error_affected"
        else:
            cause = oracle["category"]
        row = {
            "evidence_file": path.name,
            "rollout_id": rollout.get("rollout_id"),
            "group_id": sample.get("data_id"),
            "source_row_index": sample.get("extra_info", {}).get("source_row_index"),
            "question": sample["question"],
            "puzzle": list(puzzle),
            "ground_truth": gt,
            "stored_reward": stored,
            "production_decision": decision.value,
            "production_reason": decision.reason,
            "answer_source": answer_source,
            "answer_extracted_reconstructed": answer,
            "oracle": oracle,
            "context_error_affected": context_error,
            "zero_reward_cause": cause if stored == 0 else None,
            "response_triplet_count": len(rollout.get("triplets") or []),
            "final_response_sha256": hashlib.sha256(final_text.encode("utf-8")).hexdigest(),
        }
        rows.append(row)
        groups[row["group_id"]].append(row)

    puzzle_rows = []
    for group_id, members in sorted(groups.items()):
        puzzle = tuple(members[0]["puzzle"])
        known = solve_game24(puzzle)
        known_decision = deterministic_decision("24", f"<answer>{known}</answer>") if known else None
        puzzle_rows.append({
            "group_id": group_id,
            "source_row_index": members[0]["source_row_index"],
            "puzzle": list(puzzle),
            "known_valid_expression": known,
            "production_accepts_known_valid": None if known_decision is None else known_decision.value,
            "production_reason": None if known_decision is None else known_decision.reason,
            "solvable": known is not None,
        })

    causes = collections.Counter(row["zero_reward_cause"] for row in rows if row["stored_reward"] == 0)
    disagreements = [row for row in rows if row["oracle"]["oracle_valid"] != bool(row["production_decision"])]
    group_sizes = collections.Counter(len(value) for value in groups.values())
    audit_result = {
        "schema_version": 1,
        "status": "ok",
        "mode": "offline_no_external_calls",
        "evidence_dir": str(Path(args.evidence_dir).resolve()),
        "evidence_file_count": len(files),
        "trajectory_count": len(rows),
        "group_count": len(groups),
        "group_size_counts": {str(k): v for k, v in sorted(group_sizes.items())},
        "stored_reward_counts": dict(collections.Counter(str(row["stored_reward"]) for row in rows)),
        "zero_reward_causes": dict(sorted(causes.items())),
        "production_oracle_disagreement_count": len(disagreements),
        "solvability": {
            "groups_with_known_solution": sum(item["solvable"] for item in puzzle_rows),
            "groups_without_known_solution": sum(not item["solvable"] for item in puzzle_rows),
            "known_solution_production_accept_count": sum(item["production_accepts_known_valid"] is True for item in puzzle_rows),
            "known_solution_production_reject_count": sum(item["production_accepts_known_valid"] is False for item in puzzle_rows),
        },
        "puzzles": puzzle_rows,
        "rows": rows,
    }
    Path(args.audit_output).write_text(json.dumps(audit_result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.overlay_output:
        overlay_rows = [
            {
                "evidence_file": row["evidence_file"],
                "rollout_id": row["rollout_id"],
                "group_id": row["group_id"],
                "stored_reward": row["stored_reward"],
                "recomputed_deterministic_reward": int(row["production_decision"] is True),
                "production_reason": row["production_reason"],
                "oracle_valid": row["oracle"]["oracle_valid"],
            }
            for row in rows
        ]
        overlay = {
            "schema_version": 1,
            "kind": "offline_reward_overlay_never_written_back",
            "source_audit": str(Path(args.audit_output)),
            "rows": overlay_rows,
            "stored_positive_count": sum(row["stored_reward"] > 0 for row in overlay_rows),
            "recomputed_positive_count": sum(row["recomputed_deterministic_reward"] > 0 for row in overlay_rows),
        }
        Path(args.overlay_output).write_text(json.dumps(overlay, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Select strictly from persisted length evidence.  Reward fields are not
    # read in this block; this makes the selection rule auditable.
    selection = []
    for group_id, members in groups.items():
        lengths = []
        for path in files:
            evidence = json.loads(path.read_text(encoding="utf-8"))
            if evidence["original_sample"].get("data_id") != group_id:
                continue
            for triplet in evidence["rollout"].get("triplets") or []:
                response = triplet.get("response") or {}
                prompt = triplet.get("prompt") or {}
                response_ids = response.get("response_token_ids") or response.get("token_ids") or []
                prompt_ids = prompt.get("token_ids") or []
                lengths.append((len(response_ids), len(prompt_ids)))
        selection.append({
            "opaque_group_id": hashlib.sha256(f"len-only:{group_id}".encode()).hexdigest()[:16],
            "group_id": group_id,
            "source_row_index": members[0]["source_row_index"],
            "max_response_tokens": max((item[0] for item in lengths), default=0),
            "max_prompt_tokens": max((item[1] for item in lengths), default=0),
            "has_response_1024": any(item[0] == 1024 for item in lengths),
        })
    selection.sort(key=lambda item: (-int(item["has_response_1024"]), -item["max_response_tokens"], -item["max_prompt_tokens"], item["group_id"]))
    selected = selection[: min(args.max_groups, len(selection))]
    selection_manifest = {
        "schema_version": 1,
        "status": "selected_before_generation",
        "purpose": "length_context_only_2048_8192_probe",
        "selection_seed": args.probe_seed,
        "selection_rule": "sort persisted groups by has_response_1024 desc, max_response_tokens desc, max_prompt_tokens desc, group_id asc; no reward/outcome fields read",
        "source_run": Path(args.evidence_dir).name,
        "selected_group_count": len(selected),
        "selected_groups": selected,
        "all_group_length_metadata": selection,
    }
    Path(args.selection_output).write_text(json.dumps(selection_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"audit": audit_result, "selection": selection_manifest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument("--selection-output", required=True, type=Path)
    parser.add_argument("--overlay-output", type=Path)
    parser.add_argument("--max-groups", type=int, default=8)
    parser.add_argument("--probe-seed", type=int, default=20260829)
    args = parser.parse_args()
    audit(args)
    print(json.dumps({"status": "ok", "audit": str(args.audit_output), "selection": str(args.selection_output)}))


if __name__ == "__main__":
    main()
