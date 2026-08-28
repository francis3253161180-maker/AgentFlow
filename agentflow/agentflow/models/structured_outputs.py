"""Strict, local structured-output adapters used by the optional harness.

The schemas are intentionally small.  Parsing is strict and validation never
repairs or solves a candidate; it only proves whether the supplied expression
is legal for the supplied Game24 puzzle.
"""

from __future__ import annotations

import ast
import json
import re
from fractions import Fraction
from typing import Any

from pydantic import BaseModel, ConfigDict, StrictStr, ValidationError


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Game24Answer(_StrictModel):
    expression: StrictStr


class StructuredToolCall(_StrictModel):
    """Small schema for a tool invocation emitted by a role."""

    tool_name: StrictStr
    query: StrictStr


class StructuredToolResponse(_StrictModel):
    """Small schema for a tool response envelope."""

    success: bool
    result: StrictStr


class StructuredVerifierFeedback(_StrictModel):
    """Verifier feedback schema for structured retry decisions."""

    stop: bool
    reason: StrictStr


def model_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    if hasattr(model, "model_json_schema"):
        return model.model_json_schema()
    return model.schema()


def parse_strict_json(raw: Any, model: type[BaseModel]) -> BaseModel:
    """Accept exactly one JSON object and validate it with Pydantic.

    Markdown fences, leading/trailing prose, arrays, and extra keys are
    rejected.  This is intentionally stricter than the legacy role parsers.
    """
    if isinstance(raw, BaseModel):
        payload = raw.model_dump() if hasattr(raw, "model_dump") else raw.dict()
    elif isinstance(raw, dict):
        payload = raw
    elif isinstance(raw, str):
        text = raw.strip()
        if not text or text[0] != "{" or text[-1] != "}":
            raise ValueError("structured response must be one JSON object")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("structured response is invalid JSON") from exc
    else:
        raise ValueError("structured response has unsupported type")
    if not isinstance(payload, dict):
        raise ValueError("structured response must decode to an object")
    try:
        if hasattr(model, "model_validate"):
            return model.model_validate(payload)
        return model.parse_obj(payload)
    except ValidationError as exc:
        raise ValueError("structured response failed schema validation") from exc


def _clean_expression(expression: str) -> str:
    text = expression.strip()
    text = text.replace("×", "*").replace("÷", "/")
    text = re.sub(r"\\boxed\s*\{(.*)\}\s*$", r"\1", text, flags=re.DOTALL)
    text = text.replace("\\(", "").replace("\\)", "")
    text = text.strip("` $\\")
    # A structured expression may include the asserted result.  Only remove a
    # single trailing '=24'; other equalities remain invalid.
    text = re.sub(r"\s*=\s*24\s*$", "", text)
    return text.strip()


def validate_game24_expression(expression: str, numbers: list[int] | tuple[int, ...]) -> dict[str, Any]:
    """Validate, without repair, one basic-arithmetic Game24 expression."""
    cleaned = _clean_expression(expression)
    if not cleaned:
        return {"valid": False, "reason": "empty_expression", "used_numbers": []}
    if len(numbers) != 4:
        return {"valid": False, "reason": "expected_four_input_numbers", "used_numbers": []}
    if not re.fullmatch(r"[0-9+*/() .-]+", cleaned):
        return {"valid": False, "reason": "unsupported_syntax", "used_numbers": []}
    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError:
        return {"valid": False, "reason": "invalid_expression", "used_numbers": []}
    used: list[int] = []

    def evaluate(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and type(node.value) is int:
            used.append(node.value)
            return Fraction(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if right == 0:
                raise ZeroDivisionError
            return left / right
        raise ValueError("unsupported AST node")

    try:
        value = evaluate(tree)
    except ZeroDivisionError:
        return {"valid": False, "reason": "divide_by_zero", "used_numbers": used}
    except (ValueError, TypeError, RecursionError):
        return {"valid": False, "reason": "unsupported_expression", "used_numbers": used}
    if sorted(used) != sorted(numbers):
        return {"valid": False, "reason": "wrong_number_multiset", "used_numbers": used, "value": str(value)}
    if value != 24:
        return {"valid": False, "reason": "not_24", "used_numbers": used, "value": str(value)}
    return {"valid": True, "reason": "proved_fraction_24", "used_numbers": used, "value": "24"}


def parse_game24_answer(raw: Any, numbers: list[int] | tuple[int, ...]) -> tuple[Game24Answer | None, dict[str, Any]]:
    try:
        answer = parse_strict_json(raw, Game24Answer)
    except ValueError as exc:
        return None, {"valid": False, "reason": "schema_parse_failure", "detail": str(exc)}
    validation = validate_game24_expression(answer.expression, numbers)
    if not validation["valid"]:
        return None, validation
    return answer, validation


def game24_reward_decision(question: str, answer: Any) -> tuple[bool | None, dict[str, Any]]:
    """Return a strict Game24 reward decision when the task is identifiable.

    JSON schema output is preferred.  For legacy callers, only explicitly
    marked answer candidates are accepted; arbitrary prose is never promoted.
    ``None`` means the question is not identifiable as a four-number Game24
    task and the general scorer may handle it.
    """
    numbers = extract_game24_numbers(question)
    if numbers is None:
        return None, {"reason": "not_identifiable_game24"}
    parsed, validation = parse_game24_answer(answer, numbers)
    if parsed is not None:
        return True, validation
    for candidate in candidate_expressions(str(answer)):
        checked = validate_game24_expression(candidate, numbers)
        if checked["valid"]:
            return True, checked
    return False, validation


def game24_prompt(question: str, memory: str, feedback: str | None = None) -> str:
    suffix = ""
    if feedback:
        suffix = f"\nStructured retry feedback: {feedback}\n"
    return f"""Return one JSON object and nothing else.
Schema: {{"expression": "string"}}
Hard rules: use each of the four puzzle numbers exactly once; use only +, -, *, / and parentheses; the exact Fraction result must be 24; do not include prose, markdown, or an equals sign.
Examples:
Puzzle [1, 2, 3, 4] -> {{"expression":"(1+2+3)*4"}}
Puzzle [3, 3, 8, 8] -> {{"expression":"8/(3-8/3)"}}
Puzzle [1, 5, 5, 5] -> {{"expression":"5*(5-1/5)"}}

Question:
{question}

Prior agent memory (untrusted evidence, not a schema):
{memory}
{suffix}"""


def extract_game24_numbers(question: str) -> tuple[int, ...] | None:
    match = re.search(r"numbers\s*(?::|=)?\s*\[([^]]+)\]", question, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        values = tuple(int(item.strip()) for item in match.group(1).split(","))
    except ValueError:
        return None
    return values if len(values) == 4 else None


def candidate_expressions(text: str) -> list[str]:
    """Return marked candidate strings only; never promote arbitrary prose."""
    patterns = (
        r"<answer>\s*(.*?)\s*</answer>",
        r"(?:final\s+answer|answer|expression)\s*[:=]\s*([^\n]+)",
    )
    found: list[str] = []
    for pattern in patterns:
        found.extend(re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL))
    return [item.strip() for item in found if item.strip()]


def select_valid_candidate(text: str, numbers: tuple[int, ...]) -> tuple[str | None, dict[str, Any]]:
    for candidate in candidate_expressions(text):
        result = validate_game24_expression(candidate, numbers)
        if result["valid"]:
            return candidate, result
    return None, {"valid": False, "reason": "no_marked_valid_candidate"}
