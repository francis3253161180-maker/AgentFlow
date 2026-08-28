"""Typed, deterministic three-action Game24 environment.

This module deliberately contains no language-model or answer-string parsing.
The planner emits :class:`AtomicAction`; this module validates the action and
performs all arithmetic with exact :class:`fractions.Fraction` values.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from pydantic import BaseModel, ConfigDict, StrictStr


Operator = Literal["+", "-", "*", "/"]


class AtomicAction(BaseModel):
    """One typed binary combine action; extra fields are not accepted."""

    model_config = ConfigDict(extra="forbid")
    left_id: StrictStr
    operator: Operator
    right_id: StrictStr


@dataclass(frozen=True)
class AtomicNode:
    node_id: str
    value: Fraction
    expression: str
    provenance: tuple[int, ...]


class AtomicState:
    """Mutable active-node state with deterministic transitions."""

    def __init__(self, numbers: tuple[int, int, int, int]):
        if len(numbers) != 4:
            raise ValueError("Game24 atomic state requires exactly four numbers")
        self.numbers = numbers
        self.active: dict[str, AtomicNode] = {
            f"n{i}": AtomicNode(f"n{i}", Fraction(value), str(value), (i,))
            for i, value in enumerate(numbers)
        }
        self._next_id = 4

    def snapshot(self) -> list[dict[str, object]]:
        return [
            {
                "node_id": node.node_id,
                "value": str(node.value),
                "expression": node.expression,
                "provenance": list(node.provenance),
            }
            for node in self.active.values()
        ]

    def apply(self, action: AtomicAction) -> AtomicNode:
        if action.left_id == action.right_id:
            raise ValueError("operands must be distinct")
        left = self.active.get(action.left_id)
        right = self.active.get(action.right_id)
        if left is None or right is None:
            raise ValueError("both operands must be active node IDs")
        if action.operator == "+":
            value = left.value + right.value
        elif action.operator == "-":
            value = left.value - right.value
        elif action.operator == "*":
            value = left.value * right.value
        else:
            if right.value == 0:
                raise ValueError("division by zero is not allowed")
            value = left.value / right.value
        node = AtomicNode(
            node_id=f"n{self._next_id}",
            value=value,
            expression=f"({left.expression}{action.operator}{right.expression})",
            provenance=tuple(sorted(left.provenance + right.provenance)),
        )
        del self.active[left.node_id]
        del self.active[right.node_id]
        self.active[node.node_id] = node
        self._next_id += 1
        return node

    def terminal_reward(self) -> int:
        if len(self.active) != 1:
            return 0
        node = next(iter(self.active.values()))
        return int(node.value == 24 and node.provenance == (0, 1, 2, 3))


def parse_atomic_action(raw: object) -> AtomicAction:
    """Parse exactly one JSON object; do not extract or repair free-form text."""

    import json

    if not isinstance(raw, str):
        raise ValueError("planner response must be a string")
    text = raw.strip()
    if not text.startswith("{") or not text.endswith("}"):
        raise ValueError("planner response must be one JSON object")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("planner response is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("planner response must decode to an object")
    if hasattr(AtomicAction, "model_validate"):
        return AtomicAction.model_validate(payload)
    return AtomicAction.parse_obj(payload)


def extract_game24_numbers(question: str) -> tuple[int, int, int, int]:
    import re

    match = re.search(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]", question)
    if not match:
        raise ValueError("question does not contain four Game24 numbers")
    return tuple(int(item) for item in match.groups())  # type: ignore[return-value]
