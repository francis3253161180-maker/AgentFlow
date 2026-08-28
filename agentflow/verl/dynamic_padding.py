"""Small, deterministic response-padding planner for AgentFlow batches.

The generation and training caps remain hard limits.  This module only decides
the width used by the current batch after over-cap responses have been clipped.
It is intentionally independent of model, dataset, and reward semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ResponsePaddingPlan:
    hard_cap: int
    raw_max: int
    effective_width: int
    raw_lengths: tuple[int, ...]
    clipped_lengths: tuple[int, ...]

    @property
    def batch_size(self) -> int:
        return len(self.raw_lengths)

    @property
    def fixed_elements(self) -> int:
        return self.batch_size * self.hard_cap

    @property
    def dynamic_elements(self) -> int:
        return self.batch_size * self.effective_width

    @property
    def padding_elements_saved(self) -> int:
        return self.fixed_elements - self.dynamic_elements

    @property
    def padding_ratio(self) -> float:
        if self.dynamic_elements == 0:
            return 0.0
        return sum(self.effective_width - length for length in self.clipped_lengths) / self.dynamic_elements


def make_response_padding_plan(raw_lengths: Iterable[int], hard_cap: int) -> ResponsePaddingPlan:
    lengths = tuple(int(length) for length in raw_lengths)
    if hard_cap <= 0:
        raise ValueError("hard_cap must be positive")
    if any(length < 0 for length in lengths):
        raise ValueError("response lengths must be non-negative")
    raw_max = max(lengths, default=0)
    clipped = tuple(min(length, hard_cap) for length in lengths)
    effective = min(raw_max, hard_cap) if lengths else 0
    return ResponsePaddingPlan(
        hard_cap=int(hard_cap),
        raw_max=raw_max,
        effective_width=effective,
        raw_lengths=lengths,
        clipped_lengths=clipped,
    )
