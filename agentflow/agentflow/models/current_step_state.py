"""Deterministic current-step state for hierarchical planner actions.

The module intentionally avoids task, dataset, and tool-order rules.  It only
describes what is already in memory and rejects an unchanged retry following a
verifier-confirmed lack of progress.
"""

from __future__ import annotations

import re
from typing import Any


def _normalized(value: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(value).casefold())).strip()


def known_urls(memory_actions: Any) -> list[str]:
    return sorted(set(re.findall(r"https?://[^\s'\"<>)}\]]+", str(memory_actions))))


def target_gaps(current_step: dict[str, Any]) -> list[dict[str, str]]:
    """Return stable IDs for verifier gaps, or one initial objective gap."""
    step_id = str(current_step.get("step_id", "current_step"))
    missing = [str(item).strip() for item in current_step.get("missing_evidence", []) if str(item).strip()]
    values = missing or [str(current_step.get("success_criteria") or current_step.get("objective") or "current objective").strip()]
    prefix = "missing_evidence" if missing else "initial_objective"
    return [
        {"id": f"{step_id}::{prefix}::{ordinal}", "value": value}
        for ordinal, value in enumerate(values, start=1)
    ]


def build_current_step_contract(
    current_step: dict[str, Any], memory_actions: Any,
    prior_attempts: list[dict[str, Any]], last_verifier_assessment: Any,
) -> dict[str, Any]:
    """Serialize exactly the evidence state the next Planner action may use."""
    return {
        "active_step": {
            "step_id": current_step.get("step_id"),
            "objective": current_step.get("objective"),
            "success_criteria": current_step.get("success_criteria"),
        },
        "unresolved_evidence_gaps": target_gaps(current_step),
        "verified_evidence": list(current_step.get("verified_evidence", [])),
        "known_urls": known_urls(memory_actions),
        "prior_attempts_for_active_step": prior_attempts,
        "last_verifier_assessment": last_verifier_assessment if last_verifier_assessment is not None else "none (first action)",
    }


def target_gap_is_current(target_gap: Any, contract: dict[str, Any]) -> bool:
    return str(target_gap).strip() in {
        item["id"] for item in contract.get("unresolved_evidence_gaps", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }


def objective_signature(value: Any) -> str:
    """Conservative canonical form: only clear textual equivalence is merged."""
    return _normalized(value)


def should_revise_stagnant_action(
    *, tool_name: Any, target_gap: Any, sub_goal: Any, prior_attempts: list[dict[str, Any]],
) -> tuple[bool, str]:
    """Reject one purposeless repetition after a no-progress attempt.

    A near-but-not-identical objective is intentionally allowed; the guard is
    not a tool-diversity policy and should not reject a genuinely reformulated
    retrieval attempt.
    """
    signature = objective_signature(sub_goal)
    for attempt in reversed(prior_attempts):
        if attempt.get("made_progress"):
            continue
        if (
            str(attempt.get("tool_name")) == str(tool_name)
            and str(attempt.get("target_gap")) == str(target_gap)
            and attempt.get("objective_signature") == signature
        ):
            return True, (
                "The proposed action repeats the same tool, target gap, and retrieval objective "
                "after the recorded attempt made no evidence progress. Revise the target or objective."
            )
    return False, ""


def assess_step_progress(before_step: dict[str, Any], after_step: dict[str, Any]) -> dict[str, Any]:
    """Report verifier-observed evidence/gap change without semantic inference."""
    before_verified = {_normalized(item) for item in before_step.get("verified_evidence", []) if _normalized(item)}
    after_verified = {_normalized(item) for item in after_step.get("verified_evidence", []) if _normalized(item)}
    before_missing = {_normalized(item) for item in before_step.get("missing_evidence", []) if _normalized(item)}
    after_missing = {_normalized(item) for item in after_step.get("missing_evidence", []) if _normalized(item)}
    evidence_added = sorted(after_verified - before_verified)
    gaps_resolved = sorted(before_missing - after_missing)
    gaps_changed = before_missing != after_missing
    completed_changed = before_step.get("status") != after_step.get("status")
    return {
        "evidence_added": evidence_added,
        "gaps_resolved": gaps_resolved,
        "missing_evidence_changed": gaps_changed,
        "completed_changed": completed_changed,
        "made_progress": bool(evidence_added or gaps_resolved or completed_changed),
        "reason": "verifier evidence/gap state changed" if (evidence_added or gaps_resolved or completed_changed)
        else "verifier evidence/gap state did not materially change",
    }
