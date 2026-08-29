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


def stable_step_id(current_step: dict[str, Any]) -> str:
    """The activated atomic plan step is the stable target for its lifetime."""
    return str(current_step.get("step_id") or "current_step")


def build_current_step_contract(
    current_step: dict[str, Any], memory_actions: Any,
    prior_attempts: list[dict[str, Any]], last_verifier_assessment: Any,
) -> dict[str, Any]:
    """Serialize exactly the evidence state the next Planner action may use."""
    return {
        "active_step": {
            "step_id": stable_step_id(current_step),
            "objective": current_step.get("objective"),
            "success_criteria": current_step.get("success_criteria"),
        },
        "stable_step_id": stable_step_id(current_step),
        "active_goal": current_step.get("objective") or current_step.get("success_criteria"),
        # Free-text verifier gaps are useful diagnostics, not identities.  A
        # paraphrase must not create a new target or imply resolution.
        "missing_evidence_diagnostics": list(current_step.get("missing_evidence", [])),
        "verified_evidence": list(current_step.get("verified_evidence", [])),
        "known_urls": known_urls(memory_actions),
        "prior_attempts_for_active_step": prior_attempts,
        "last_verifier_assessment": last_verifier_assessment if last_verifier_assessment is not None else "none (first action)",
    }


def objective_signature(value: Any) -> str:
    """Conservative canonical form: only clear textual equivalence is merged."""
    return _normalized(value)


def should_revise_stagnant_action(
    *, tool_name: Any, stable_step_id: Any, executable_signature: Any, prior_attempts: list[dict[str, Any]],
) -> tuple[bool, str]:
    """Reject one purposeless repetition after a no-progress attempt.

    A near-but-not-identical objective is intentionally allowed; the guard is
    not a tool-diversity policy and should not reject a genuinely reformulated
    retrieval attempt.
    """
    signature = str(executable_signature)
    for attempt in reversed(prior_attempts):
        if attempt.get("made_progress"):
            continue
        if (
            str(attempt.get("tool_name")) == str(tool_name)
            and str(attempt.get("stable_step_id")) == str(stable_step_id)
            and attempt.get("executable_signature") == signature
        ):
            return True, (
                "The proposed action repeats the same tool and executable retrieval intent "
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
    gaps_changed = before_missing != after_missing
    completed_changed = before_step.get("status") != after_step.get("status")
    return {
        "evidence_added": evidence_added,
        "verified_evidence_before": sorted(before_verified),
        "verified_evidence_after": sorted(after_verified),
        "missing_evidence_before": sorted(before_missing),
        "missing_evidence_after": sorted(after_missing),
        "missing_evidence_changed": gaps_changed,
        "completed_changed": completed_changed,
        # Missing-evidence text can be paraphrased across verifier turns; it
        # never by itself proves a gap was resolved.
        "made_progress": bool(evidence_added or completed_changed),
        "reason": "verifier added evidence or changed step status" if (evidence_added or completed_changed)
        else "verifier evidence/gap state did not materially change",
    }


def executable_signature(tool_name: Any, command: Any, context: Any, sub_goal: Any) -> str:
    """Prefer the actual retrieval query/URL over planner prose for duplicate checks."""
    text = str(command)
    query = re.search(r"\bquery\s*=\s*(['\"])(.*?)\1", text, flags=re.DOTALL)
    url = re.search(r"\burl\s*=\s*(['\"])(.*?)\1", text, flags=re.DOTALL)
    if query:
        intent = f"query:{query.group(2)}"
    elif url:
        intent = f"url:{url.group(2)}"
    elif text.strip():
        intent = f"command:{text}"
    else:
        intent = f"proposal:{context}|{sub_goal}"
    return f"{str(tool_name)}::{objective_signature(intent)}"
