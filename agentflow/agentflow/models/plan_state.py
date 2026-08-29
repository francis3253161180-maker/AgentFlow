"""Deterministic state transitions for optional hierarchical AgentFlow plans."""

from __future__ import annotations

import copy
import re
from typing import Any


VALID_STATUSES = {"pending", "in_progress", "completed", "failed"}


def _as_mapping(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    elif hasattr(value, "dict"):
        value = value.dict()
    return value if isinstance(value, dict) else {}


def normalize_plan(plan: Any, max_steps: int) -> dict[str, Any]:
    """Create a serializable pending-step plan without task-specific repairs."""
    payload = _as_mapping(plan)
    source_steps = payload.get("steps", [])
    if not isinstance(source_steps, list) or not source_steps:
        raise ValueError("hierarchical plan must contain at least one step")
    if len(source_steps) > max_steps:
        source_steps = source_steps[:max_steps]

    steps: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ordinal, raw_step in enumerate(source_steps, start=1):
        raw = _as_mapping(raw_step)
        step_id = str(raw.get("step_id", "")).strip() or f"step_{ordinal}"
        if step_id in seen:
            step_id = f"{step_id}_{ordinal}"
        seen.add(step_id)
        depends_on = raw.get("depends_on", [])
        if not isinstance(depends_on, list):
            depends_on = []
        steps.append({
            "step_id": step_id,
            "objective": str(raw.get("objective", "")).strip(),
            "success_criteria": str(raw.get("success_criteria", "")).strip(),
            "depends_on": [str(item) for item in depends_on],
            "status": "pending",
            "verified_evidence": [],
            "missing_evidence": [],
        })
    if any(not step["objective"] or not step["success_criteria"] for step in steps):
        raise ValueError("hierarchical plan steps require objective and success_criteria")
    known_ids = {step["step_id"] for step in steps}
    for step in steps:
        step["depends_on"] = [dep for dep in step["depends_on"] if dep in known_ids and dep != step["step_id"]]
    return {"steps": steps}


def snapshot(plan: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(plan)


def attach_requirement_coverage(plan: dict[str, Any], coverage: dict[str, Any]) -> dict[str, list[str]]:
    """Persist the fixed coverage audit's requirement-to-step mapping in plan state."""
    mapping: dict[str, list[str]] = {}
    for item in coverage.get("requirement_coverage", []) if isinstance(coverage, dict) else []:
        if not isinstance(item, dict):
            continue
        requirement = str(item.get("requirement", "")).strip()
        refs = item.get("covered_step_ids", [])
        if requirement and isinstance(refs, list):
            mapping[requirement] = [str(step_id) for step_id in refs]
    plan["requirement_to_step"] = mapping
    return mapping


def requirements_for_step(plan: dict[str, Any], step_id: str) -> list[str]:
    """Return only requirements explicitly mapped to this active atomic step."""
    mapping = plan.get("requirement_to_step", {})
    if not isinstance(mapping, dict):
        return []
    return sorted(
        str(requirement) for requirement, step_ids in mapping.items()
        if isinstance(step_ids, list) and str(step_id) in {str(value) for value in step_ids}
    )


def validate_grounded_step_completion(
    verification: dict[str, Any], active_step_id: str, plan: dict[str, Any], memory_actions: dict[str, Any],
) -> dict[str, Any]:
    """Check only provenance structure for a verifier-requested completion.

    This deliberately does not decide whether an evidence quote semantically
    proves a requirement.  The fixed verifier makes that judgement; this gate
    only requires its claimed evidence to name existing Memory actions and
    quote text traceable to their raw tool outputs.
    """
    if not bool(verification.get("completed")):
        return {"accepted": True, "reason": "completion_not_requested", "active_requirements": requirements_for_step(plan, active_step_id)}
    requirements = requirements_for_step(plan, active_step_id)
    if not requirements:
        return {
            "accepted": False,
            "reason": "active_step_has_no_requirement_mapping",
            "active_requirements": [],
            "rejections": ["no requirement-to-step mapping was available for completed step"],
        }
    entries = verification.get("requirement_evidence", [])
    if not isinstance(entries, list):
        entries = []
    by_requirement: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        if isinstance(entry, dict):
            by_requirement.setdefault(_normalized(entry.get("requirement")), []).append(entry)
    rejections: list[str] = []
    accepted_evidence: dict[str, list[dict[str, Any]]] = {}
    for requirement in requirements:
        matching = by_requirement.get(_normalized(requirement), [])
        if not matching:
            rejections.append(f"missing provenance entry for mapped requirement: {requirement}")
            continue
        requirement_evidence: list[dict[str, Any]] = []
        for entry in matching:
            refs = entry.get("action_step_refs", [])
            quotes = entry.get("evidence_quotes", [])
            if not isinstance(refs, list) or not refs or not isinstance(quotes, list) or not quotes:
                rejections.append(f"requirement lacks action refs or evidence quotes: {requirement}")
                continue
            missing_refs = [str(ref) for ref in refs if str(ref) not in memory_actions]
            if missing_refs:
                rejections.append(f"unknown Memory action refs for {requirement}: {missing_refs}")
                continue
            source_text = "\n".join(
                str((memory_actions[str(ref)] or {}).get("result", ""))
                for ref in refs if isinstance(memory_actions.get(str(ref)), dict)
            )
            traceable_quotes = []
            for quote in quotes:
                normalized_quote = _normalized(quote)
                if len(normalized_quote) < 8 or normalized_quote not in _normalized(source_text):
                    rejections.append(f"untraceable evidence quote for {requirement}: {str(quote)[:160]}")
                    continue
                traceable_quotes.append(str(quote))
            if traceable_quotes:
                requirement_evidence.append({
                    "action_step_refs": [str(ref) for ref in refs],
                    "evidence_quotes": traceable_quotes,
                })
        if not requirement_evidence:
            rejections.append(f"no traceable evidence remained for mapped requirement: {requirement}")
        else:
            accepted_evidence[requirement] = requirement_evidence
    return {
        "accepted": not rejections,
        "reason": "all active-step requirement evidence is traceable" if not rejections else "grounded_completion_provenance_failed",
        "active_requirements": requirements,
        "accepted_evidence": accepted_evidence,
        "rejections": rejections,
    }


def _normalized(value: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(value).casefold())).strip()


def all_steps_completed(plan: dict[str, Any]) -> bool:
    steps = plan.get("steps", [])
    return bool(steps) and all(step.get("status") == "completed" for step in steps)


def current_step(plan: dict[str, Any]) -> dict[str, Any] | None:
    active = [step for step in plan.get("steps", []) if step.get("status") == "in_progress"]
    if len(active) > 1:
        raise ValueError("hierarchical plan has more than one in_progress step")
    return active[0] if active else None


def activate_next_step(plan: dict[str, Any], transitions: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Activate one dependency-satisfied pending step, preserving completed work."""
    active = current_step(plan)
    if active is not None:
        return active
    completed = {step["step_id"] for step in plan.get("steps", []) if step.get("status") == "completed"}
    for step in plan.get("steps", []):
        if step.get("status") == "pending" and set(step.get("depends_on", [])).issubset(completed):
            step["status"] = "in_progress"
            transitions.append({"event": "activate", "step_id": step["step_id"], "reason": "dependencies_satisfied"})
            return step
    return None


def apply_step_verification(
    plan: dict[str, Any],
    active_step_id: str,
    verification: dict[str, Any],
    transitions: list[dict[str, Any]],
) -> None:
    """Apply evidence-only verification to the current step and explicit reopens."""
    steps_by_id = {step["step_id"]: step for step in plan.get("steps", [])}
    step = steps_by_id.get(active_step_id)
    if step is None or step.get("status") != "in_progress":
        raise ValueError("verification must apply to the in-progress step")
    step["verified_evidence"] = [str(item) for item in verification.get("verified_evidence", [])]
    step["missing_evidence"] = [str(item) for item in verification.get("missing_evidence", [])]
    if bool(verification.get("completed")):
        step["status"] = "completed"
        transitions.append({"event": "complete", "step_id": active_step_id, "reason": verification.get("rationale", "")})
    else:
        transitions.append({"event": "remain_in_progress", "step_id": active_step_id, "reason": verification.get("rationale", "")})

    if bool(verification.get("contradiction")):
        reason = verification.get("rationale", "explicit verifier contradiction")
        for step_id in verification.get("invalidated_step_ids", []):
            invalidated = steps_by_id.get(str(step_id))
            if invalidated is not None and invalidated.get("status") == "completed":
                invalidated["status"] = "pending"
                invalidated["missing_evidence"] = ["reopened after explicit contradiction"]
                transitions.append({"event": "reopen", "step_id": invalidated["step_id"], "reason": reason})


def unresolved_steps(plan: dict[str, Any]) -> list[dict[str, Any]]:
    return [step for step in plan.get("steps", []) if step.get("status") != "completed"]
