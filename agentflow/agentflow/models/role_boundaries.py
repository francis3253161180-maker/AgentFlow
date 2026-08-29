"""Generic diagnostics and sanitization for fixed-role capability boundaries.

The checks here deliberately identify structural capabilities (tool/URL/query
syntax, answer markers, and newly introduced arithmetic).  They do not decide
whether any factual claim is correct.  This lets a fixed planning or verifier
role be audited without giving it a covert action channel into planner_main.
"""

from __future__ import annotations

import json
import re
from typing import Any


_TOOL_NAME = re.compile(r"\b[A-Za-z][A-Za-z0-9]*_(?:RAG_)?[A-Za-z0-9]*Tool\b")
_URL = re.compile(r"https?://[^\s'\"<>)}\]]+", re.IGNORECASE)
_COMMAND = re.compile(r"\b(?:query|url|execution|command)\s*=|tool\.execute\s*\(", re.IGNORECASE)
_ANSWER = re.compile(r"<answer>|\b(?:final\s+answer|answer\s+is)\b", re.IGNORECASE)
_ARITHMETIC = re.compile(r"(?<![\w.])\d+(?:\.\d+)?\s*[-+*/=]\s*\d+(?:\.\d+)?(?![\w.])")


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def audit_fixed_role_output(value: Any, *, recorded_evidence: Any = None) -> dict[str, Any]:
    """Return a structural leakage diagnostic without semantic adjudication."""

    text = _text(value)
    evidence_text = _text(recorded_evidence) if recorded_evidence is not None else ""
    arithmetic = sorted(set(_ARITHMETIC.findall(text)))
    evidence_arithmetic = set(_ARITHMETIC.findall(evidence_text))
    markers = {
        "tool_names": sorted(set(_TOOL_NAME.findall(text))),
        "urls": sorted(set(_URL.findall(text))),
        "command_or_query_syntax": bool(_COMMAND.search(text)),
        "direct_answer_language": bool(_ANSWER.search(text)),
        # This is intentionally only a textual novelty indicator.  It does
        # not claim the arithmetic is right, wrong, or an answer.
        "arithmetic_not_in_recorded_evidence": sorted(
            expression for expression in arithmetic if expression not in evidence_arithmetic
        ),
    }
    marker_count = (
        len(markers["tool_names"])
        + len(markers["urls"])
        + int(markers["command_or_query_syntax"])
        + int(markers["direct_answer_language"])
        + len(markers["arithmetic_not_in_recorded_evidence"])
    )
    return {"marker_count": marker_count, "markers": markers}


def structurally_safe_supervisor_output(audit: dict[str, Any]) -> bool:
    """Whether planning/coverage output has no prohibited action/answer channel."""

    markers = audit.get("markers", {}) if isinstance(audit, dict) else {}
    return not (
        markers.get("tool_names")
        or markers.get("urls")
        or markers.get("command_or_query_syntax")
        or markers.get("direct_answer_language")
        or markers.get("arithmetic_not_in_recorded_evidence")
    )


def sanitize_verifier_assessment(value: Any) -> dict[str, Any]:
    """Expose only evidence-state fields to planner_main, never free rationale."""

    payload = value if isinstance(value, dict) else {}
    evidence = []
    for entry in payload.get("requirement_evidence", []):
        if not isinstance(entry, dict):
            continue
        evidence.append({
            "requirement": entry.get("requirement", ""),
            "action_step_refs": list(entry.get("action_step_refs", [])),
            "evidence_quotes": list(entry.get("evidence_quotes", [])),
        })
    return {
        "completed": bool(payload.get("completed", False)),
        "missing_evidence": list(payload.get("missing_evidence", [])),
        "verified_evidence": list(payload.get("verified_evidence", [])),
        "contradiction": bool(payload.get("contradiction", False)),
        "invalidated_step_ids": list(payload.get("invalidated_step_ids", [])),
        "requirement_evidence": evidence,
    }
