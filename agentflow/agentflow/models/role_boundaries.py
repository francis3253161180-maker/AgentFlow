"""Generic diagnostics and sanitization for fixed-role capability boundaries.

The checks here deliberately identify structural capabilities (tool/URL/query
syntax, answer markers, and newly introduced arithmetic).  They do not decide
whether any factual claim is correct.  This lets a fixed planning or verifier
role be audited without giving it a covert action channel into planner_main.
"""

from __future__ import annotations

import json
import re
import hashlib
from typing import Any


_TOOL_NAME = re.compile(r"\b[A-Za-z][A-Za-z0-9_]*_Tool\b")
_URL = re.compile(r"https?://[^\s'\"<>)}\]]+", re.IGNORECASE)
_COMMAND = re.compile(r"\b(?:query|url|execution|command)\s*=|tool\.execute\s*\(", re.IGNORECASE)
_ANSWER = re.compile(r"<answer>|\b(?:final\s+answer|answer\s+is)\b", re.IGNORECASE)
_ARITHMETIC = re.compile(r"(?<![\w.])\d+(?:\.\d+)?\s*[-+*/=]\s*\d+(?:\.\d+)?(?![\w.])")


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _is_year_range(expression: str) -> bool:
    """Keep date/season ranges out of the arithmetic capability diagnostic."""

    match = re.fullmatch(r"(\d{4})-(\d{2}|\d{4})", expression.replace(" ", ""))
    return bool(match)


def audit_fixed_role_output(value: Any, *, recorded_evidence: Any = None) -> dict[str, Any]:
    """Return a structural leakage diagnostic without semantic adjudication."""

    text = _text(value)
    evidence_text = _text(recorded_evidence) if recorded_evidence is not None else ""
    arithmetic = sorted(expression for expression in set(_ARITHMETIC.findall(text)) if not _is_year_range(expression))
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


def redacted_boundary_telemetry(
    raw: Any, *, parsed: Any = None, parse_error: str | None = None,
    recorded_evidence: Any = None,
) -> dict[str, Any]:
    """Record enough rejection evidence without retaining model response text."""

    raw_text = _text(raw)
    audited_value = parsed if parsed is not None else raw_text
    audit = audit_fixed_role_output(audited_value, recorded_evidence=recorded_evidence)
    return {
        "response_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        "response_length": len(raw_text),
        "parsed_schema_success": parsed is not None,
        "parse_error_category": parse_error,
        "marker_categories": audit["markers"],
        "marker_count": audit["marker_count"],
    }


def structurally_safe_supervisor_output(audit: dict[str, Any]) -> bool:
    """Whether planning/coverage output has no prohibited action/answer channel."""

    markers = (
        audit.get("markers", audit.get("marker_categories", {}))
        if isinstance(audit, dict) else {}
    )
    return not (
        markers.get("tool_names")
        or markers.get("urls")
        or markers.get("command_or_query_syntax")
        or markers.get("direct_answer_language")
        or markers.get("arithmetic_not_in_recorded_evidence")
    )


def boundary_categories(telemetry: dict[str, Any]) -> list[str]:
    """Return only category labels for supervisor self-revision feedback."""

    categories: list[str] = []
    if not telemetry.get("parsed_schema_success", False):
        categories.append("schema")
    for name, value in (telemetry.get("marker_categories", {}) or {}).items():
        if value:
            categories.append(str(name))
    return categories


def sanitize_verifier_assessment(value: Any, *, recorded_evidence: Any = None) -> dict[str, Any]:
    """Expose only evidence-state fields to planner_main, never free rationale."""

    payload = value if isinstance(value, dict) else {}

    def safe_texts(items: Any) -> list[str]:
        kept: list[str] = []
        for item in items if isinstance(items, list) else []:
            text = str(item)
            audit = audit_fixed_role_output(text, recorded_evidence=recorded_evidence)
            if structurally_safe_supervisor_output(audit):
                kept.append(text)
        return kept

    evidence = []
    for entry in payload.get("requirement_evidence", []):
        if not isinstance(entry, dict):
            continue
        requirement = str(entry.get("requirement", ""))
        audit = audit_fixed_role_output(requirement, recorded_evidence=recorded_evidence)
        if structurally_safe_supervisor_output(audit):
            evidence.append({
                "requirement": requirement,
                "action_step_refs": safe_texts(entry.get("action_step_refs", [])),
                "evidence_quotes": safe_texts(entry.get("evidence_quotes", [])),
            })
    return {
        "completed": bool(payload.get("completed", False)),
        "missing_evidence": safe_texts(payload.get("missing_evidence", [])),
        "verified_evidence": safe_texts(payload.get("verified_evidence", [])),
        "contradiction": bool(payload.get("contradiction", False)),
        "invalidated_step_ids": list(payload.get("invalidated_step_ids", [])),
        "requirement_evidence": evidence,
    }
