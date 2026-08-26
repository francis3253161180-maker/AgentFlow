#!/usr/bin/env python3
"""Build a conservative, reviewed quality filter for the fixed 100 groups.

The review map is deliberately small and explicit.  It contains only defects
already supported by the full offline semantic audit; model difficulty and
model mistakes remain eligible.  No rollout or judge call is made.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


# Primary categories are mutually exclusive for reporting.  ``categories``
# preserves a second supported aspect where useful (for example a missing
# time anchor plus a metric-definition ambiguity).
REVIEW_MAP: dict[str, dict[str, Any]] = {
    "mathhard:9398": {
        "primary_category": "gt_defect",
        "categories": ["gt_defect"],
        "reason": "The supplied Klein bottle minimum is 6; the standard minimal triangulation has 8 vertices.",
        "evidence": "Full audit manual review identified a benchmark ground-truth error; all four answers gave 8.",
    },
    "mathhard:44830": {
        "primary_category": "definition_sensitive",
        "categories": ["definition_sensitive", "ambiguous_question"],
        "reason": "Inconsistent premises make the requested status and formal -2 result non-binary.",
        "evidence": "Full audit marked 3/4 rows ambiguous because the prompt premises are inconsistent.",
    },
    "mathhard:50939": {
        "primary_category": "underspecified",
        "categories": ["underspecified", "ambiguous_question"],
        "reason": "T(x,y) is not specified, so a unique numeric rate cannot be inferred.",
        "evidence": "All four full-audit rows were ambiguous for the same missing-function-definition reason.",
    },
    "mathhard:55524": {
        "primary_category": "definition_sensitive",
        "categories": ["definition_sensitive", "ambiguous_question"],
        "reason": "The domain interpretation changes whether the expression is defined and whether b>0 is a sufficient answer.",
        "evidence": "Full audit marked the row ambiguous because real-domain and intended formal interpretations differ.",
    },
    "mathhard:80320": {
        "primary_category": "definition_sensitive",
        "categories": ["definition_sensitive", "ambiguous_question"],
        "reason": "Real-valued and formal/complex interpretations produce different readings of the negative-base expression.",
        "evidence": "Full audit marked one row ambiguous and retained the remaining answers as interpretation-dependent.",
    },
    "mathhard:85528": {
        "primary_category": "underspecified",
        "categories": ["underspecified", "ambiguous_question"],
        "reason": "The prompt does not provide enough geometric setup or a value of x for a unique numeric answer.",
        "evidence": "Full audit marked one row ambiguous for missing geometric/domain context.",
    },
    "nq:25306": {
        "primary_category": "stale_or_time_sensitive",
        "categories": ["stale_or_time_sensitive", "ambiguous_question"],
        "reason": "The cabinet-office question has no date anchor and the office holder changes over time.",
        "evidence": "Full audit found the supplied GT Carla Qualtrough conflicts with later historical/current answers without a time anchor.",
    },
    "nq:26032": {
        "primary_category": "underspecified",
        "categories": ["underspecified", "ambiguous_question"],
        "reason": "The phrase does not identify a work or source, so multiple unrelated interpretations remain possible.",
        "evidence": "Full audit marked all four rows ambiguous because the source context is missing.",
    },
    "nq:27020": {
        "primary_category": "ambiguous_question",
        "categories": ["ambiguous_question", "definition_sensitive"],
        "reason": "The question/GT pair does not resolve whether the requested name is the actor or the character.",
        "evidence": "Full audit found answers give actor Lee J. Cobb while GT gives character Johnny Friendly.",
    },
    "nq:39070": {
        "primary_category": "definition_sensitive",
        "categories": ["definition_sensitive", "gt_defect", "ambiguous_question"],
        "reason": "Nine symmetry axes is inconsistent with a regular hexagon's six, but regularity is not stated.",
        "evidence": "Full audit identified a possible GT defect conditional on the missing regularity assumption.",
    },
    "nq:39248": {
        "primary_category": "stale_or_time_sensitive",
        "categories": ["stale_or_time_sensitive", "ambiguous_question"],
        "reason": "The broad service-tax question has no time anchor; 14.5% and later 15% rates are both time-dependent.",
        "evidence": "Full audit marked all four rows ambiguous due to the historical-rate mismatch.",
    },
    "nq:39820": {
        "primary_category": "ambiguous_question",
        "categories": ["ambiguous_question", "underspecified"],
        "reason": "The candidate response is internally inconsistent about whether the total is six or seven coasters, preventing a clear binary audit label.",
        "evidence": "Full audit marked one row ambiguous for conflicting counts; this is retained as an unresolved item rather than a model-difficulty filter.",
    },
    "nq:4588": {
        "primary_category": "stale_or_time_sensitive",
        "categories": ["stale_or_time_sensitive", "ambiguous_question"],
        "reason": "The playoff question has no date/season anchor while the GT is 2017.",
        "evidence": "Full audit found the answers discuss later playoff appearances, so the benchmark target is not time-stable.",
    },
    "nq:50166": {
        "primary_category": "stale_or_time_sensitive",
        "categories": ["stale_or_time_sensitive", "definition_sensitive", "ambiguous_question"],
        "reason": "Debt/GDP depends on year and debt definition; neither is specified alongside GT 41.4.",
        "evidence": "Full audit found later general-government figures cannot be compared to the unanchored GT.",
    },
    "nq:69358": {
        "primary_category": "underspecified",
        "categories": ["underspecified", "ambiguous_question"],
        "reason": "The wording does not say whether one season or a two-season Wizards tenure should be aggregated.",
        "evidence": "Full audit found 20.0, 21.5, and an erroneous 21.2 under different readings.",
    },
    "nq:78405": {
        "primary_category": "definition_sensitive",
        "categories": ["definition_sensitive", "ambiguous_question"],
        "reason": "Mineral counts depend on classification and inclusion rules that are not specified.",
        "evidence": "Full audit found the answers cannot reproduce the GT category breakdown without a classification convention.",
    },
}


def bin_counts(groups: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts = Counter(str(group.get(field)) for group in groups)
    return {name: counts.get(name, 0) for name in ("0/4", "1/4", "2/4", "3/4", "4/4")}


def stats(groups: list[dict[str, Any]], rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    group_keys = {group["group_key"] for group in groups}
    scoped_rows = [row for row in rows if row["group_key"] in group_keys]
    bins = bin_counts(groups, field)
    mixed = sum(bins[name] for name in ("1/4", "2/4", "3/4"))
    by_source: dict[str, Any] = {}
    for source in ("nq", "mathhard"):
        source_groups = [group for group in groups if group["source"] == source]
        source_bins = bin_counts(source_groups, field)
        source_mixed = sum(source_bins[name] for name in ("1/4", "2/4", "3/4"))
        by_source[source] = {
            "group_count": len(source_groups),
            "bins": source_bins,
            "mixed_group_count": source_mixed,
            "mixed_ratio": source_mixed / len(source_groups) if source_groups else None,
        }
    return {
        "group_count": len(groups),
        "row_count": len(scoped_rows),
        "bins": bins,
        "mixed_group_count": mixed,
        "mixed_ratio": mixed / len(groups) if groups else None,
        "by_source": by_source,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--eligible", type=Path, required=True)
    parser.add_argument("--exclusion", type=Path, required=True)
    args = parser.parse_args()

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    groups = audit["groups"]
    rows = audit["rows"]
    group_records = []
    for group in groups:
        review = REVIEW_MAP.get(group["group_key"])
        if review is None:
            record = {
                "group_key": group["group_key"],
                "group_id": group["group_id"],
                "idx": group["idx"],
                "source": group["source"],
                "quality_status": "eligible",
                "primary_category": "clean",
                "categories": ["clean"],
                "reason": "No supported question/ground-truth quality defect was identified in the full audit.",
                "evidence": "Clear or otherwise evaluable semantic audit; scorer/model errors are not quality exclusions.",
                "original_group_class": group["original_group_class"],
                "corrected_status": group["corrected_status"],
                "corrected_class": group["corrected_class"],
                "manual_ambiguous_count": group["manual_ambiguous_count"],
            }
        else:
            record = {
                "group_key": group["group_key"],
                "group_id": group["group_id"],
                "idx": group["idx"],
                "source": group["source"],
                "quality_status": "excluded",
                **review,
                "original_group_class": group["original_group_class"],
                "corrected_status": group["corrected_status"],
                "corrected_class": group["corrected_class"],
                "manual_ambiguous_count": group["manual_ambiguous_count"],
            }
        group_records.append(record)

    eligible_groups = [group for group in group_records if group["quality_status"] == "eligible"]
    excluded_groups = [group for group in group_records if group["quality_status"] == "excluded"]
    eligible_group_data = [group for group in groups if group["group_key"] in {item["group_key"] for item in eligible_groups}]
    excluded_group_data = [group for group in groups if group["group_key"] in {item["group_key"] for item in excluded_groups}]

    eligible_manifest = {
        "manifest": "2026-08-26 outcome reward clean eligible subset",
        "version": "2026-08-26.v1",
        "source_audit_commit": "96ef69f",
        "policy": "Only clean, evaluable question/GT groups are eligible; difficulty and model errors are retained.",
        "group_count": len(eligible_groups),
        "groups": [item["group_key"] for item in eligible_groups],
    }
    exclusion_manifest = {
        "manifest": "2026-08-26 outcome reward data-quality exclusions",
        "version": "2026-08-26.v1",
        "source_audit_commit": "96ef69f",
        "group_count": len(excluded_groups),
        "groups": excluded_groups,
    }
    output = {
        "manifest": "2026-08-26 outcome reward data quality",
        "version": "2026-08-26.v1",
        "source_audit_commit": "96ef69f",
        "scope": "100 fixed prompt groups / 400 saved rollouts; no new rollout or judge call",
        "policy": {
            "eligible": "clean only",
            "excluded": "all 15 unresolved groups plus the explicit Klein bottle GT defect and conditional definition-sensitive invalidity",
            "not_excluded": "model error, difficult prompt, scorer error, or surface diversity alone",
        },
        "category_counts": dict(Counter(item["primary_category"] for item in group_records)),
        "quality_status_counts": dict(Counter(item["quality_status"] for item in group_records)),
        "groups": group_records,
        "eligible_static_stats": {
            "original": stats(eligible_group_data, rows, "original_group_class"),
            "corrected_manual": stats(eligible_group_data, rows, "corrected_class"),
        },
        "excluded_static_stats": {
            "original": stats(excluded_group_data, rows, "original_group_class"),
            "corrected_manual": stats(excluded_group_data, rows, "corrected_class"),
        },
        "eligible_manifest_path": str(args.eligible),
        "exclusion_manifest_path": str(args.exclusion),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.eligible.write_text(json.dumps(eligible_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.exclusion.write_text(json.dumps(exclusion_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
