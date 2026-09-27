from __future__ import annotations

from collections import Counter
from typing import Any

from .review import was_human_override
from .schema import EvaluationResult, ValidationStatus


def build_summary(results: list[EvaluationResult]) -> dict[str, Any]:
    total = len(results)
    status_counts = Counter(item.validation_status.value for item in results)
    reason_counts = Counter(reason for item in results for reason in item.validation_reasons)
    label_counts = Counter()
    human_reviewed = 0
    human_override = 0

    for item in results:
        labels = item.final_label or item.model_label
        label_counts.update(labels.values())
        if item.human_review_label is not None:
            human_reviewed += 1
        if was_human_override(item):
            human_override += 1

    summary: dict[str, Any] = {
        "total_samples": total,
        "status_counts": {status.value: status_counts.get(status.value, 0) for status in ValidationStatus},
        "status_rates": {
            status.value: _rate(status_counts.get(status.value, 0), total)
            for status in ValidationStatus
        },
        "human_reviewed_count": human_reviewed,
        "human_reviewed_rate": _rate(human_reviewed, total),
        "human_override_count": human_override,
        "human_override_rate": _rate(human_override, human_reviewed),
        "label_distribution": dict(sorted(label_counts.items())),
        "validation_reason_distribution": dict(sorted(reason_counts.items())),
    }
    return summary


def build_bad_cases(results: list[EvaluationResult]) -> list[dict[str, Any]]:
    bad_cases = []
    for item in results:
        if item.validation_status != ValidationStatus.PASS or was_human_override(item):
            bad_cases.append(
                {
                    "sample_id": item.sample_id,
                    "model_output": item.model_output.to_dict() if item.model_output else None,
                    "validation_status": item.validation_status.value,
                    "validation_reasons": item.validation_reasons,
                    "human_review_label": item.human_review_label,
                    "human_review_note": item.human_review_note,
                    "final_label": item.final_label,
                    "error": item.error,
                }
            )
    return bad_cases


def render_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Evaluation Summary",
        "",
        f"- Total samples: {summary['total_samples']}",
        f"- Human reviewed: {summary['human_reviewed_count']} ({summary['human_reviewed_rate']:.2%})",
        f"- Human overrides: {summary['human_override_count']} ({summary['human_override_rate']:.2%})",
        "",
        "## Validation Status",
    ]
    for status, count in summary["status_counts"].items():
        rate = summary["status_rates"][status]
        lines.append(f"- {status}: {count} ({rate:.2%})")
    lines.extend(["", "## Label Distribution"])
    for label, count in summary["label_distribution"].items():
        lines.append(f"- {label}: {count}")
    lines.extend(["", "## Validation Reasons"])
    if summary["validation_reason_distribution"]:
        for reason, count in summary["validation_reason_distribution"].items():
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- None")
    return "\n".join(lines)


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0
