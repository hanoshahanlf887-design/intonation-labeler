from __future__ import annotations

from .schema import ALLOWED_LABELS, EvaluationResult


def apply_human_review(
    result: EvaluationResult,
    human_review_label: dict[str, str] | None = None,
    human_review_note: str = "",
    accept_model: bool = False,
) -> EvaluationResult:
    if accept_model:
        final_label = dict(result.model_label)
        human_label = dict(result.model_label)
    else:
        human_label = human_review_label or {}
        _validate_human_labels(human_label)
        final_label = dict(human_label)

    result.human_review_label = human_label
    result.human_review_note = human_review_note
    result.final_label = final_label
    result.needs_human_review = False
    return result


def assign_default_final_label(result: EvaluationResult) -> EvaluationResult:
    if not result.needs_human_review and result.final_label is None:
        result.final_label = dict(result.model_label)
    return result


def was_human_override(result: EvaluationResult) -> bool:
    if not result.human_review_label:
        return False
    return result.human_review_label != result.model_label


def _validate_human_labels(labels: dict[str, str]) -> None:
    for anchor_id, label in labels.items():
        if not anchor_id:
            raise ValueError("human_review_label contains an empty anchor id.")
        if label not in ALLOWED_LABELS:
            raise ValueError(f"Invalid human label for {anchor_id}: {label}")
