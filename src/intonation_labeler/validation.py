from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .io import FeatureData
from .schema import (
    ALLOWED_LABELS,
    AnchorPrediction,
    EvaluationResult,
    ModelOutput,
    ValidationReason,
    ValidationStatus,
)


REQUIRED_FEATURE_KEYS = {"t_maxf0", "z_f0", "rangeF0", "slope", "t_min"}


@dataclass(frozen=True)
class ValidationConfig:
    confidence_threshold: float = 0.6
    enable_acoustic_conflict_rules: bool = False


class ModelOutputParseError(ValueError):
    """Raised when a model response cannot be parsed as the expected JSON object."""


def parse_model_output(raw_response: str) -> ModelOutput:
    raw_response = (raw_response or "").strip()
    if raw_response.startswith("```"):
        raw_response = _strip_code_fence(raw_response)
    try:
        data = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ModelOutputParseError("Model response is not valid JSON.") from exc

    if not isinstance(data, dict):
        raise ModelOutputParseError("Model response must be a JSON object.")

    transcript = data.get("transcript")
    annotations = data.get("annotations")
    if not isinstance(transcript, str):
        raise ModelOutputParseError("Model response is missing transcript.")
    if not isinstance(annotations, list):
        raise ModelOutputParseError("Model response is missing annotations list.")

    parsed_annotations: list[AnchorPrediction] = []
    for item in annotations:
        if not isinstance(item, dict):
            raise ModelOutputParseError("Each annotation must be an object.")
        anchor_id = item.get("anchor_id")
        label = item.get("label")
        if not isinstance(anchor_id, str) or not anchor_id:
            raise ModelOutputParseError("Each annotation must include anchor_id.")
        if not isinstance(label, str) or not label:
            raise ModelOutputParseError("Each annotation must include label.")
        confidence = item.get("confidence")
        if confidence is not None:
            try:
                confidence = float(confidence)
            except (TypeError, ValueError) as exc:
                raise ModelOutputParseError("confidence must be numeric when provided.") from exc
        reason = item.get("reason", "")
        parsed_annotations.append(
            AnchorPrediction(
                anchor_id=anchor_id,
                label=label,
                confidence=confidence,
                reason=reason if isinstance(reason, str) else str(reason),
            )
        )

    annotated_text = data.get("annotated_text", "")
    return ModelOutput(
        transcript=transcript,
        annotations=parsed_annotations,
        annotated_text=annotated_text if isinstance(annotated_text, str) else "",
        raw_response=raw_response,
    )


def evaluate_model_output(
    sample_id: str,
    feature_data: FeatureData,
    model_output: ModelOutput,
    config: ValidationConfig | None = None,
) -> EvaluationResult:
    config = config or ValidationConfig()
    reasons: list[str] = []
    anchor_ids = set(feature_data.anchors.keys())
    prediction_ids = {item.anchor_id for item in model_output.annotations}

    if _normalize_transcript_for_comparison(model_output.transcript) != _normalize_transcript_for_comparison(feature_data.transcript):
        reasons.append(ValidationReason.TRANSCRIPT_CHANGED.value)

    missing_anchor_ids = sorted(anchor_ids - prediction_ids)
    if missing_anchor_ids:
        reasons.append(ValidationReason.MISSING_ANCHOR_RESULT.value)

    unknown_anchor_ids = sorted(prediction_ids - anchor_ids)
    if unknown_anchor_ids:
        reasons.append(ValidationReason.UNKNOWN_ANCHOR_RESULT.value)

    for prediction in model_output.annotations:
        if prediction.label not in ALLOWED_LABELS:
            reasons.append(ValidationReason.INVALID_LABEL.value)
        if prediction.confidence is None:
            reasons.append(ValidationReason.MISSING_REQUIRED_FIELD.value)
        elif prediction.confidence < config.confidence_threshold:
            reasons.append(ValidationReason.LOW_MODEL_CONFIDENCE.value)

    acoustic_features = extract_acoustic_features(feature_data)
    for anchor_id in sorted(anchor_ids):
        feature = acoustic_features.get(anchor_id)
        if not isinstance(feature, dict) or not feature:
            reasons.append(ValidationReason.MISSING_ACOUSTIC_FEATURE.value)
            continue
        missing_feature_keys = REQUIRED_FEATURE_KEYS - set(feature)
        if missing_feature_keys:
            reasons.append(ValidationReason.MISSING_ACOUSTIC_FEATURE.value)
        if _has_abnormal_feature_value(feature):
            reasons.append(ValidationReason.ABNORMAL_ACOUSTIC_FEATURE.value)

    if config.enable_acoustic_conflict_rules:
        conflict_reasons = detect_acoustic_conflicts(model_output, acoustic_features)
        reasons.extend(conflict_reasons)

    reasons = sorted(set(reasons))
    status = _status_from_reasons(reasons)
    needs_human_review = status != ValidationStatus.PASS
    final_label = None if needs_human_review else model_output.model_label
    return EvaluationResult(
        sample_id=sample_id,
        transcript=feature_data.transcript,
        acoustic_features=acoustic_features,
        model_output=model_output,
        validation_status=status,
        validation_reasons=reasons,
        needs_human_review=needs_human_review,
        final_label=final_label,
    )


def build_invalid_result(sample_id: str, error: str, transcript: str = "") -> EvaluationResult:
    return EvaluationResult(
        sample_id=sample_id,
        transcript=transcript,
        validation_status=ValidationStatus.INVALID_INPUT,
        validation_reasons=[ValidationReason.INVALID_MODEL_OUTPUT.value],
        needs_human_review=True,
        error=error,
    )


def extract_acoustic_features(feature_data: FeatureData) -> dict[str, Any]:
    features: dict[str, Any] = {}
    for anchor_id, anchor in feature_data.anchors.items():
        if isinstance(anchor, dict):
            features[anchor_id] = anchor.get("feature", {})
        else:
            features[anchor_id] = {}
    return features


def detect_acoustic_conflicts(
    model_output: ModelOutput,
    acoustic_features: dict[str, Any],
) -> list[str]:
    # Reserved for future project-specific rules. The current public schema does
    # not define enough reliable thresholds to infer intonation from F0 features.
    return []


def _status_from_reasons(reasons: list[str]) -> ValidationStatus:
    if not reasons:
        return ValidationStatus.PASS
    if ValidationReason.ACOUSTIC_CONFLICT.value in reasons:
        return ValidationStatus.CONFLICT
    if ValidationReason.MISSING_ACOUSTIC_FEATURE.value in reasons and len(reasons) == 1:
        return ValidationStatus.REVIEW
    return ValidationStatus.REVIEW


def _has_abnormal_feature_value(feature: dict[str, Any]) -> bool:
    for value in feature.values():
        if value in ("", None):
            return True
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return True
        if numeric != numeric or numeric in (float("inf"), float("-inf")):
            return True
    return False


def _normalize_transcript_for_comparison(transcript: str) -> str:
    text = transcript.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        return text[1:-1].strip()
    return text


def _strip_code_fence(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()
