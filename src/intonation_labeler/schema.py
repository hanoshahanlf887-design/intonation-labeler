from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


ALLOWED_LABELS = {"RISING", "FALLING", "LEVEL"}


class ValidationStatus(str, Enum):
    PASS = "PASS"
    REVIEW = "REVIEW"
    CONFLICT = "CONFLICT"
    INVALID_INPUT = "INVALID_INPUT"


class ValidationReason(str, Enum):
    LOW_MODEL_CONFIDENCE = "LOW_MODEL_CONFIDENCE"
    MISSING_ACOUSTIC_FEATURE = "MISSING_ACOUSTIC_FEATURE"
    ABNORMAL_ACOUSTIC_FEATURE = "ABNORMAL_ACOUSTIC_FEATURE"
    INVALID_MODEL_OUTPUT = "INVALID_MODEL_OUTPUT"
    API_CALL_FAILED = "API_CALL_FAILED"
    MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"
    MISSING_ANCHOR_RESULT = "MISSING_ANCHOR_RESULT"
    UNKNOWN_ANCHOR_RESULT = "UNKNOWN_ANCHOR_RESULT"
    INVALID_LABEL = "INVALID_LABEL"
    TRANSCRIPT_CHANGED = "TRANSCRIPT_CHANGED"
    ACOUSTIC_CONFLICT = "ACOUSTIC_CONFLICT"


@dataclass(frozen=True)
class AnchorPrediction:
    anchor_id: str
    label: str
    confidence: float | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_id": self.anchor_id,
            "label": self.label,
            "confidence": self.confidence,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ModelOutput:
    transcript: str
    annotations: list[AnchorPrediction]
    annotated_text: str = ""
    raw_response: str = ""

    @property
    def model_label(self) -> dict[str, str]:
        return {item.anchor_id: item.label for item in self.annotations}

    @property
    def model_confidence(self) -> dict[str, float | None]:
        return {item.anchor_id: item.confidence for item in self.annotations}

    @property
    def model_reason(self) -> dict[str, str]:
        return {item.anchor_id: item.reason for item in self.annotations}

    def to_dict(self) -> dict[str, Any]:
        return {
            "transcript": self.transcript,
            "annotated_text": self.annotated_text,
            "annotations": [item.to_dict() for item in self.annotations],
            "model_label": self.model_label,
            "model_confidence": self.model_confidence,
            "model_reason": self.model_reason,
            "raw_response": self.raw_response,
        }


@dataclass
class EvaluationResult:
    sample_id: str
    transcript: str = ""
    acoustic_features: dict[str, Any] = field(default_factory=dict)
    model_output: ModelOutput | None = None
    validation_status: ValidationStatus = ValidationStatus.REVIEW
    validation_reasons: list[str] = field(default_factory=list)
    needs_human_review: bool = True
    human_review_label: dict[str, str] | None = None
    human_review_note: str = ""
    final_label: dict[str, str] | None = None
    error: str | None = None

    @property
    def model_label(self) -> dict[str, str]:
        return self.model_output.model_label if self.model_output else {}

    @property
    def model_confidence(self) -> dict[str, float | None]:
        return self.model_output.model_confidence if self.model_output else {}

    @property
    def model_reason(self) -> dict[str, str]:
        return self.model_output.model_reason if self.model_output else {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "transcript": self.transcript,
            "acoustic_features": self.acoustic_features,
            "model_output": self.model_output.to_dict() if self.model_output else None,
            "model_label": self.model_label,
            "model_confidence": self.model_confidence,
            "model_reason": self.model_reason,
            "validation_status": self.validation_status.value,
            "validation_reasons": self.validation_reasons,
            "needs_human_review": self.needs_human_review,
            "human_review_label": self.human_review_label,
            "human_review_note": self.human_review_note,
            "final_label": self.final_label,
            "error": self.error,
        }
