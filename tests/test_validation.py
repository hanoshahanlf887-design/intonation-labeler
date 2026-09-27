import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intonation_labeler.io import FeatureData
from intonation_labeler.schema import ValidationReason, ValidationStatus
from intonation_labeler.validation import ValidationConfig, evaluate_model_output, parse_model_output


def feature_data() -> FeatureData:
    return FeatureData(
        transcript="Could we meet after lunch?",
        anchors={
            "tag_1": {
                "word": "lunch?",
                "feature": {
                    "t_maxf0": "0.78",
                    "z_f0": "0.64",
                    "rangeF0": "92",
                    "slope": "18.40",
                    "t_min": "0.12",
                },
            }
        },
    )


def response(**overrides):
    data = {
        "transcript": "Could we meet after lunch?",
        "annotated_text": "Could we meet after lunch?【上升】",
        "annotations": [
            {
                "anchor_id": "tag_1",
                "label": "RISING",
                "confidence": 0.82,
                "reason": "Audible final rise with supporting acoustic cues.",
            }
        ],
    }
    data.update(overrides)
    return parse_model_output(json.dumps(data))


def transcript_feature_data(transcript: str) -> FeatureData:
    data = feature_data()
    return FeatureData(transcript=transcript, anchors=data.anchors)


def transcript_response(transcript: str):
    return response(transcript=transcript)


class ValidationTests(unittest.TestCase):
    def test_pass_case_assigns_final_label(self):
        result = evaluate_model_output("synthetic_001", feature_data(), response())
        self.assertEqual(result.validation_status, ValidationStatus.PASS)
        self.assertFalse(result.needs_human_review)
        self.assertEqual(result.final_label, {"tag_1": "RISING"})

    def test_low_confidence_routes_to_review(self):
        model_output = response(
            annotations=[{"anchor_id": "tag_1", "label": "RISING", "confidence": 0.4, "reason": "uncertain"}]
        )
        result = evaluate_model_output("synthetic_001", feature_data(), model_output, ValidationConfig(confidence_threshold=0.6))
        self.assertEqual(result.validation_status, ValidationStatus.REVIEW)
        self.assertIn(ValidationReason.LOW_MODEL_CONFIDENCE.value, result.validation_reasons)
        self.assertTrue(result.needs_human_review)

    def test_invalid_label_routes_to_review(self):
        model_output = response(
            annotations=[{"anchor_id": "tag_1", "label": "UP", "confidence": 0.9, "reason": "bad enum"}]
        )
        result = evaluate_model_output("synthetic_001", feature_data(), model_output)
        self.assertIn(ValidationReason.INVALID_LABEL.value, result.validation_reasons)

    def test_missing_anchor_routes_to_review(self):
        model_output = response(annotations=[])
        result = evaluate_model_output("synthetic_001", feature_data(), model_output)
        self.assertIn(ValidationReason.MISSING_ANCHOR_RESULT.value, result.validation_reasons)

    def test_transcript_changed_routes_to_review(self):
        model_output = response(transcript="Could we meet after dinner?")
        result = evaluate_model_output("synthetic_001", feature_data(), model_output)
        self.assertIn(ValidationReason.TRANSCRIPT_CHANGED.value, result.validation_reasons)

    def test_identical_transcript_does_not_route_to_review(self):
        data = transcript_feature_data("Where are you going?")
        result = evaluate_model_output("synthetic_001", data, transcript_response("Where are you going?"))
        self.assertNotIn(ValidationReason.TRANSCRIPT_CHANGED.value, result.validation_reasons)

    def test_double_quoted_outer_transcript_does_not_route_to_review(self):
        data = transcript_feature_data("Where are you going?")
        result = evaluate_model_output("synthetic_001", data, transcript_response('"Where are you going?"'))
        self.assertNotIn(ValidationReason.TRANSCRIPT_CHANGED.value, result.validation_reasons)

    def test_single_quoted_outer_transcript_does_not_route_to_review(self):
        data = transcript_feature_data("Where are you going?")
        result = evaluate_model_output("synthetic_001", data, transcript_response("'Where are you going?'"))
        self.assertNotIn(ValidationReason.TRANSCRIPT_CHANGED.value, result.validation_reasons)

    def test_outer_quotes_with_whitespace_do_not_route_to_review(self):
        data = transcript_feature_data("Where are you going?")
        result = evaluate_model_output("synthetic_001", data, transcript_response('  "Where are you going?"  '))
        self.assertNotIn(ValidationReason.TRANSCRIPT_CHANGED.value, result.validation_reasons)

    def test_added_word_still_routes_to_review(self):
        data = transcript_feature_data("Where are you going?")
        result = evaluate_model_output("synthetic_001", data, transcript_response("Where are you going tomorrow?"))
        self.assertIn(ValidationReason.TRANSCRIPT_CHANGED.value, result.validation_reasons)

    def test_deleted_word_still_routes_to_review(self):
        data = transcript_feature_data("Where are you going?")
        result = evaluate_model_output("synthetic_001", data, transcript_response("Where are you?"))
        self.assertIn(ValidationReason.TRANSCRIPT_CHANGED.value, result.validation_reasons)

    def test_contraction_expansion_still_routes_to_review(self):
        data = transcript_feature_data("I don't know.")
        result = evaluate_model_output("synthetic_001", data, transcript_response("I do not know."))
        self.assertIn(ValidationReason.TRANSCRIPT_CHANGED.value, result.validation_reasons)

    def test_acoustic_feature_missing_routes_to_review(self):
        data = FeatureData("Could we meet after lunch?", {"tag_1": {"word": "lunch?"}})
        result = evaluate_model_output("synthetic_001", data, response())
        self.assertIn(ValidationReason.MISSING_ACOUSTIC_FEATURE.value, result.validation_reasons)

    def test_parse_rejects_missing_fields(self):
        with self.assertRaises(ValueError):
            parse_model_output('{"transcript": "x"}')

    def test_parse_accepts_json_code_fence(self):
        raw = "```json\n" + json.dumps(
            {
                "transcript": "Could we meet after lunch?",
                "annotations": [{"anchor_id": "tag_1", "label": "RISING", "confidence": 0.8, "reason": "test"}],
            }
        ) + "\n```"
        parsed = parse_model_output(raw)
        self.assertEqual(parsed.model_label, {"tag_1": "RISING"})

    def test_parse_accepts_plain_code_fence_with_outer_whitespace(self):
        raw = "\n  ```\n" + json.dumps(
            {
                "transcript": "Could we meet after lunch?",
                "annotations": [{"anchor_id": "tag_1", "label": "FALLING", "confidence": 0.8, "reason": "test"}],
            }
        ) + "\n```\n"
        parsed = parse_model_output(raw)
        self.assertEqual(parsed.model_label, {"tag_1": "FALLING"})

    def test_api_call_failed_reason_exists_for_live_environment_failures(self):
        self.assertEqual(ValidationReason.API_CALL_FAILED.value, "API_CALL_FAILED")


if __name__ == "__main__":
    unittest.main()
