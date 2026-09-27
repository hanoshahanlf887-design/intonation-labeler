import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intonation_labeler.io import FeatureData
from intonation_labeler.report import build_bad_cases, build_summary
from intonation_labeler.review import apply_human_review, was_human_override
from intonation_labeler.validation import ValidationConfig, evaluate_model_output, parse_model_output


def make_result(confidence=0.82):
    feature_data = FeatureData(
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
    model_output = parse_model_output(
        json.dumps(
            {
                "transcript": feature_data.transcript,
                "annotated_text": "Could we meet after lunch?【上升】",
                "annotations": [
                    {
                        "anchor_id": "tag_1",
                        "label": "RISING",
                        "confidence": confidence,
                        "reason": "Synthetic reason.",
                    }
                ],
            }
        )
    )
    return evaluate_model_output("synthetic_001", feature_data, model_output, ValidationConfig(confidence_threshold=0.6))


class ReviewReportTests(unittest.TestCase):
    def test_accept_model_sets_final_label(self):
        result = make_result(confidence=0.4)
        apply_human_review(result, accept_model=True, human_review_note="acceptable after listening")
        self.assertEqual(result.final_label, {"tag_1": "RISING"})
        self.assertFalse(result.needs_human_review)

    def test_human_override_is_preserved(self):
        result = make_result(confidence=0.4)
        apply_human_review(result, {"tag_1": "FALLING"}, "clear fall on review")
        self.assertEqual(result.model_label, {"tag_1": "RISING"})
        self.assertEqual(result.final_label, {"tag_1": "FALLING"})
        self.assertTrue(was_human_override(result))

    def test_summary_counts_status_and_overrides(self):
        pass_result = make_result()
        review_result = make_result(confidence=0.4)
        apply_human_review(review_result, {"tag_1": "FALLING"}, "override")
        summary = build_summary([pass_result, review_result])
        self.assertEqual(summary["total_samples"], 2)
        self.assertEqual(summary["status_counts"]["PASS"], 1)
        self.assertEqual(summary["status_counts"]["REVIEW"], 1)
        self.assertEqual(summary["human_reviewed_count"], 1)
        self.assertEqual(summary["human_override_count"], 1)

    def test_bad_cases_include_non_pass_and_overrides(self):
        pass_result = make_result()
        review_result = make_result(confidence=0.4)
        apply_human_review(review_result, {"tag_1": "FALLING"}, "override")
        bad_cases = build_bad_cases([pass_result, review_result])
        self.assertEqual(len(bad_cases), 1)
        self.assertEqual(bad_cases[0]["sample_id"], "synthetic_001")
        self.assertEqual(bad_cases[0]["final_label"], {"tag_1": "FALLING"})


if __name__ == "__main__":
    unittest.main()
