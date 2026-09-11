import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intonation_labeler.demo import DEMO_NOTICE, build_demo_result
from intonation_labeler.io import FeatureData


class DemoTests(unittest.TestCase):
    def test_dry_run_does_not_need_api_key(self):
        old_value = os.environ.pop("GOOGLE_API_KEY", None)
        try:
            result = build_demo_result(FeatureData("Could we meet after lunch?", {"tag_1": {"word": "lunch?"}}))
            self.assertIn(DEMO_NOTICE, result)
        finally:
            if old_value is not None:
                os.environ["GOOGLE_API_KEY"] = old_value

    def test_mock_result_is_marked_as_demo(self):
        result = build_demo_result(FeatureData("Could we meet after lunch?", {"tag_1": {"word": "lunch?"}}))
        self.assertIn("Demo result", result)
        self.assertIn("no Gemini API call", result)

    def test_demo_does_not_call_gemini_client(self):
        client = Mock()
        build_demo_result(FeatureData("Could we meet after lunch?", {"tag_1": {"word": "lunch?"}}))
        client.assert_not_called()


if __name__ == "__main__":
    unittest.main()

