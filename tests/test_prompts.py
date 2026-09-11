import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intonation_labeler.prompts import build_prompt


class PromptTests(unittest.TestCase):
    def test_prompt_contains_transcript_and_anchor(self):
        prompt = build_prompt("Could we meet after lunch?", {"tag_1": {"word": "lunch?"}})
        self.assertIn("Could we meet after lunch?", prompt)
        self.assertIn("tag_1", prompt)
        self.assertIn("lunch?", prompt)

    def test_prompt_does_not_contain_api_key(self):
        prompt = build_prompt("Could we meet after lunch?", {"tag_1": {"word": "lunch?"}})
        self.assertNotIn("GOOGLE_API_KEY", prompt)
        self.assertNotIn("AIza", prompt)

    def test_prompt_does_not_depend_on_historical_sample(self):
        prompt = build_prompt("Could we meet after lunch?", {"tag_1": {"word": "lunch?"}})
        self.assertNotIn("legacy_sample_id", prompt)


if __name__ == "__main__":
    unittest.main()
