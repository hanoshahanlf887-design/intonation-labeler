import json
import sys
import unittest
import zipfile
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intonation_labeler.export import AnnotationResult, build_results_zip, render_merged_text, render_results_json


class ExportTests(unittest.TestCase):
    def test_renders_merged_text(self):
        text = render_merged_text([AnnotationResult(sample_id="synthetic_001", result="demo")])
        self.assertIn("ID: synthetic_001", text)
        self.assertIn("Result: demo", text)

    def test_renders_json(self):
        data = json.loads(render_results_json([AnnotationResult(sample_id="synthetic_001", result="demo")]))
        self.assertEqual(data[0]["id"], "synthetic_001")
        self.assertEqual(data[0]["result"], "demo")

    def test_zip_contains_expected_files(self):
        zip_bytes = build_results_zip([AnnotationResult(sample_id="synthetic_001", result="demo")])
        with zipfile.ZipFile(BytesIO(zip_bytes)) as archive:
            names = set(archive.namelist())
        self.assertIn("m_results.txt", names)
        self.assertIn("m_results.json", names)
        self.assertIn("individual/synthetic_001.txt", names)


if __name__ == "__main__":
    unittest.main()

