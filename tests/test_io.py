import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intonation_labeler.io import FeatureFormatError, load_feature_json, pair_directory


class IoTests(unittest.TestCase):
    def test_loads_canonical_json(self):
        with tempfile.NamedTemporaryFile("w+", suffix=".json", encoding="utf-8", delete=False) as file:
            json.dump({"transcript": "May I join?", "anchors": {"tag_1": {"word": "join?"}}}, file)
            path = Path(file.name)
        try:
            data = load_feature_json(path, "sample")
            self.assertEqual(data.transcript, "May I join?")
            self.assertIn("tag_1", data.anchors)
        finally:
            path.unlink(missing_ok=True)

    def test_missing_transcript_raises(self):
        with tempfile.NamedTemporaryFile("w+", suffix=".json", encoding="utf-8", delete=False) as file:
            json.dump({"anchors": {"tag_1": {"word": "join?"}}}, file)
            path = Path(file.name)
        try:
            with self.assertRaises(FeatureFormatError):
                load_feature_json(path, "sample")
        finally:
            path.unlink(missing_ok=True)

    def test_missing_anchors_raises(self):
        with tempfile.NamedTemporaryFile("w+", suffix=".json", encoding="utf-8", delete=False) as file:
            json.dump({"transcript": "May I join?"}, file)
            path = Path(file.name)
        try:
            with self.assertRaises(FeatureFormatError):
                load_feature_json(path, "sample")
        finally:
            path.unlink(missing_ok=True)

    def test_pairs_same_stem_files(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "sample_001.wav").write_bytes(b"RIFF")
            (base / "sample_001.json").write_text("{}", encoding="utf-8")
            pairs, missing_wav, missing_json = pair_directory(base)
            self.assertEqual([pair.sample_id for pair in pairs], ["sample_001"])
            self.assertEqual(missing_wav, [])
            self.assertEqual(missing_json, [])

    def test_reports_missing_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "audio_only.wav").write_bytes(b"RIFF")
            (base / "json_only.json").write_text("{}", encoding="utf-8")
            pairs, missing_wav, missing_json = pair_directory(base)
            self.assertEqual(pairs, [])
            self.assertEqual(missing_wav, ["json_only"])
            self.assertEqual(missing_json, ["audio_only"])


if __name__ == "__main__":
    unittest.main()

