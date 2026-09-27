import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intonation_labeler.incremental import (  # noqa: E402
    ErrorKind,
    PilotTask,
    classify_inference_error,
    load_manifest_tasks,
    load_success_result,
    run_incremental_batch,
)


def write_feature_json(path: Path, transcript: str = "Could we meet after lunch?") -> None:
    path.write_text(
        json.dumps(
            {
                "transcript": transcript,
                "anchors": {
                    "tag_1": {
                        "word": "lunch?",
                        "start": 1.0,
                        "end": 1.4,
                        "feature": {
                            "t_maxf0": "0.78",
                            "z_f0": "0.64",
                            "rangeF0": "92",
                            "slope": "18.40",
                            "t_min": "0.12",
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def model_response(label: str = "RISING", confidence: float = 0.82) -> str:
    return json.dumps(
        {
            "transcript": "Could we meet after lunch?",
            "annotations": [
                {
                    "anchor_id": "tag_1",
                    "label": label,
                    "confidence": confidence,
                    "reason": "Synthetic model response.",
                }
            ],
        }
    )


class ReadTimeout(Exception):
    pass


class IncrementalBatchTests(unittest.TestCase):
    def make_task(self, directory: Path, sample_id: str = "sample_001") -> PilotTask:
        wav_path = directory / f"{sample_id}.wav"
        json_path = directory / f"{sample_id}_feature.json"
        wav_path.write_bytes(b"RIFF")
        write_feature_json(json_path)
        return PilotTask(sample_id, wav_path, json_path)

    def test_successful_sample_is_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            task = self.make_task(base)
            output = base / "out"

            summary = run_incremental_batch(
                [task],
                output,
                "gemini-3.8-flash",
                lambda model, wav, prompt: model_response(),
                retry_base_delay=0,
                sleep_fn=lambda seconds: None,
            )

            self.assertEqual(summary.succeeded, 1)
            self.assertIsNotNone(load_success_result(output, task.sample_id))
            self.assertTrue((output / "individual" / f"{task.sample_id}.json").exists())

    def test_resume_skips_successful_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            task = self.make_task(base)
            output = base / "out"
            calls = []

            run_incremental_batch(
                [task],
                output,
                "gemini-3.8-flash",
                lambda model, wav, prompt: model_response(),
                retry_base_delay=0,
                sleep_fn=lambda seconds: None,
            )

            summary = run_incremental_batch(
                [task],
                output,
                "gemini-3.8-flash",
                lambda model, wav, prompt: calls.append(task.sample_id) or model_response("FALLING"),
                retry_base_delay=0,
                sleep_fn=lambda seconds: None,
            )

            self.assertEqual(summary.skipped_success, 1)
            self.assertEqual(calls, [])

    def test_failed_sample_is_retried_and_then_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            task = self.make_task(base)
            output = base / "out"
            calls = {"count": 0}

            def annotator(model, wav, prompt):
                calls["count"] += 1
                if calls["count"] == 1:
                    raise RuntimeError("Relay HTTP 429: exhausted")
                return model_response()

            summary = run_incremental_batch(
                [task],
                output,
                "gemini-3.8-flash",
                annotator,
                retry_base_delay=0,
                sleep_fn=lambda seconds: None,
            )

            self.assertEqual(calls["count"], 2)
            self.assertEqual(summary.succeeded, 1)
            self.assertFalse((output / "failures" / f"{task.sample_id}.json").exists())

    def test_429_and_timeout_are_retryable(self):
        self.assertEqual(classify_inference_error(RuntimeError("Relay HTTP 429: exhausted")).kind, ErrorKind.RETRYABLE)
        self.assertEqual(classify_inference_error(ReadTimeout("read timed out")).kind, ErrorKind.RETRYABLE)

    def test_400_is_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            task = self.make_task(base)
            calls = {"count": 0}

            def annotator(model, wav, prompt):
                calls["count"] += 1
                raise RuntimeError("Relay HTTP 400: bad request")

            summary = run_incremental_batch(
                [task],
                base / "out",
                "gemini-3.8-flash",
                annotator,
                retry_base_delay=0,
                sleep_fn=lambda seconds: None,
            )

            self.assertEqual(calls["count"], 1)
            self.assertEqual(summary.failed, 1)

    def test_401_and_403_stop_batch(self):
        for status in (401, 403):
            with self.subTest(status=status):
                with tempfile.TemporaryDirectory() as directory:
                    base = Path(directory)
                    task1 = self.make_task(base, "sample_001")
                    task2 = self.make_task(base, "sample_002")
                    calls = {"count": 0}

                    def annotator(model, wav, prompt):
                        calls["count"] += 1
                        raise RuntimeError(f"Relay HTTP {status}: forbidden")

                    summary = run_incremental_batch(
                        [task1, task2],
                        base / "out",
                        "gemini-3.8-flash",
                        annotator,
                        retry_base_delay=0,
                        sleep_fn=lambda seconds: None,
                    )

                    self.assertTrue(summary.paused)
                    self.assertEqual(calls["count"], 1)

    def test_existing_success_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            task = self.make_task(base)
            output = base / "out"

            run_incremental_batch(
                [task],
                output,
                "gemini-3.8-flash",
                lambda model, wav, prompt: model_response("RISING"),
                retry_base_delay=0,
                sleep_fn=lambda seconds: None,
            )
            path = output / "individual" / f"{task.sample_id}.json"
            original = path.read_text(encoding="utf-8")

            run_incremental_batch(
                [task],
                output,
                "gemini-3.8-flash",
                lambda model, wav, prompt: model_response("FALLING"),
                retry_base_delay=0,
                sleep_fn=lambda seconds: None,
            )

            self.assertEqual(path.read_text(encoding="utf-8"), original)

    def test_manifest_loader_does_not_open_human_label_file(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            wav_path = base / "sample_001.wav"
            json_path = base / "sample_001_feature.json"
            manifest_path = base / "inference_manifest.csv"
            human_path = base / "human_labels.csv"
            human_path.write_text("human_label\nFALLING\n", encoding="utf-8")
            with manifest_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=["sample_id", "wav_path", "feature_json_path", "transcript", "anchor_id", "anchor_word", "anchor_start", "anchor_end"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "sample_id": "sample_001",
                        "wav_path": str(wav_path),
                        "feature_json_path": str(json_path),
                        "transcript": "Could we meet after lunch?",
                        "anchor_id": "tag_1",
                        "anchor_word": "lunch?",
                        "anchor_start": "1.0",
                        "anchor_end": "1.4",
                    }
                )

            original_open = io.open

            def guarded_open(path, *args, **kwargs):
                if str(path).endswith("human_labels.csv"):
                    raise AssertionError("inference path must not read human labels")
                return original_open(path, *args, **kwargs)

            with mock.patch("io.open", side_effect=guarded_open):
                tasks = load_manifest_tasks(manifest_path)
                write_feature_json(json_path)
                wav_path.write_bytes(b"RIFF")
                summary = run_incremental_batch(
                    tasks,
                    base / "out",
                    "gemini-3.8-flash",
                    lambda model, wav, prompt: model_response(),
                    retry_base_delay=0,
                    sleep_fn=lambda seconds: None,
                )

            self.assertEqual([task.sample_id for task in tasks], ["sample_001"])
            self.assertEqual(summary.succeeded, 1)


if __name__ == "__main__":
    unittest.main()
