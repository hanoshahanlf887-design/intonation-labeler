import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intonation_labeler.agent.state import AgentState, AgentStatus
from intonation_labeler.agent.tools import EvaluationAgentTools
from intonation_labeler.incremental import ErrorClassification, ErrorKind, persist_failure


class AgentToolsMockTests(unittest.TestCase):
    def make_dataset(self, base: Path, sample_id: str = "sample_001", review: bool = False) -> Path:
        wav_path = base / f"{sample_id}.wav"
        json_path = base / f"{sample_id}.json"
        wav_path.write_bytes(b"RIFF")
        feature = {} if review else {"t_maxf0": "0.78", "z_f0": "0.64", "rangeF0": "92", "slope": "18.40", "t_min": "0.12"}
        json_path.write_text(
            json.dumps(
                {
                    "transcript": "Could we meet after lunch?",
                    "anchors": {"tag_1": {"word": "lunch?", "start": 1.0, "end": 1.4, "feature": feature}},
                }
            ),
            encoding="utf-8",
        )
        manifest = base / "manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["sample_id", "wav_path", "feature_json_path"])
            writer.writeheader()
            writer.writerow({"sample_id": sample_id, "wav_path": str(wav_path), "feature_json_path": str(json_path)})
        return manifest

    def test_inspect_valid_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            manifest = self.make_dataset(base)
            state = AgentState("t", str(base), str(base / "out"), manifest_path=str(manifest))
            result = EvaluationAgentTools().inspect_dataset(state)
            self.assertTrue(result.observation["valid"])
            self.assertEqual(result.state.next_allowed_actions, ["run_evaluation_batch"])

    def test_real_dry_run_workflow_reaches_report_for_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            manifest = self.make_dataset(base)
            state = AgentState("t", str(base), str(base / "out"), manifest_path=str(manifest))
            tools = EvaluationAgentTools()
            inspected = tools.inspect_dataset(state).state
            run = tools.run_evaluation_batch(inspected).state
            report = tools.generate_report(run)
            self.assertTrue(report.state.report_ready)
            self.assertTrue((base / "out" / "evaluation_results.json").exists())

    def test_review_case_goes_to_review_queue_without_confirmed_error(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            manifest = self.make_dataset(base, review=True)
            state = AgentState("t", str(base), str(base / "out"), manifest_path=str(manifest))
            tools = EvaluationAgentTools()
            run = tools.run_evaluation_batch(tools.inspect_dataset(state).state).state
            queue = tools.get_review_queue(run)
            self.assertEqual(queue.state.status, AgentStatus.WAITING_FOR_REVIEW)
            self.assertEqual(queue.observation["review_queue_count"], 1)
            self.assertNotIn("confirmed_model_error", json.dumps(queue.observation))

    def test_retryable_failure_status_can_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            manifest = self.make_dataset(base)
            output = base / "out"
            output.mkdir()
            persist_failure(output, "sample_001", ErrorClassification(ErrorKind.RETRYABLE, status_code=429, message="timeout"), attempts=1)
            state = AgentState(
                "t",
                str(base),
                str(output),
                status=AgentStatus.PARTIAL,
                total_samples=1,
                pending_count=1,
                manifest_path=str(manifest),
                last_error={"kind": "RETRYABLE", "message": "timeout"},
            )
            resumed = EvaluationAgentTools().resume_evaluation_batch(state)
            self.assertEqual(resumed.state.completed_count, 1)
            self.assertEqual(resumed.state.failed_count, 0)

    def test_report_generation_does_not_call_model(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            manifest = self.make_dataset(base)
            tools = EvaluationAgentTools()
            state = AgentState("t", str(base), str(base / "out"), manifest_path=str(manifest))
            run = tools.run_evaluation_batch(tools.inspect_dataset(state).state).state
            report = tools.generate_report(run)
            self.assertTrue(report.observation["report_ready"])
            self.assertEqual(report.action, "generate_report")


if __name__ == "__main__":
    unittest.main()
