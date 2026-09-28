import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intonation_labeler.agent.policy import decide_next_action, validate_action
from intonation_labeler.agent.state import AgentState, AgentStatus


class AgentPolicyTests(unittest.TestCase):
    def test_new_goes_to_inspect(self):
        state = AgentState(task_id="t", input_location="in", output_dir="out")
        decision = decide_next_action(state)
        self.assertEqual(decision.action, "inspect_dataset")

    def test_valid_observation_can_run(self):
        state = AgentState(
            task_id="t",
            input_location="in",
            output_dir="out",
            status=AgentStatus.RUNNING,
            next_allowed_actions=["run_evaluation_batch"],
        )
        self.assertEqual(decide_next_action(state).action, "run_evaluation_batch")

    def test_partial_retryable_resumes(self):
        state = AgentState(
            task_id="t",
            input_location="in",
            output_dir="out",
            status=AgentStatus.PARTIAL,
            pending_count=1,
            last_error={"kind": "RETRYABLE", "message": "timeout"},
        )
        self.assertEqual(decide_next_action(state).action, "resume_evaluation_batch")

    def test_fatal_auth_blocks(self):
        state = AgentState(
            task_id="t",
            input_location="in",
            output_dir="out",
            status=AgentStatus.BLOCKED,
            last_error={"kind": "FATAL", "message": "HTTP 401"},
        )
        self.assertEqual(decide_next_action(state).action, "stop")

    def test_non_retryable_schema_error_is_not_repeatedly_retried(self):
        state = AgentState(
            task_id="t",
            input_location="in",
            output_dir="out",
            status=AgentStatus.PARTIAL,
            pending_count=1,
            last_error={"kind": "NON_RETRYABLE", "message": "schema"},
        )
        self.assertEqual(decide_next_action(state).action, "stop")

    def test_waiting_for_review_stops(self):
        state = AgentState(
            task_id="t",
            input_location="in",
            output_dir="out",
            status=AgentStatus.WAITING_FOR_REVIEW,
            pending_human_review=True,
            review_count=1,
        )
        self.assertEqual(decide_next_action(state).action, "stop")

    def test_human_review_completed_generates_report(self):
        state = AgentState(
            task_id="t",
            input_location="in",
            output_dir="out",
            status=AgentStatus.WAITING_FOR_REVIEW,
            pending_human_review=False,
            review_count=0,
        )
        self.assertEqual(decide_next_action(state).action, "generate_report")

    def test_completed_stops(self):
        state = AgentState(task_id="t", input_location="in", output_dir="out", status=AgentStatus.COMPLETED)
        self.assertEqual(decide_next_action(state).action, "stop")

    def test_forbidden_rule_modification_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_action("modify_validation_rule")

    def test_unsupported_action_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_action("unknown_action")


if __name__ == "__main__":
    unittest.main()
