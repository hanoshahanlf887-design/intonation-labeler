import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intonation_labeler.agent.runner import EvaluationAgent
from intonation_labeler.agent.state import AgentState, AgentStatus
from intonation_labeler.agent.tools import ToolResult


class FakeTools:
    def __init__(self, review: bool = False, fatal: bool = False, retryable_once: bool = False):
        self.review = review
        self.fatal = fatal
        self.retryable_once = retryable_once
        self.calls = []

    def inspect_dataset(self, state):
        self.calls.append("inspect_dataset")
        return ToolResult(
            "inspect_dataset",
            {"valid": True, "total_samples": 1},
            state.copy(status=AgentStatus.RUNNING.value, total_samples=1, pending_count=1, next_allowed_actions=["run_evaluation_batch"]),
            "valid=True",
        )

    def run_evaluation_batch(self, state):
        self.calls.append("run_evaluation_batch")
        if self.fatal:
            return ToolResult(
                "run_evaluation_batch",
                {"last_error_kind": "FATAL"},
                state.copy(status=AgentStatus.BLOCKED.value, last_error={"kind": "FATAL", "message": "HTTP 401"}, next_allowed_actions=[]),
                "fatal auth",
            )
        if self.retryable_once:
            return ToolResult(
                "run_evaluation_batch",
                {"completed": 0, "pending": 1, "last_error_kind": "RETRYABLE"},
                state.copy(status=AgentStatus.PARTIAL.value, completed_count=0, pending_count=1, last_error={"kind": "RETRYABLE", "message": "timeout"}, next_allowed_actions=[]),
                "retryable timeout",
            )
        review_count = 1 if self.review else 0
        return ToolResult(
            "run_evaluation_batch",
            {"completed": 1, "review_count": review_count},
            state.copy(
                status=AgentStatus.WAITING_FOR_REVIEW.value if self.review else AgentStatus.RUNNING.value,
                completed_count=1,
                pending_count=0,
                review_count=review_count,
                pending_human_review=self.review,
                next_allowed_actions=["get_review_queue"] if self.review else ["generate_report"],
            ),
            "batch complete",
        )

    def resume_evaluation_batch(self, state):
        self.calls.append("resume_evaluation_batch")
        return ToolResult(
            "resume_evaluation_batch",
            {"completed": 1, "pending": 0},
            state.copy(status=AgentStatus.RUNNING.value, completed_count=1, pending_count=0, last_error={"kind": None, "message": None}, next_allowed_actions=["generate_report"]),
            "resumed",
        )

    def get_review_queue(self, state):
        self.calls.append("get_review_queue")
        return ToolResult(
            "get_review_queue",
            {"review_queue_count": 1},
            state.copy(status=AgentStatus.WAITING_FOR_REVIEW.value, pending_human_review=True, next_allowed_actions=[]),
            "review_queue=1",
        )

    def generate_report(self, state):
        self.calls.append("generate_report")
        return ToolResult(
            "generate_report",
            {"report_ready": True},
            state.copy(status=AgentStatus.COMPLETED.value, report_ready=True, next_allowed_actions=[]),
            "report_ready=True",
        )


class AgentRunnerTests(unittest.TestCase):
    def test_normal_batch_reports(self):
        tools = FakeTools()
        agent = EvaluationAgent(tools)
        final = agent.run(AgentState("t", "in", "out"))
        self.assertEqual(final.status, AgentStatus.COMPLETED)
        self.assertEqual(tools.calls, ["inspect_dataset", "run_evaluation_batch", "generate_report"])

    def test_review_stops_for_human(self):
        tools = FakeTools(review=True)
        agent = EvaluationAgent(tools)
        final = agent.run(AgentState("t", "in", "out"))
        self.assertEqual(final.status, AgentStatus.WAITING_FOR_REVIEW)
        self.assertTrue(final.pending_human_review)
        self.assertNotIn("generate_report", tools.calls)
        self.assertEqual(agent.trace[-1].selected_action, "stop")

    def test_resume_after_human_review_completed(self):
        tools = FakeTools()
        agent = EvaluationAgent(tools)
        state = AgentState("t", "in", "out", status=AgentStatus.WAITING_FOR_REVIEW, pending_human_review=False)
        final = agent.run(state)
        self.assertEqual(final.status, AgentStatus.COMPLETED)
        self.assertEqual(tools.calls, ["generate_report"])

    def test_retryable_partial_resumes(self):
        tools = FakeTools(retryable_once=True)
        agent = EvaluationAgent(tools)
        final = agent.run(AgentState("t", "in", "out"))
        self.assertEqual(final.status, AgentStatus.COMPLETED)
        self.assertIn("resume_evaluation_batch", tools.calls)

    def test_fatal_auth_blocks(self):
        tools = FakeTools(fatal=True)
        agent = EvaluationAgent(tools)
        final = agent.run(AgentState("t", "in", "out"))
        self.assertEqual(final.status, AgentStatus.BLOCKED)
        self.assertEqual(agent.trace[-1].selected_action, "stop")

    def test_agent_cannot_call_write_human_label(self):
        agent = EvaluationAgent(FakeTools())
        with self.assertRaises(ValueError):
            agent.execute_action(AgentState("t", "in", "out"), "write_human_label")

    def test_action_trace_contains_decision_context(self):
        agent = EvaluationAgent(FakeTools())
        agent.run(AgentState("t", "in", "out"))
        first = agent.trace_dicts()[0]
        self.assertIn("state_before", first)
        self.assertIn("observation", first)
        self.assertIn("selected_action", first)
        self.assertIn("state_after", first)


if __name__ == "__main__":
    unittest.main()
