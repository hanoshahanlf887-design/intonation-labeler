from __future__ import annotations

from dataclasses import dataclass

from .state import AgentState, AgentStatus


@dataclass(frozen=True)
class PolicyDecision:
    action: str
    reason: str


FORBIDDEN_ACTIONS = {
    "modify_prompt",
    "modify_validation_rule",
    "modify_threshold",
    "write_human_label",
    "enable_acoustic_conflict",
    "git_commit",
    "git_push",
}

ALLOWED_ACTIONS = {
    "inspect_dataset",
    "run_evaluation_batch",
    "resume_evaluation_batch",
    "get_batch_status",
    "get_validation_summary",
    "get_review_queue",
    "get_case_detail",
    "generate_report",
    "stop",
}


def validate_action(action: str) -> None:
    if action in FORBIDDEN_ACTIONS:
        raise ValueError(f"Action is forbidden by the evaluation-agent guardrails: {action}")
    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"Action is not in the evaluation-agent whitelist: {action}")


def decide_next_action(state: AgentState) -> PolicyDecision:
    if state.status == AgentStatus.NEW:
        return PolicyDecision("inspect_dataset", "New task must inspect input before running evaluation.")

    if state.status == AgentStatus.WAITING_FOR_REVIEW:
        if state.pending_human_review and state.next_allowed_actions:
            action = state.next_allowed_actions[0]
            validate_action(action)
            return PolicyDecision(action, "Review is pending; prepare the review queue before stopping for a human.")
        if state.pending_human_review:
            return PolicyDecision("stop", "Human review is pending; the agent must not write human labels.")
        return PolicyDecision("generate_report", "Human review is complete; final report can be generated.")

    if state.status in {AgentStatus.FAILED, AgentStatus.BLOCKED, AgentStatus.COMPLETED}:
        return PolicyDecision("stop", f"Task status is {state.status.value}; no autonomous action is allowed.")

    if state.next_allowed_actions:
        action = state.next_allowed_actions[0]
        validate_action(action)
        return PolicyDecision(action, "State exposes an explicit safe next action from the previous observation.")

    if state.status == AgentStatus.RUNNING:
        return PolicyDecision("get_batch_status", "Running task should observe current batch status.")

    if state.status == AgentStatus.PARTIAL:
        if state.last_error.kind == "RETRYABLE":
            return PolicyDecision("resume_evaluation_batch", "Partial task has retryable failure; resume without overwriting successes.")
        if state.last_error.kind == "FATAL":
            return PolicyDecision("stop", "Fatal error blocks further autonomous action.")
        if state.pending_count > 0 and not state.last_error.kind:
            return PolicyDecision("resume_evaluation_batch", "Partial task still has pending samples.")
        return PolicyDecision("stop", "Partial task has no safe automatic next action.")

    if state.total_samples and state.completed_count >= state.total_samples:
        if state.review_count > 0 or state.pending_human_review:
            return PolicyDecision("get_review_queue", "Completed batch has review cases; prepare review queue.")
        return PolicyDecision("generate_report", "Completed batch has no review cases; generate report.")

    return PolicyDecision("stop", "No policy rule matched this state.")

