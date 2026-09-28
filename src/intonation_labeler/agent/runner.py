from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .policy import decide_next_action, validate_action
from .state import AgentState
from .tools import EvaluationAgentTools, ToolResult


@dataclass
class ActionTraceEntry:
    state_before: dict[str, Any]
    observation: dict[str, Any]
    selected_action: str
    tool_result_summary: str
    state_after: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_before": self.state_before,
            "observation": self.observation,
            "selected_action": self.selected_action,
            "tool_result_summary": self.tool_result_summary,
            "state_after": self.state_after,
        }


class EvaluationAgent:
    def __init__(self, tools: EvaluationAgentTools | None = None) -> None:
        self.tools = tools or EvaluationAgentTools()
        self.trace: list[ActionTraceEntry] = []

    def run(self, state: AgentState, max_steps: int = 20) -> AgentState:
        current = state
        for _ in range(max_steps):
            decision = decide_next_action(current)
            if decision.action == "stop":
                self._record(current, decision.action, {"reason": decision.reason}, "stop", current)
                break
            result = self.execute_action(current, decision.action)
            self._record(current, decision.action, result.observation, result.summary, result.state)
            current = result.state
        return current

    def execute_action(self, state: AgentState, action: str, **kwargs: Any) -> ToolResult:
        validate_action(action)
        method = getattr(self.tools, action, None)
        if method is None:
            raise ValueError(f"No tool implementation for action: {action}")
        return method(state, **kwargs)

    def _record(
        self,
        state_before: AgentState,
        selected_action: str,
        observation: dict[str, Any],
        summary: str,
        state_after: AgentState,
    ) -> None:
        self.trace.append(
            ActionTraceEntry(
                state_before=state_before.to_dict(),
                observation=observation,
                selected_action=selected_action,
                tool_result_summary=summary,
                state_after=state_after.to_dict(),
            )
        )

    def trace_dicts(self) -> list[dict[str, Any]]:
        return [entry.to_dict() for entry in self.trace]
