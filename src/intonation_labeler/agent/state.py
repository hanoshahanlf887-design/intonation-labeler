from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class AgentStatus(str, Enum):
    NEW = "NEW"
    RUNNING = "RUNNING"
    PARTIAL = "PARTIAL"
    WAITING_FOR_REVIEW = "WAITING_FOR_REVIEW"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"


class ErrorKind(str, Enum):
    RETRYABLE = "RETRYABLE"
    NON_RETRYABLE = "NON_RETRYABLE"
    FATAL = "FATAL"


@dataclass
class AgentError:
    kind: str | None = None
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentState:
    task_id: str
    input_location: str
    output_dir: str
    provider: str = "dry-run"
    model: str = "demo-model"
    status: AgentStatus = AgentStatus.NEW
    total_samples: int | None = None
    completed_count: int = 0
    failed_count: int = 0
    pending_count: int = 0
    review_count: int = 0
    pending_human_review: bool = False
    report_ready: bool = False
    last_error: AgentError = field(default_factory=AgentError)
    next_allowed_actions: list[str] = field(default_factory=list)
    manifest_path: str | None = None
    dry_run: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.status, AgentStatus):
            self.status = AgentStatus(self.status)
        if isinstance(self.last_error, dict):
            self.last_error = AgentError(**self.last_error)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentState":
        payload = dict(data)
        payload["status"] = AgentStatus(payload.get("status", AgentStatus.NEW.value))
        error = payload.get("last_error") or {}
        if isinstance(error, dict):
            payload["last_error"] = AgentError(**error)
        return cls(**payload)

    def copy(self, **updates: Any) -> "AgentState":
        data = self.to_dict()
        data.update(updates)
        return AgentState.from_dict(data)
