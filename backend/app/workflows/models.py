from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class WorkflowStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATUSES = (WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED)


@dataclass
class WorkflowContext:
    """What a step function actually gets — the shared `data` dict every step
    reads and writes, plus a narrow hook for registering a container it starts
    (`manager.py`'s `cancel()` needs a container id to kill for real; a step
    doesn't need — and shouldn't get — the rest of `WorkflowRecord`'s
    bookkeeping to do that).
    """

    data: dict[str, Any]
    _on_container_started: Callable[[str | None], None] | None = None

    def set_active_container(self, container_id: str | None) -> None:
        self._on_container_started(container_id)


StepFn = Callable[[WorkflowContext], Awaitable[dict[str, Any]]]


@dataclass
class WorkflowDefinition:
    name: str
    steps: list[StepFn]


@dataclass
class WorkflowRecord:
    workflow_id: str
    definition: WorkflowDefinition
    context: dict[str, Any]
    status: WorkflowStatus = WorkflowStatus.PENDING
    step_index: int = 0
    progress: float = 0.0
    message: str = "queued"
    error: str | None = None
    active_container_id: str | None = None
    task: Any = field(default=None, repr=False, compare=False)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "workflow_type": self.definition.name,
            "status": self.status.value,
            "step_index": self.step_index,
            "total_steps": len(self.definition.steps),
            "progress": self.progress,
            "message": self.message,
            "error": self.error,
            "result": self.context if self.status == WorkflowStatus.COMPLETED else None,
        }
