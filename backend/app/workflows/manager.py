import asyncio
import uuid

from backend.app.sandbox.docker_sandbox import get_docker_client
from backend.app.workflows.models import (
    WorkflowContext,
    WorkflowDefinition,
    WorkflowRecord,
    WorkflowStatus,
)


class WorkflowNotFoundError(Exception):
    pass


class WorkflowNotResumableError(Exception):
    pass


class WorkflowManager:
    """In-process async workflow runner (§7.2).

    Deliberately not Celery/Temporal: this project's reuse posture has already
    favored built-ins over new queue/orchestration infra everywhere a built-in
    covers the need, and here the need — "run a sequence of async steps,
    report progress, allow cancel/resume" — is exactly what an `asyncio.Task`
    walking a list of step functions over a shared context dict already does.

    State lives in an in-memory dict, matching the same "in-process job table,
    does not survive a restart" posture `api/v1/documents.py`'s `_JOBS` already
    uses for ingestion jobs — not a new pattern introduced here. `resume()`
    therefore means "continue a workflow this process still has a record of
    after cancellation or a step failure," not "recover after the server
    itself restarted." Durable, cross-restart resume would need a persistence
    layer this project has consistently deferred adding until something
    concretely needs it.
    """

    def __init__(self) -> None:
        self._records: dict[str, WorkflowRecord] = {}

    def start(self, definition: WorkflowDefinition, context: dict) -> str:
        workflow_id = str(uuid.uuid4())
        record = WorkflowRecord(workflow_id, definition, context=context)
        self._records[workflow_id] = record
        record.task = asyncio.create_task(self._run(record))
        return workflow_id

    def get(self, workflow_id: str) -> WorkflowRecord:
        try:
            return self._records[workflow_id]
        except KeyError:
            raise WorkflowNotFoundError(workflow_id) from None

    async def _run(self, record: WorkflowRecord, *, resume_from: int = 0) -> None:
        record.status = WorkflowStatus.RUNNING
        total = len(record.definition.steps)

        def register_container(container_id: str | None) -> None:
            record.active_container_id = container_id

        ctx = WorkflowContext(record.context, _on_container_started=register_container)

        try:
            for index in range(resume_from, total):
                record.message = f"running step {index + 1}/{total}"
                update = await record.definition.steps[index](ctx)
                record.context.update(update)
                record.step_index = index + 1
                record.progress = record.step_index / total if total else 1.0

            record.status = WorkflowStatus.COMPLETED
            record.message = "completed"
        except asyncio.CancelledError:
            record.status = WorkflowStatus.CANCELLED
            record.message = f"cancelled after step {record.step_index}/{total}"
        except Exception as exc:
            record.status = WorkflowStatus.FAILED
            record.error = f"{type(exc).__name__}: {exc}"
            record.message = f"failed at step {record.step_index + 1}/{total}"
        finally:
            record.active_container_id = None

    def cancel(self, workflow_id: str) -> bool:
        """Kill whatever's actually running, not just the `asyncio.Task`.

        A step whose body awaits `asyncio.to_thread(...)` — every sandbox call,
        since `docker-py` is a sync client (pyproject.toml's note) — can't be
        interrupted by `Task.cancel()` alone: the worker thread runs the
        blocking Docker call to completion regardless of what the awaiting
        coroutine does, so cancelling would otherwise look like it worked (the
        API call returns right away) while the container kept running to its
        own timeout. Killing the registered container directly is what makes
        "container killed" real, verified directly in
        tests/integration/test_workflows.py, rather than just "the caller
        stopped waiting for it."
        """
        record = self.get(workflow_id)

        if record.active_container_id:
            try:
                get_docker_client().containers.get(record.active_container_id).kill()
            except Exception:
                pass

        if record.task is not None and not record.task.done():
            record.task.cancel()
            return True

        return False

    def resume(self, workflow_id: str) -> None:
        record = self.get(workflow_id)

        if record.status not in (WorkflowStatus.FAILED, WorkflowStatus.CANCELLED):
            raise WorkflowNotResumableError(
                f"Workflow {workflow_id!r} is {record.status.value!r}, not resumable."
            )

        record.error = None
        record.task = asyncio.create_task(self._run(record, resume_from=record.step_index))


_manager: WorkflowManager | None = None


def get_workflow_manager() -> WorkflowManager:
    global _manager
    if _manager is None:
        _manager = WorkflowManager()
    return _manager
