import asyncio

import pytest

from backend.app.workflows.manager import (
    WorkflowManager,
    WorkflowNotFoundError,
    WorkflowNotResumableError,
)
from backend.app.workflows.models import WorkflowStatus
from backend.app.workflows.tasks import demo_multi_step_workflow


async def _wait_for_status(manager, workflow_id, statuses, *, timeout=2.0):
    async def poll():
        while manager.get(workflow_id).status not in statuses:
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout=timeout)
    return manager.get(workflow_id)


async def test_start_runs_all_steps_and_completes():
    manager = WorkflowManager()
    definition = demo_multi_step_workflow(3, 0.01)

    workflow_id = manager.start(definition, {})

    record = await _wait_for_status(manager, workflow_id, WorkflowStatus.COMPLETED)

    assert record.step_index == 3
    assert record.progress == 1.0
    assert record.context == {"step_0": "done", "step_1": "done", "step_2": "done"}


async def test_get_unknown_workflow_raises():
    manager = WorkflowManager()

    with pytest.raises(WorkflowNotFoundError):
        manager.get("does-not-exist")


async def test_cancel_stops_a_running_workflow_partway():
    manager = WorkflowManager()
    definition = demo_multi_step_workflow(5, 0.2)
    workflow_id = manager.start(definition, {})

    await asyncio.sleep(0.25)
    cancelled = manager.cancel(workflow_id)

    record = await _wait_for_status(manager, workflow_id, WorkflowStatus.CANCELLED)

    assert cancelled is True
    assert record.status == WorkflowStatus.CANCELLED
    assert record.step_index < 5


async def test_resume_continues_from_the_last_completed_step_not_from_scratch():
    manager = WorkflowManager()
    definition = demo_multi_step_workflow(4, 0.15)
    workflow_id = manager.start(definition, {})

    await asyncio.sleep(0.2)
    manager.cancel(workflow_id)
    await _wait_for_status(manager, workflow_id, WorkflowStatus.CANCELLED)

    step_index_at_cancel = manager.get(workflow_id).step_index

    manager.resume(workflow_id)

    record = await _wait_for_status(manager, workflow_id, WorkflowStatus.COMPLETED, timeout=3.0)

    assert step_index_at_cancel >= 1
    assert record.status == WorkflowStatus.COMPLETED
    assert record.step_index == 4


async def test_resume_rejects_a_still_running_workflow():
    manager = WorkflowManager()
    definition = demo_multi_step_workflow(2, 0.3)
    workflow_id = manager.start(definition, {})

    with pytest.raises(WorkflowNotResumableError):
        manager.resume(workflow_id)


async def test_a_failing_step_marks_the_workflow_failed_with_the_error_recorded():
    async def boom(ctx):
        raise ValueError("step blew up")

    from backend.app.workflows.models import WorkflowDefinition

    manager = WorkflowManager()
    definition = WorkflowDefinition(name="boom", steps=[boom])
    workflow_id = manager.start(definition, {})

    record = await _wait_for_status(manager, workflow_id, WorkflowStatus.FAILED)

    assert "step blew up" in record.error


async def test_cancel_on_already_completed_workflow_is_a_no_op():
    manager = WorkflowManager()
    definition = demo_multi_step_workflow(1, 0.01)
    workflow_id = manager.start(definition, {})

    await _wait_for_status(manager, workflow_id, WorkflowStatus.COMPLETED)

    assert manager.cancel(workflow_id) is False
