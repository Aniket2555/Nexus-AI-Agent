"""Exercises §7.2's workflow cancellation against a *real* running Docker
container — the part `tests/unit/test_workflow_manager.py` can't cover, since
that only exercises the demo (Docker-free) workflow. Requires a running Docker
daemon, same as tests/integration/test_sandbox.py.
"""

import asyncio

import docker
import docker.errors

from backend.app.workflows.manager import WorkflowManager
from backend.app.workflows.models import WorkflowStatus
from backend.app.workflows.tasks import sandboxed_code_workflow


async def _wait_for_status(manager, workflow_id, statuses, *, timeout=10.0):
    async def poll():
        while manager.get(workflow_id).status not in statuses:
            await asyncio.sleep(0.05)

    await asyncio.wait_for(poll(), timeout=timeout)
    return manager.get(workflow_id)


async def test_cancel_kills_the_actual_running_container():
    manager = WorkflowManager()

    definition = sandboxed_code_workflow(
        "python", "import time; time.sleep(60)", timeout_seconds=90
    )

    workflow_id = manager.start(definition, context={})

    await asyncio.wait_for(
        _poll_until(lambda: manager.get(workflow_id).active_container_id is not None),
        timeout=10.0,
    )

    container_id = manager.get(workflow_id).active_container_id

    client = docker.from_env()

    assert client.containers.get(container_id).status in ("running", "created")

    cancelled = manager.cancel(workflow_id)

    assert cancelled is True

    await asyncio.sleep(0.5)

    try:
        assert client.containers.get(container_id).status != "running"
    except docker.errors.NotFound:
        pass

    record = await _wait_for_status(manager, workflow_id, WorkflowStatus.CANCELLED, timeout=10.0)

    assert record.status == WorkflowStatus.CANCELLED

    async def container_is_gone() -> bool:
        try:
            client.containers.get(container_id)
            return False
        except docker.errors.NotFound:
            return True

    await asyncio.wait_for(_poll_until_async(container_is_gone), timeout=10.0)


async def _poll_until(predicate, interval: float = 0.05) -> None:
    while not predicate():
        await asyncio.sleep(interval)


async def _poll_until_async(predicate, interval: float = 0.1) -> None:
    while not await predicate():
        await asyncio.sleep(interval)
