import asyncio
import json
from collections.abc import AsyncIterator
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.app.api.middleware.auth import Principal
from backend.app.api.middleware.rate_limit import enforce_api_rate_limit
from backend.app.workflows.manager import (
    WorkflowNotFoundError,
    WorkflowNotResumableError,
    get_workflow_manager,
)
from backend.app.workflows.models import TERMINAL_STATUSES
from backend.app.workflows.tasks import demo_multi_step_workflow, sandboxed_code_workflow

router = APIRouter(prefix="/workflows", tags=["workflows"])

_STREAM_POLL_SECONDS = 0.25


class SandboxWorkflowRequest(BaseModel):
    language: Literal["python", "node", "shell"]
    code: str
    timeout_seconds: float | None = None


class DemoWorkflowRequest(BaseModel):
    step_count: int = 4
    step_seconds: float = 0.5


@router.post("/sandbox", status_code=status.HTTP_202_ACCEPTED)
async def start_sandbox_workflow(
    request: SandboxWorkflowRequest,
    principal: Annotated[Principal, Depends(enforce_api_rate_limit)],
) -> dict[str, Any]:
    manager = get_workflow_manager()
    definition = sandboxed_code_workflow(
        request.language, request.code, timeout_seconds=request.timeout_seconds
    )
    workflow_id = manager.start(definition, context={})
    return {"workflow_id": workflow_id, "status": "pending"}


@router.post("/demo", status_code=status.HTTP_202_ACCEPTED)
async def start_demo_workflow(
    request: DemoWorkflowRequest,
    principal: Annotated[Principal, Depends(enforce_api_rate_limit)],
) -> dict[str, Any]:
    """Exercises §7.2's progress/cancel/resume mechanics without needing
    Docker — see `tasks.py`'s `demo_multi_step_workflow` docstring.
    """
    manager = get_workflow_manager()
    definition = demo_multi_step_workflow(request.step_count, request.step_seconds)
    workflow_id = manager.start(definition, context={})
    return {"workflow_id": workflow_id, "status": "pending"}


@router.get("/{workflow_id}")
async def get_workflow(
    workflow_id: str, principal: Annotated[Principal, Depends(enforce_api_rate_limit)]
) -> dict[str, Any]:
    try:
        record = get_workflow_manager().get(workflow_id)
    except WorkflowNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown workflow id.") from None

    return record.to_public_dict()


@router.post("/{workflow_id}/cancel")
async def cancel_workflow(
    workflow_id: str, principal: Annotated[Principal, Depends(enforce_api_rate_limit)]
) -> dict[str, Any]:
    try:
        cancelled = get_workflow_manager().cancel(workflow_id)
    except WorkflowNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown workflow id.") from None

    return {"workflow_id": workflow_id, "cancel_requested": cancelled}


@router.post("/{workflow_id}/resume")
async def resume_workflow(
    workflow_id: str, principal: Annotated[Principal, Depends(enforce_api_rate_limit)]
) -> dict[str, Any]:
    try:
        get_workflow_manager().resume(workflow_id)
    except WorkflowNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown workflow id.") from None
    except WorkflowNotResumableError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from None

    return {"workflow_id": workflow_id, "resumed": True}


@router.get("/{workflow_id}/stream")
async def stream_workflow(
    workflow_id: str, principal: Annotated[Principal, Depends(enforce_api_rate_limit)]
) -> StreamingResponse:
    """Server-sent progress events — §7.2's "progress bar updates in UI." Polls
    the same in-memory record `GET /{workflow_id}` reads, at
    `_STREAM_POLL_SECONDS`, until the workflow reaches a terminal status.

    Note: a browser's native `EventSource` can't set custom request headers,
    so a real frontend consuming this endpoint would need to either fetch it
    via `fetch()` + a manual stream reader (headers allowed) or move the key
    into a short-lived query param/cookie exchanged for it server-side — a
    frontend-side concern out of scope for this middleware pass, not a gap in
    this endpoint's own auth.
    """
    manager = get_workflow_manager()
    try:
        manager.get(workflow_id)
    except WorkflowNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown workflow id.") from None

    async def events() -> AsyncIterator[str]:
        while True:
            record = manager.get(workflow_id)
            payload = record.to_public_dict()
            yield f"data: {json.dumps(payload)}\n\n"

            if record.status in TERMINAL_STATUSES:
                return

            await asyncio.sleep(_STREAM_POLL_SECONDS)

    return StreamingResponse(events(), media_type="text/event-stream")
