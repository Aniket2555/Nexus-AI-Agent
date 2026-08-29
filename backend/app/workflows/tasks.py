import asyncio
from typing import Any

from backend.app.sandbox import manager as sandbox_manager
from backend.app.workflows.models import StepFn, WorkflowContext, WorkflowDefinition


def sandboxed_code_workflow(
    language: str, code: str, *, timeout_seconds: float | None = None
) -> WorkflowDefinition:
    """One-step workflow wrapping a single §7.1 sandbox run as a cancellable,
    trackable long-running task — §7.2's "large document analysis, code
    refactoring" case, minus the analysis-specific steps: any long sandbox run
    goes through this. `ctx.set_active_container` is what lets
    `WorkflowManager.cancel()` actually kill the container instead of just
    detaching from it.
    """

    async def execute(ctx: WorkflowContext) -> dict[str, Any]:
        result = await sandbox_manager.run(
            language, code, timeout_seconds=timeout_seconds, on_start=ctx.set_active_container
        )
        return {"sandbox_result": result}

    return WorkflowDefinition(name="sandbox_execute", steps=[execute])


def _make_demo_step(index: int, step_seconds: float) -> StepFn:
    async def step(ctx: WorkflowContext) -> dict[str, Any]:
        await asyncio.sleep(step_seconds)
        return {f"step_{index}": "done"}

    return step


def demo_multi_step_workflow(step_count: int = 4, step_seconds: float = 0.05) -> WorkflowDefinition:
    return WorkflowDefinition(
        name="demo_multi_step",
        steps=[_make_demo_step(i, step_seconds) for i in range(step_count)],
    )
