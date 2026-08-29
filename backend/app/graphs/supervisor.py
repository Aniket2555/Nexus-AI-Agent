import asyncio
from typing import Any, Literal

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from backend.app.graphs.identity import identity_from_config
from backend.app.graphs.nodes.planning import create_plan
from backend.app.graphs.nodes.review import review_draft
from backend.app.graphs.nodes.routing import all_tasks_complete, compute_ready_tasks
from backend.app.graphs.specialists.browser import run_browser
from backend.app.graphs.specialists.code import run_code
from backend.app.graphs.specialists.data import run_data
from backend.app.graphs.specialists.report import run_report
from backend.app.graphs.specialists.research import run_research
from backend.app.graphs.specialists.vision import run_vision
from backend.app.graphs.states.shared_state import SpecialistResult, SubTask
from backend.app.graphs.states.supervisor_state import SupervisorState
from backend.app.loop_engine.reflection import should_retry
from backend.app.observability.metrics import (
    agent_loop_iterations,
    agent_tasks_total,
    push_metrics,
)
from backend.app.utils.messages import as_text


async def planner_node(state: SupervisorState, config: RunnableConfig) -> dict[str, Any]:
    """§6.1's `planner` node: decompose the request into `sub_tasks` once, up front.

    Runs exactly once per top-level invocation — retries (reflect_edge, below) loop
    back to `coordinator`, not here, so a review failure re-synthesizes from the same
    specialist results rather than re-decomposing (and re-running specialists) from
    scratch.
    """
    request = as_text(state["messages"][-1]) if state.get("messages") else ""
    plan, sub_tasks = await create_plan(request)
    return {"current_plan": plan, "sub_tasks": sub_tasks}


async def _run_specialist(task: SubTask, tenant_id: str, user_id: str) -> SpecialistResult:
    specialist = task["specialist"]
    if specialist == "research":
        return await run_research(task["id"], task["description"], tenant_id, user_id)
    if specialist == "code":
        return await run_code(task["id"], task["description"])
    if specialist == "data":
        return await run_data(task["id"], task["description"])
    if specialist == "vision":
        return await run_vision(task["id"], task.get("target") or "", task["description"])
    if specialist == "browser":
        return await run_browser(task["id"], task.get("target") or "", task["description"])

    return SpecialistResult(
        task_id=task["id"],
        specialist=specialist,
        summary="",
        success=False,
        citations=[],
        details={"error": f"Dispatch does not route to specialist {specialist!r}."},
    )


async def dispatch_node(state: SupervisorState, config: RunnableConfig) -> dict[str, Any]:
    """§6.2: one node, run repeatedly via `dispatch_edge` below, rather than
    LangGraph's `Send` API — see routing.py's `compute_ready_tasks` docstring for
    why a dependency-respecting multi-round scheduler doesn't fit `Send`'s
    fixed-arity single-round fan-out. Every task ready in a round runs concurrently
    via `asyncio.gather`; between-round sequencing is the graph loop itself.
    """
    tenant_id, user_id = identity_from_config(config)
    completed_ids = {r["task_id"] for r in state.get("agent_results", [])}
    ready = compute_ready_tasks(state.get("sub_tasks", []), completed_ids)
    if not ready:
        return {}

    results = await asyncio.gather(
        *(_run_specialist(task, tenant_id, user_id) for task in ready)
    )
    return {"agent_results": list(results)}


def dispatch_edge(state: SupervisorState) -> Literal["dispatch", "coordinator"]:
    sub_tasks = state.get("sub_tasks", [])
    completed_ids = {r["task_id"] for r in state.get("agent_results", [])}
    if all_tasks_complete(sub_tasks, completed_ids):
        return "coordinator"
    if not compute_ready_tasks(sub_tasks, completed_ids):
        return "coordinator"
    return "dispatch"


async def coordinator_node(state: SupervisorState, config: RunnableConfig) -> dict[str, Any]:
    """§6.1's `coordinator` node: merge every specialist's output into one
    response. Always calls the report specialist itself, exactly once per round of
    this loop — the planner is instructed never to schedule its own "report"
    sub-task (planning.py), so synthesis never runs twice.
    """
    request = as_text(state["messages"][0]) if state.get("messages") else ""
    if state.get("review_feedback"):
        request += (
            "\n\nRevision needed: the previous draft failed quality review. "
            f"Feedback: {state['review_feedback']}"
            "\nRevise your synthesis to address this."
        )

    result = await run_report("coordinator", request, state.get("agent_results", []))
    return {"draft_response": result["summary"]}


async def reviewer_node(state: SupervisorState, config: RunnableConfig) -> dict[str, Any]:
    """§6.1's `reviewer` node: grounding/hallucination check, distinct from the
    generic quality score reflect_node below derives from it.
    """
    passed, feedback, cost_usd = await review_draft(
        state.get("draft_response") or "", state.get("agent_results", [])
    )
    return {
        "review_passed": passed,
        "review_feedback": feedback,
        "cost_usd_spent": state.get("cost_usd_spent", 0.0) + cost_usd,
    }


def reflect_node(state: SupervisorState) -> dict[str, Any]:
    """§6.1's `reflection` node: turn the reviewer's pass/fail into the
    quality_score/iteration bookkeeping `reflect_edge` decides on — deliberately
    not a second LLM call. The reviewer already made the judgement; this just
    records it in the same shape Phase 3's reflection loop uses, so
    `loop_engine.reflection.should_retry` (built for a 0.0-1.0 score) can be reused
    verbatim for the retry decision instead of a second bespoke policy.
    """
    passed = bool(state.get("review_passed"))
    return {
        "quality_score": 1.0 if passed else 0.0,
        "reflection_feedback": state.get("review_feedback", ""),
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def reflect_edge(state: SupervisorState) -> Literal["retry", "escalate", "finalize"]:
    if state.get("review_passed"):
        return "finalize"

    decision = should_retry(
        state.get("iteration_count", 0),
        state.get("quality_score", 0.0),
        state.get("cost_usd_spent", 0.0),
    )
    return "retry" if decision == "retry" else "escalate"


def escalate_node(state: SupervisorState) -> dict[str, Any]:
    """§6.2's "Escalation" handoff: repeated review failure (iteration cap or cost
    ceiling hit, per `reflect_edge`) pauses for a human decision rather than
    silently shipping an ungrounded answer — same `interrupt()` mechanism as Phase
    3's `human_approval.py`, called inline here rather than by invoking that
    module's separately-compiled graph: a graph served by LangGraph Platform (which
    this one is, per langgraph.json) gets Platform's own injected persistence, so
    `interrupt()` works directly without this graph needing its own checkpointer.
    """
    decision = interrupt(
        {
            "question": (
                "The multi-agent supervisor could not produce a response that "
                "passed quality review after multiple attempts. Use the last "
                "draft anyway?"
            ),
            "draft": state.get("draft_response"),
            "review_feedback": state.get("review_feedback"),
            "iteration_count": state.get("iteration_count", 0),
        }
    )
    approved = bool(decision.get("approved") if isinstance(decision, dict) else decision)

    if approved:
        return {"final_response": state.get("draft_response")}

    return {
        "final_response": (
            "I wasn't able to produce a response that passed quality review, and "
            "a human reviewer declined to approve the best draft. Last review "
            f"feedback: {state.get('review_feedback', 'none')}"
        )
    }


def finalize_node(state: SupervisorState) -> dict[str, Any]:
    """The one exit point every run reaches — happy path, retried-then-passed,
    or escalated — so it's also §8.6's one place to record "did this run end
    up passing review" and "how many reflection iterations did it take",
    rather than duplicating that bookkeeping at every path that can reach here.
    """
    final = state.get("final_response") or state.get("draft_response") or ""

    agent_tasks_total.labels(
        status="passed" if state.get("review_passed") else "escalated"
    ).inc()
    agent_loop_iterations.observe(state.get("iteration_count", 0))
    push_metrics("nexus_supervisor")

    return {"final_response": final, "messages": [AIMessage(content=final)]}


builder = StateGraph(SupervisorState)
builder.add_node("planner", planner_node)
builder.add_node("dispatch", dispatch_node)
builder.add_node("coordinator", coordinator_node)
builder.add_node("reviewer", reviewer_node)
builder.add_node("reflect", reflect_node)
builder.add_node("escalate", escalate_node)
builder.add_node("finalize", finalize_node)

builder.set_entry_point("planner")
builder.add_edge("planner", "dispatch")
builder.add_conditional_edges("dispatch", dispatch_edge, ["dispatch", "coordinator"])
builder.add_edge("coordinator", "reviewer")
builder.add_edge("reviewer", "reflect")
builder.add_conditional_edges(
    "reflect", reflect_edge, {"retry": "coordinator", "escalate": "escalate", "finalize": "finalize"}
)
builder.add_edge("escalate", "finalize")
builder.add_edge("finalize", END)

supervisor_graph = builder.compile()
