from typing import Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt


class ApprovalState(TypedDict):
    pending_action: dict[str, Any]
    impact_assessment: str
    human_approved: bool


def human_approval_node(state: ApprovalState) -> dict[str, Any]:
    """Pause execution for human review (§3.3), via LangGraph's interrupt().

    `interrupt()` raises a control-flow signal the runtime catches: it persists
    state via the graph's checkpointer and returns the interrupt payload to the
    caller instead of a normal result. Resuming means re-invoking the *same*
    thread_id with `Command(resume=<decision>)`, which is what turns this call
    into whatever `decision` the human supplied. None of this works without a
    checkpointer to reload state from between the pause and the resume.
    """
    decision = interrupt(
        {
            "question": "The agent wants to execute the following action. Approve?",
            "action": state.get("pending_action", {}),
            "estimated_impact": state.get("impact_assessment", "unknown"),
        }
    )
    return {"human_approved": decision.get("approved", False)}


builder = StateGraph(ApprovalState)
builder.add_node("human_approval", human_approval_node)
builder.set_entry_point("human_approval")
builder.add_edge("human_approval", END)
human_approval_graph = builder.compile(checkpointer=InMemorySaver())
