from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from backend.app.graphs.states.shared_state import SpecialistResult
from backend.app.mcp.tools.custom.vision import DEFAULT_QUESTION, analyze_image


class VisionState(TypedDict):
    image_url: str
    question: str
    description: str


async def analyze_node(state: VisionState) -> dict[str, Any]:
    description = await analyze_image(state["image_url"], state.get("question") or DEFAULT_QUESTION)
    return {"description": description}


builder = StateGraph(VisionState)
builder.add_node("analyze", analyze_node)
builder.set_entry_point("analyze")
builder.add_edge("analyze", END)
vision_graph = builder.compile()


async def run_vision(task_id: str, image_url: str, question: str = "") -> SpecialistResult:
    """Reuses §5.2's `analyze_image` directly — the Vision specialist's entire job
    is what that tool already does; wrapping it in a second implementation here
    would be exactly the kind of duplication DECISIONS.md D4 argues against.
    """
    result = await vision_graph.ainvoke({"image_url": image_url, "question": question})

    return SpecialistResult(
        task_id=task_id,
        specialist="vision",
        summary=result.get("description", ""),
        success=bool(result.get("description")),
        citations=[],
        details={"image_url": image_url},
    )
