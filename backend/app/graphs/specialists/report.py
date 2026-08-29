from typing import Any, TypedDict

from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph

from backend.app.graphs.states.shared_state import SpecialistResult
from backend.app.llm.provider import get_chat_model
from backend.app.security.content_boundary import wrap_untrusted_content
from backend.app.utils.messages import as_text

REPORT_PROMPT = (
    "You are the Report specialist in a multi-agent system. Synthesize\n"
    "the specialist results below into one coherent, well-organized response to the\n"
    "original request. Preserve citations from research results exactly as given\n"
    "([Source: <file>, Page: <n>]). Note plainly if any specialist failed rather than\n"
    "silently omitting its task.\n\n"
    "Original request: {request}\n\n{results_block}\n"
)


class ReportState(TypedDict):
    request: str
    agent_results: list[SpecialistResult]
    report: str


def format_results(results: list[SpecialistResult]) -> str:
    blocks = []
    for result in results:
        status = "succeeded" if result.get("success") else "FAILED"
        citations = result.get("citations") or []
        citation_lines = "\n".join(
            f"  [Source: {c.get('source', 'unknown')}, Page {c.get('page', '?')}]"
            for c in citations
        )
        block = f"## {result['specialist']} ({status})\n{result.get('summary', '')}"
        if citation_lines:
            block += f"\n{citation_lines}"
        blocks.append(block)
    return "\n\n".join(blocks) if blocks else "(no specialist results were produced)"


async def synthesize_node(state: ReportState) -> dict[str, Any]:
    results_block = wrap_untrusted_content(format_results(state["agent_results"]))
    prompt = REPORT_PROMPT.format(request=state["request"], results_block=results_block)
    response = await get_chat_model().ainvoke([SystemMessage(content=prompt)])
    return {"report": as_text(response)}


builder = StateGraph(ReportState)
builder.add_node("synthesize", synthesize_node)
builder.set_entry_point("synthesize")
builder.add_edge("synthesize", END)
report_graph = builder.compile()


async def run_report(
    task_id: str, request: str, agent_results: list[SpecialistResult]
) -> SpecialistResult:
    result = await report_graph.ainvoke({"request": request, "agent_results": agent_results})
    all_citations = [c for r in agent_results for c in (r.get("citations") or [])]

    return SpecialistResult(
        task_id=task_id,
        specialist="report",
        summary=result.get("report", ""),
        success=bool(result.get("report")),
        citations=all_citations,
        details={"synthesized_from": [r["specialist"] for r in agent_results]},
    )
