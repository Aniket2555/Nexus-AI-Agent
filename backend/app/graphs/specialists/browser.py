from typing import Any, TypedDict

from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph

from backend.app.graphs.states.shared_state import SpecialistResult
from backend.app.llm.provider import get_chat_model
from backend.app.mcp.client import get_client
from backend.app.security.egress import EgressDeniedError, check_url_allowed
from backend.app.utils.messages import as_text

SUMMARIZE_PROMPT = (
    "Summarize the following page content to answer the task.\n\n"
    "Task: {task}\n\nPage content:\n{content}\n"
)

_NAVIGATE_CANDIDATES = ("browser_navigate", "navigate", "goto")
_CONTENT_CANDIDATES = ("browser_snapshot", "browser_get_text", "get_content", "get_text")


class BrowserState(TypedDict):
    url: str
    task: str
    page_content: str
    summary: str
    error: str | None


def _find_tool(tools: dict, candidates: tuple[str, ...]):
    for name in candidates:
        if name in tools:
            return tools[name]
    return None


async def fetch_node(state: BrowserState) -> dict[str, Any]:
    """Navigate to `url` via the playwright MCP server (§5.1) and extract its
    content.

    The egress allowlist (§5.5) is checked *first*, before the MCP server is even
    asked to navigate — an unreachable/misconfigured server should never be the
    only thing standing between an injected instruction and an arbitrary URL.
    """
    try:
        check_url_allowed(state["url"])
    except EgressDeniedError as exc:
        return {"page_content": "", "error": str(exc)}

    try:
        client = get_client(server_names=["playwright"])
        tools = {t.name: t for t in await client.get_tools(server="playwright")}
    except Exception as exc:
        return {"page_content": "", "error": f"Could not reach the browser MCP server: {exc}"}

    navigate = _find_tool(tools, _NAVIGATE_CANDIDATES)
    get_content = _find_tool(tools, _CONTENT_CANDIDATES)
    if navigate is None or get_content is None:
        return {
            "page_content": "",
            "error": (
                "Playwright server did not expose an expected navigate/content "
                f"tool. Available: {sorted(tools)}"
            ),
        }

    await navigate.ainvoke({"url": state["url"]})
    result = await get_content.ainvoke({})

    if isinstance(result, list) and result:
        content = result[0]["text"]
    else:
        content = str(result)

    return {"page_content": content, "error": None}


async def summarize_node(state: BrowserState) -> dict[str, Any]:
    if state.get("error"):
        return {"summary": f"Could not complete the browsing task: {state['error']}"}

    prompt = SUMMARIZE_PROMPT.format(task=state["task"], content=state["page_content"][:8000])
    response = await get_chat_model().ainvoke([SystemMessage(content=prompt)])
    return {"summary": as_text(response)}


builder = StateGraph(BrowserState)
builder.add_node("fetch", fetch_node)
builder.add_node("summarize", summarize_node)
builder.set_entry_point("fetch")
builder.add_edge("fetch", "summarize")
builder.add_edge("summarize", END)
browser_graph = builder.compile()


async def run_browser(task_id: str, url: str, task: str) -> SpecialistResult:
    result = await browser_graph.ainvoke({"url": url, "task": task})

    return SpecialistResult(
        task_id=task_id,
        specialist="browser",
        summary=result.get("summary", ""),
        success=result.get("error") is None,
        citations=[],
        details={"url": url, "error": result.get("error")},
    )
