import json
import re
from typing import Any, TypedDict

from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph

from backend.app.graphs.states.shared_state import SpecialistResult
from backend.app.llm.provider import get_chat_model
from backend.app.mcp.tools.custom.sql import UnsafeQueryError, run_readonly_query
from backend.app.utils.messages import as_text

SCHEMA_QUERY = (
    "SELECT table_name, column_name, data_type FROM information_schema.columns "
    "WHERE table_schema = 'public' ORDER BY table_name, ordinal_position"
)

GENERATE_SQL_PROMPT = (
    "You are the Data specialist in a multi-agent system. Given the\n"
    "task and the known table schema below, write a single read-only SQL query "
    "(SELECT or\nWITH...SELECT only — no INSERT/UPDATE/DELETE/DDL) that would "
    "answer it. Output ONLY the\nSQL query, no explanation, no markdown fences.\n\n"
    "Known schema:\n{schema}\n\nTask: {task}\n"
)

SUMMARIZE_PROMPT = (
    "Summarize these query results in 2-4 sentences, directly answering\n"
    "the task. Task: {task}\n\nResults ({row_count} rows, showing up to 20):\n{rows}\n"
)

_SQL_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class DataState(TypedDict):
    task: str
    schema: str
    sql_query: str
    rows: list[dict[str, Any]]
    error: str | None
    summary: str


def _extract_sql(text: str) -> str:
    """Models wrap SQL in ```sql fences despite being asked not to often enough
    that this is worth stripping defensively rather than trusting the prompt."""
    match = _SQL_FENCE.search(text)
    return (match.group(1) if match else text).strip()


async def get_schema_summary() -> str:
    """Reuses sql.py's `run_readonly_query` directly (§5.2) — the schema
    introspection query is just another read-only SELECT."""
    rows = await run_readonly_query(SCHEMA_QUERY)
    if not rows:
        return "(no tables exist in the public schema yet)"

    by_table: dict[str, list[str]] = {}
    for row in rows:
        column = f"{row['column_name']} ({row['data_type']})"
        by_table.setdefault(row["table_name"], []).append(column)

    return "\n".join(f"{table}: {', '.join(cols)}" for table, cols in by_table.items())


async def schema_node(state: DataState) -> dict[str, Any]:
    return {"schema": await get_schema_summary()}


async def generate_sql_node(state: DataState) -> dict[str, Any]:
    model = get_chat_model(temperature=0)
    prompt = GENERATE_SQL_PROMPT.format(schema=state["schema"], task=state["task"])
    response = await model.ainvoke([SystemMessage(content=prompt)])
    return {"sql_query": _extract_sql(as_text(response))}


async def execute_node(state: DataState) -> dict[str, Any]:
    """`run_readonly_query` already enforces the read-only/single-statement check
    (§5.2) — a query the model generated that tries to mutate anything is rejected
    here, reported as an analysis failure, not raised as an uncaught exception
    that would crash the specialist.
    """
    try:
        rows = await run_readonly_query(state["sql_query"])
        return {"rows": rows, "error": None}
    except UnsafeQueryError as exc:
        return {"rows": [], "error": str(exc)}
    except Exception as exc:
        return {"rows": [], "error": f"{type(exc).__name__}: {exc}"}


async def summarize_node(state: DataState) -> dict[str, Any]:
    if state.get("error"):
        return {"summary": f"Could not complete the analysis: {state['error']}"}

    prompt = SUMMARIZE_PROMPT.format(
        task=state["task"],
        row_count=len(state["rows"]),
        rows=json.dumps(state["rows"][:20], default=str),
    )
    response = await get_chat_model().ainvoke([SystemMessage(content=prompt)])
    return {"summary": as_text(response)}


builder = StateGraph(DataState)
builder.add_node("schema", schema_node)
builder.add_node("generate_sql", generate_sql_node)
builder.add_node("execute", execute_node)
builder.add_node("summarize", summarize_node)
builder.set_entry_point("schema")
builder.add_edge("schema", "generate_sql")
builder.add_edge("generate_sql", "execute")
builder.add_edge("execute", "summarize")
builder.add_edge("summarize", END)
data_graph = builder.compile()


async def run_data(task_id: str, task: str) -> SpecialistResult:
    result = await data_graph.ainvoke({"task": task})

    return SpecialistResult(
        task_id=task_id,
        specialist="data",
        summary=result.get("summary", ""),
        success=result.get("error") is None,
        citations=[],
        details={
            "sql_query": result.get("sql_query", ""),
            "row_count": len(result.get("rows", [])),
            "rows": result.get("rows", [])[:200],
            "error": result.get("error"),
        },
    )
