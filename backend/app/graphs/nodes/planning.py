import json
import logging
import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage

from backend.app.graphs.states.shared_state import SPECIALIST_TYPES, SubTask
from backend.app.llm.provider import get_chat_model
from backend.app.utils.messages import as_text

logger = logging.getLogger(__name__)

PLANNING_PROMPT = (
    "You are the planner in a multi-agent system. Break the user's request into "
    "a small number of sub-tasks (as few as necessary — a simple request needs "
    "only one), each assigned to exactly one specialist.\n\n"
    "Available specialists:\n"
    "- research: retrieval + synthesis over documents, memory, and the knowledge graph\n"
    "- code: writes code (does not execute it — no sandbox exists yet)\n"
    "- data: SQL analysis over the project database\n"
    '- browser: fetches and summarizes one web page — set "target" to the exact URL\n'
    '- vision: analyzes one image — set "target" to the exact image URL\n\n'
    'Do NOT plan a "report" task yourself — the supervisor always synthesizes the final\n'
    "answer from your other sub-tasks' results after they finish, automatically.\n\n"
    "Respond with ONLY valid JSON, no markdown fences, no commentary, in exactly this shape:\n"
    "{{\n"
    '  "plan": "<one sentence restating the goal and your approach>",\n'
    '  "sub_tasks": [\n'
    '    {{"id": "t1", "description": "...", "specialist": "research", "depends_on": [],\n'
    '      "target": null}},\n'
    '    {{"id": "t2", "description": "...", "specialist": "browser", "depends_on": ["t1"],\n'
    '      "target": "https://example.com"}}\n'
    "  ]\n"
    "}}\n\n"
    "User request: {request}\n"
)

_CODE_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

_DISPATCHABLE_SPECIALISTS = tuple(s for s in SPECIALIST_TYPES if s != "report")


def _fallback_task(request: str) -> SubTask:
    """A single research task covering the whole request — the safe default when
    the planner's output can't be parsed into a real plan at all. Research is the
    broadest specialist (retrieval + synthesis), the closest single substitute for
    "just try to answer the request" when decomposition itself failed.
    """
    return SubTask(
        id="t1", description=request, specialist="research", depends_on=[], target=None
    )


def _has_cycle(sub_tasks: list[SubTask]) -> bool:
    """DFS cycle detection. A cyclic dependency would mean no task in the cycle
    ever becomes "ready" (§6.2's dispatch requires every dependency to already be
    complete) — dispatch would stall forever without this check.
    """
    graph = {t["id"]: t["depends_on"] for t in sub_tasks}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> bool:
        if node_id in visited:
            return False
        if node_id in visiting:
            return True
        visiting.add(node_id)
        for dep_id in graph.get(node_id, ()):
            if dep_id in graph and visit(dep_id):
                return True
        visiting.discard(node_id)
        visited.add(node_id)
        return False

    return any(visit(task["id"]) for task in sub_tasks)


def parse_plan(raw: str, fallback_request: str) -> tuple[str, list[SubTask]]:
    """Defensive parsing, same posture as knowledge_graph.py's extraction (§4.2):
    drop what doesn't fit the allowed vocabulary rather than raising, and fall
    back to a safe single-task plan — never a crashed supervisor — if the
    response can't be parsed into anything usable at all.
    """
    text = _CODE_FENCE.sub("", raw.strip())
    try:
        data = json.loads(text)
        plan = str(data.get("plan", ""))
        raw_tasks = data.get("sub_tasks", [])
        if not isinstance(raw_tasks, list):
            raise ValueError("sub_tasks is not a list")
    except (json.JSONDecodeError, ValueError, AttributeError):
        logger.warning("Planner returned unparseable output: %r", raw[:200])
        return "", [_fallback_task(fallback_request)]

    valid_ids = {
        item.get("id") for item in raw_tasks if isinstance(item, dict) and item.get("id")
    }

    sub_tasks: list[SubTask] = []
    for item in raw_tasks:
        if not isinstance(item, dict):
            continue
        task_id = item.get("id")
        specialist = item.get("specialist")
        description = item.get("description")
        if not (task_id and description and specialist in _DISPATCHABLE_SPECIALISTS):
            continue
        depends_on = [
            d for d in item.get("depends_on", []) if d in valid_ids and d != task_id
        ]
        target = item.get("target")
        sub_tasks.append(
            SubTask(
                id=task_id,
                description=description,
                specialist=specialist,
                depends_on=depends_on,
                target=target if isinstance(target, str) else None,
            )
        )

    if not sub_tasks:
        return plan, [_fallback_task(fallback_request)]

    if _has_cycle(sub_tasks):
        logger.warning("Planner produced a cyclic dependency graph; dropping all depends_on.")
        sub_tasks = [SubTask(**{**task, "depends_on": []}) for task in sub_tasks]

    return plan, sub_tasks


async def create_plan(
    request: str, *, chat_model: BaseChatModel | None = None
) -> tuple[str, list[SubTask]]:
    model = chat_model or get_chat_model(temperature=0)
    response = await model.ainvoke(
        [SystemMessage(content=PLANNING_PROMPT.format(request=request))]
    )
    return parse_plan(as_text(response), request)
