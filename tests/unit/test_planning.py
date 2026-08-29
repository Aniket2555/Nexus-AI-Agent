import json

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from backend.app.graphs.nodes import planning


def test_parse_plan_extracts_valid_sub_tasks():
    raw = json.dumps(
        {
            "plan": "Research then report.",
            "sub_tasks": [
                {
                    "id": "t1",
                    "description": "Find X",
                    "specialist": "research",
                    "depends_on": [],
                    "target": None,
                }
            ],
        }
    )

    plan, sub_tasks = planning.parse_plan(raw, "fallback request")

    assert plan == "Research then report."
    assert sub_tasks == [
        {
            "id": "t1",
            "description": "Find X",
            "specialist": "research",
            "depends_on": [],
            "target": None,
        }
    ]


def test_parse_plan_strips_markdown_fences():
    raw = "```json\n" + json.dumps({"plan": "p", "sub_tasks": []}) + "\n```"

    plan, sub_tasks = planning.parse_plan(raw, "fallback request")

    assert plan == "p"
    assert sub_tasks == [planning._fallback_task("fallback request")]


def test_parse_plan_falls_back_on_unparseable_output():
    plan, sub_tasks = planning.parse_plan("not json at all", "fallback request")

    assert plan == ""
    assert sub_tasks == [planning._fallback_task("fallback request")]


def test_parse_plan_drops_tasks_with_invalid_specialist():
    raw = json.dumps(
        {
            "plan": "p",
            "sub_tasks": [
                {"id": "t1", "description": "d", "specialist": "not-a-specialist", "depends_on": []}
            ],
        }
    )

    plan, sub_tasks = planning.parse_plan(raw, "fallback request")

    assert sub_tasks == [planning._fallback_task("fallback request")]


def test_parse_plan_drops_report_specialist_even_if_llm_disobeys():
    raw = json.dumps(
        {
            "plan": "p",
            "sub_tasks": [
                {"id": "t1", "description": "research it", "specialist": "research", "depends_on": []},
                {"id": "t2", "description": "write it up", "specialist": "report", "depends_on": ["t1"]},
            ],
        }
    )

    _, sub_tasks = planning.parse_plan(raw, "fallback")

    assert [t["specialist"] for t in sub_tasks] == ["research"]


def test_parse_plan_drops_dependency_on_unknown_task_id():
    raw = json.dumps(
        {
            "plan": "p",
            "sub_tasks": [
                {"id": "t1", "description": "d", "specialist": "research", "depends_on": ["nonexistent"]}
            ],
        }
    )

    _, sub_tasks = planning.parse_plan(raw, "fallback")

    assert sub_tasks[0]["depends_on"] == []


def test_parse_plan_breaks_cycles():
    raw = json.dumps(
        {
            "plan": "p",
            "sub_tasks": [
                {"id": "t1", "description": "d1", "specialist": "research", "depends_on": ["t2"]},
                {"id": "t2", "description": "d2", "specialist": "code", "depends_on": ["t1"]},
            ],
        }
    )

    _, sub_tasks = planning.parse_plan(raw, "fallback")

    assert all(t["depends_on"] == [] for t in sub_tasks)


def test_parse_plan_captures_browser_target():
    raw = json.dumps(
        {
            "plan": "p",
            "sub_tasks": [
                {
                    "id": "t1",
                    "description": "Summarize this page",
                    "specialist": "browser",
                    "depends_on": [],
                    "target": "https://example.com",
                }
            ],
        }
    )

    _, sub_tasks = planning.parse_plan(raw, "fallback")

    assert sub_tasks[0]["target"] == "https://example.com"


async def test_create_plan_uses_the_chat_model_and_parses_its_output():
    raw = json.dumps({"plan": "p", "sub_tasks": []})
    model = GenericFakeChatModel(messages=iter([AIMessage(raw)]))

    plan, sub_tasks = await planning.create_plan("do something", chat_model=model)

    assert plan == "p"
    assert sub_tasks == [planning._fallback_task("do something")]
