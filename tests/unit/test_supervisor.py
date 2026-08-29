from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from backend.app.graphs import supervisor as supervisor_module
from backend.app.graphs.states.shared_state import SpecialistResult, SubTask

RESEARCH_TASK = SubTask(
    id="t1", description="Find X", specialist="research", depends_on=[], target=None
)

CODE_TASK = SubTask(
    id="t2", description="Write Y", specialist="code", depends_on=["t1"], target=None
)

RESEARCH_RESULT = SpecialistResult(
    task_id="t1",
    specialist="research",
    summary="X is true.",
    success=True,
    citations=[],
    details={},
)

CODE_RESULT = SpecialistResult(
    task_id="t2",
    specialist="code",
    summary="def y(): ...",
    success=True,
    citations=[],
    details={},
)


async def test_dispatch_node_runs_only_ready_tasks_concurrently(monkeypatch):
    calls = []

    async def fake_run_research(task_id, description, tenant_id, user_id):
        calls.append(("research", task_id, tenant_id, user_id))
        return RESEARCH_RESULT

    async def fake_run_code(task_id, description):
        calls.append(("code", task_id))
        return CODE_RESULT

    monkeypatch.setattr(supervisor_module, "run_research", fake_run_research)
    monkeypatch.setattr(supervisor_module, "run_code", fake_run_code)

    state = {"sub_tasks": [RESEARCH_TASK, CODE_TASK], "agent_results": []}
    config = {"configurable": {"tenant_id": "acme", "user_id": "u1"}}

    result = await supervisor_module.dispatch_node(state, config)

    assert len(result["agent_results"]) == 1
    assert result["agent_results"][0]["task_id"] == "t1"
    assert calls == [("research", "t1", "acme", "u1")]


async def test_dispatch_node_returns_nothing_when_no_task_is_ready():
    state = {"sub_tasks": [CODE_TASK], "agent_results": []}

    result = await supervisor_module.dispatch_node(state, {"configurable": {}})

    assert result == {}


def test_dispatch_edge_loops_while_ready_tasks_remain():
    state = {"sub_tasks": [RESEARCH_TASK, CODE_TASK], "agent_results": [RESEARCH_RESULT]}

    assert supervisor_module.dispatch_edge(state) == "dispatch"


def test_dispatch_edge_moves_to_coordinator_once_complete():
    state = {
        "sub_tasks": [RESEARCH_TASK, CODE_TASK],
        "agent_results": [RESEARCH_RESULT, CODE_RESULT],
    }

    assert supervisor_module.dispatch_edge(state) == "coordinator"


def test_dispatch_edge_moves_to_coordinator_on_unresolvable_dependency():
    stuck_task = SubTask(
        id="t2", description="d", specialist="code", depends_on=["never-runs"], target=None
    )
    state = {"sub_tasks": [stuck_task], "agent_results": []}

    assert supervisor_module.dispatch_edge(state) == "coordinator"


async def test_coordinator_node_calls_report_with_original_request(monkeypatch):
    captured = {}

    async def fake_run_report(agent_name, request, agent_results):
        captured["request"] = request
        return {"summary": "Final draft.", "success": True, "citations": [], "details": {}}

    monkeypatch.setattr(supervisor_module, "run_report", fake_run_report)

    state = {
        "messages": [HumanMessage("Please research X")],
        "agent_results": [RESEARCH_RESULT],
        "review_feedback": "",
    }

    result = await supervisor_module.coordinator_node(state, {})

    assert result["draft_response"] == "Final draft."
    assert captured["request"] == "Please research X"


async def test_coordinator_node_appends_revision_note_on_retry(monkeypatch):
    captured = {}

    async def fake_run_report(agent_name, request, agent_results):
        captured["request"] = request
        return {"summary": "Revised draft.", "success": True, "citations": [], "details": {}}

    monkeypatch.setattr(supervisor_module, "run_report", fake_run_report)

    state = {
        "messages": [HumanMessage("Please research X")],
        "agent_results": [RESEARCH_RESULT],
        "review_feedback": "Claim about X is unsupported.",
    }

    await supervisor_module.coordinator_node(state, {})

    assert "Please research X" in captured["request"]
    assert "Claim about X is unsupported." in captured["request"]


async def test_reviewer_node_accumulates_cost(monkeypatch):
    async def fake_review_draft(draft_response, agent_results):
        return (True, "Well grounded.", 0.001)

    monkeypatch.setattr(supervisor_module, "review_draft", fake_review_draft)

    state = {"draft_response": "Final draft.", "agent_results": [], "cost_usd_spent": 0.002}

    result = await supervisor_module.reviewer_node(state, {})

    assert result["review_passed"] is True
    assert result["review_feedback"] == "Well grounded."
    assert result["cost_usd_spent"] == 0.003


def test_reflect_node_maps_pass_to_full_quality_score():
    state = {"review_passed": True, "review_feedback": "ok", "iteration_count": 0}

    result = supervisor_module.reflect_node(state)

    assert result == {"quality_score": 1.0, "reflection_feedback": "ok", "iteration_count": 1}


def test_reflect_node_maps_fail_to_zero_quality_score():
    state = {"review_passed": False, "review_feedback": "bad", "iteration_count": 1}

    result = supervisor_module.reflect_node(state)

    assert result == {"quality_score": 0.0, "reflection_feedback": "bad", "iteration_count": 2}


def test_reflect_edge_finalizes_when_review_passed():
    state = {"review_passed": True, "iteration_count": 1, "quality_score": 1.0, "cost_usd_spent": 0.0}

    assert supervisor_module.reflect_edge(state) == "finalize"


def test_reflect_edge_retries_when_budget_remains():
    state = {"review_passed": False, "iteration_count": 1, "quality_score": 0.0, "cost_usd_spent": 0.0}

    assert supervisor_module.reflect_edge(state) == "retry"


def test_reflect_edge_escalates_once_iteration_cap_hit():
    from backend.app.config import get_settings

    state = {
        "review_passed": False,
        "iteration_count": get_settings().reflection_max_iterations,
        "quality_score": 0.0,
        "cost_usd_spent": 0.0,
    }

    assert supervisor_module.reflect_edge(state) == "escalate"


def test_escalate_node_uses_draft_when_human_approves(monkeypatch):
    monkeypatch.setattr(supervisor_module, "interrupt", lambda _: {"approved": True})

    state = {"draft_response": "Best effort draft.", "review_feedback": "iffy"}

    result = supervisor_module.escalate_node(state)

    assert result["final_response"] == "Best effort draft."


def test_escalate_node_declines_gracefully_when_human_rejects(monkeypatch):
    monkeypatch.setattr(supervisor_module, "interrupt", lambda _: {"approved": False})

    state = {"draft_response": "Best effort draft.", "review_feedback": "iffy"}

    result = supervisor_module.escalate_node(state)

    assert "Best effort draft." not in result["final_response"]


def test_finalize_node_appends_ai_message():
    state = {"final_response": "The answer."}

    result = supervisor_module.finalize_node(state)

    assert result["final_response"] == "The answer."
    assert result["messages"] == [AIMessage(content="The answer.")]


async def test_supervisor_graph_end_to_end_happy_path(monkeypatch):
    """Full compiled graph: planner decomposes into one research task, dispatch
    runs it, coordinator synthesizes, reviewer passes immediately, finalize appends
    the AIMessage. No specialist here talks to a real network/LLM boundary that
    isn't already stubbed via monkeypatch on the supervisor module's own imports.
    """

    async def fake_create_plan(request):
        return "Answer the question.", [RESEARCH_TASK]

    async def fake_run_research(task_id, description, tenant_id, user_id):
        return RESEARCH_RESULT

    async def fake_run_report(agent_name, request, agent_results):
        return {
            "summary": "X is true, per research.",
            "success": True,
            "citations": [],
            "details": {},
        }

    async def fake_review_draft(draft_response, agent_results):
        return (True, "Grounded.", 0.0)

    monkeypatch.setattr(supervisor_module, "create_plan", fake_create_plan)
    monkeypatch.setattr(supervisor_module, "run_research", fake_run_research)
    monkeypatch.setattr(supervisor_module, "run_report", fake_run_report)
    monkeypatch.setattr(supervisor_module, "review_draft", fake_review_draft)

    result = await supervisor_module.supervisor_graph.ainvoke(
        {"messages": [HumanMessage("Is X true?")]},
        {"configurable": {"tenant_id": "acme", "user_id": "u1"}},
    )

    assert result["final_response"] == "X is true, per research."
    assert result["review_passed"] is True
    assert result["messages"][-1].content == "X is true, per research."


async def test_supervisor_graph_retries_coordinator_on_review_failure(monkeypatch):
    review_calls = {"n": 0}

    async def fake_create_plan(request):
        return "Answer the question.", [RESEARCH_TASK]

    async def fake_run_research(task_id, description, tenant_id, user_id):
        return RESEARCH_RESULT

    async def fake_run_report(agent_name, request, agent_results):
        if "Revision needed" in request:
            return {
                "summary": "Revised, grounded answer.",
                "success": True,
                "citations": [],
                "details": {},
            }
        return {"summary": "First draft.", "success": True, "citations": [], "details": {}}

    async def fake_review_draft(draft_response, agent_results):
        review_calls["n"] += 1
        if review_calls["n"] == 1:
            return (False, "Missing citation.", 0.0)
        return (True, "Grounded now.", 0.0)

    monkeypatch.setattr(supervisor_module, "create_plan", fake_create_plan)
    monkeypatch.setattr(supervisor_module, "run_research", fake_run_research)
    monkeypatch.setattr(supervisor_module, "run_report", fake_run_report)
    monkeypatch.setattr(supervisor_module, "review_draft", fake_review_draft)

    result = await supervisor_module.supervisor_graph.ainvoke(
        {"messages": [HumanMessage("Is X true?")]},
        {"configurable": {"tenant_id": "acme", "user_id": "u1"}},
    )

    assert review_calls["n"] == 2
    assert result["final_response"] == "Revised, grounded answer."


async def test_supervisor_graph_escalates_and_resumes_via_real_interrupt(monkeypatch):
    """The compiled `supervisor_graph` (module-level, no checkpointer — Platform
    injects its own, per rag_chat.py's D6 precedent) can't actually pause/resume
    outside Platform. So this test compiles a second instance from the same
    `builder` with an explicit `InMemorySaver`, exactly like `human_approval.py`'s
    own standalone carve-out, to exercise the real interrupt()/Command(resume=...)
    round trip end to end rather than just unit-testing escalate_node in isolation.
    """
    from backend.app.config import get_settings

    async def fake_create_plan(request):
        return "Answer the question.", [RESEARCH_TASK]

    async def fake_run_research(task_id, description, tenant_id, user_id):
        return RESEARCH_RESULT

    async def fake_run_report(agent_name, request, agent_results):
        return {
            "summary": "Best-effort draft.",
            "success": True,
            "citations": [],
            "details": {},
        }

    async def fake_review_draft(draft_response, agent_results):
        return (False, "Never satisfied.", 0.0)

    monkeypatch.setattr(supervisor_module, "create_plan", fake_create_plan)
    monkeypatch.setattr(supervisor_module, "run_research", fake_run_research)
    monkeypatch.setattr(supervisor_module, "run_report", fake_run_report)
    monkeypatch.setattr(supervisor_module, "review_draft", fake_review_draft)

    graph = supervisor_module.builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "escalate-1", "tenant_id": "acme", "user_id": "u1"}}

    result = await graph.ainvoke({"messages": [HumanMessage("Is X true?")]}, config)

    assert "__interrupt__" in result
    assert result["iteration_count"] == get_settings().reflection_max_iterations
    payload = result["__interrupt__"][0].value
    assert payload["draft"] == "Best-effort draft."

    resumed = await graph.ainvoke(Command(resume={"approved": True}), config)

    assert "__interrupt__" not in resumed
    assert resumed["final_response"] == "Best-effort draft."
