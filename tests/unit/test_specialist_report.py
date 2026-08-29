from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from backend.app.graphs.specialists import report as report_module
from backend.app.graphs.states.shared_state import SpecialistResult

RESULTS = [
    SpecialistResult(
        task_id="t1",
        specialist="research",
        summary="NEXUS uses Groq for chat.",
        success=True,
        citations=[{"source": "handbook.pdf", "page": 3, "chunk_id": "x"}],
        details={},
    ),
    SpecialistResult(
        task_id="t2",
        specialist="code",
        summary="Generated an add function.",
        success=False,
        citations=[],
        details={"syntax_error": "unexpected EOF"},
    ),
]


def test_format_results_includes_status_and_citations():
    formatted = report_module.format_results(RESULTS)

    assert "research (succeeded)" in formatted
    assert "code (FAILED)" in formatted
    assert "[Source: handbook.pdf, Page 3]" in formatted


def test_format_results_handles_no_results():
    assert "no specialist results" in report_module.format_results([])


async def test_synthesize_node_wraps_results_in_the_untrusted_content_boundary(monkeypatch):
    model = GenericFakeChatModel(messages=iter([AIMessage("Final synthesized report.")]))
    monkeypatch.setattr(report_module, "get_chat_model", lambda: model)

    result = await report_module.synthesize_node({"request": "summarize", "agent_results": RESULTS})

    assert result["report"] == "Final synthesized report."


async def test_run_report_aggregates_citations_from_every_specialist(monkeypatch):
    model = GenericFakeChatModel(messages=iter([AIMessage("Combined report text.")]))
    monkeypatch.setattr(report_module, "get_chat_model", lambda: model)

    result = await report_module.run_report("task-final", "summarize everything", RESULTS)

    assert result["specialist"] == "report"
    assert result["summary"] == "Combined report text."
    assert result["success"] is True
    assert len(result["citations"]) == 1
    assert result["details"]["synthesized_from"] == ["research", "code"]
