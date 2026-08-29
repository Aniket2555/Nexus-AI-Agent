"""Unit-level: verifies benchmarks.py's own wiring (report composition,
GROQ_API_KEY gating) with every infra-touching piece mocked out. The retrieval
half (`run_eval.py`) and agent half (`agent_evaluator.py`) each have their own
test coverage already — this only tests that `benchmarks.run()` combines them
correctly, not their internals again.
"""

from backend.app.evaluation import benchmarks as benchmarks_module
from backend.app.evaluation.agent_evaluator import AgentBenchmarkResult, AgentBenchmarkSummary

FAKE_RAG_REPORT = {"mean_context_precision": 0.9, "mean_context_recall": 0.8}


async def test_run_skips_agent_benchmark_without_a_groq_api_key(monkeypatch, tmp_path):
    async def fake_retrieval_eval(report_path, baseline_path):
        return FAKE_RAG_REPORT

    monkeypatch.setattr(benchmarks_module, "run_retrieval_eval", fake_retrieval_eval)
    monkeypatch.setattr(benchmarks_module, "get_settings", lambda: _FakeSettings(groq_api_key=""))

    report_path = tmp_path / "report.json"

    combined = await benchmarks_module.run(report_path, None)

    assert combined["retrieval"] == FAKE_RAG_REPORT
    assert combined["agent"] is None
    assert report_path.exists()


async def test_run_includes_agent_benchmark_when_a_groq_api_key_is_set(monkeypatch, tmp_path):
    async def fake_retrieval_eval(report_path, baseline_path):
        return FAKE_RAG_REPORT

    fake_summary = AgentBenchmarkSummary(
        task_success_rate=1.0,
        mean_tool_call_accuracy=1.0,
        mean_planning_quality=0.9,
        mean_loop_iterations=1.0,
        mean_hallucination_rate=0.0,
        results=[
            AgentBenchmarkResult(
                task_id="a01",
                success=True,
                dispatched_specialists={"research"},
                tool_call_accuracy=1.0,
                planning_quality=0.9,
                iteration_count=1,
                hallucination_rate=0.0,
                final_response="done",
            )
        ],
    )

    async def fake_run_agent_benchmark(*args, **kwargs):
        return fake_summary

    monkeypatch.setattr(benchmarks_module, "run_retrieval_eval", fake_retrieval_eval)
    monkeypatch.setattr(benchmarks_module, "get_settings", lambda: _FakeSettings(groq_api_key="fake-key"))
    monkeypatch.setattr(benchmarks_module, "get_chat_model", lambda: object())
    monkeypatch.setattr(benchmarks_module, "load_agent_benchmark_tasks", lambda: ["placeholder-task"])
    monkeypatch.setattr(benchmarks_module, "run_agent_benchmark", fake_run_agent_benchmark)

    report_path = tmp_path / "report.json"

    combined = await benchmarks_module.run(report_path, None)

    assert combined["agent"]["task_success_rate"] == 1.0
    assert combined["agent"]["results"][0]["task_id"] == "a01"
    assert "cost" in combined


class _FakeSettings:
    def __init__(self, groq_api_key: str) -> None:
        self.groq_api_key = groq_api_key
