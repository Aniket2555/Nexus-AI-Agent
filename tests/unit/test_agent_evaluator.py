from langchain_core.messages import AIMessage

from backend.app.evaluation.agent_evaluator import (
    AgentBenchmarkTask,
    evaluate_agent_task,
    run_agent_benchmark,
)
from backend.app.evaluation.cost_tracker import CostTracker

SUCCESSFUL_STATE = {
    "sub_tasks": [{"id": "t1", "specialist": "research", "description": "find X", "depends_on": []}],
    "agent_results": [{"summary": "X is true.", "success": True}],
    "current_plan": "Find X, then report it.",
    "review_passed": True,
    "iteration_count": 1,
    "final_response": "X is true.",
}


class _RoutingJudge:
    def __init__(self, planning_score="0.9"):
        self._planning_score = planning_score
        self.calls = []

    async def ainvoke(self, prompt):
        self.calls.append(prompt)

        if "grading a task-decomposition plan" in prompt:
            return AIMessage(self._planning_score)
        if "Decompose the following answer" in prompt:
            return AIMessage("1. X is true.")
        if "checking factual claims against a context" in prompt:
            return AIMessage("1: SUPPORTED")

        raise AssertionError(f"unexpected prompt: {prompt[:80]}")


def _invoke_returns(state: dict):
    async def invoke(task: str):
        return state

    return invoke


async def test_evaluate_agent_task_without_a_judge_leaves_judged_metrics_none():
    task = AgentBenchmarkTask(id="t1", task="find X", expected_specialists={"research"})

    result = await evaluate_agent_task(task, _invoke_returns(SUCCESSFUL_STATE))

    assert result.success is True
    assert result.dispatched_specialists == {"research"}
    assert result.tool_call_accuracy == 1.0
    assert result.planning_quality is None
    assert result.hallucination_rate is None
    assert result.iteration_count == 1


async def test_tool_call_accuracy_penalizes_missing_and_extra_specialists():
    task = AgentBenchmarkTask(id="t1", task="find X", expected_specialists={"research", "code"})

    result = await evaluate_agent_task(task, _invoke_returns(SUCCESSFUL_STATE))

    assert result.tool_call_accuracy == 0.5


async def test_judge_populates_planning_quality_and_hallucination_rate():
    task = AgentBenchmarkTask(id="t1", task="find X", expected_specialists={"research"})
    judge = _RoutingJudge()

    result = await evaluate_agent_task(task, _invoke_returns(SUCCESSFUL_STATE), judge=judge)

    assert result.planning_quality == 0.9
    assert result.hallucination_rate == 0.0


async def test_failed_review_marks_the_task_unsuccessful():
    failed_state = {**SUCCESSFUL_STATE, "review_passed": False}
    task = AgentBenchmarkTask(id="t1", task="find X")

    result = await evaluate_agent_task(task, _invoke_returns(failed_state))

    assert result.success is False


async def test_cost_tracker_is_populated_when_a_judge_runs():
    task = AgentBenchmarkTask(id="t1", task="find X", expected_specialists={"research"})
    judge = _RoutingJudge()
    tracker = CostTracker()

    await evaluate_agent_task(
        task, _invoke_returns(SUCCESSFUL_STATE), judge=judge, cost_tracker=tracker
    )

    assert tracker.call_count == len(judge.calls)
    assert "agent_evaluator" in tracker.cost_by_label


async def test_run_agent_benchmark_aggregates_across_tasks():
    tasks = [
        AgentBenchmarkTask(id="t1", task="find X", expected_specialists={"research"}),
        AgentBenchmarkTask(id="t2", task="find Y", expected_specialists={"research"}),
    ]

    summary = await run_agent_benchmark(tasks, _invoke_returns(SUCCESSFUL_STATE))

    assert summary.task_success_rate == 1.0
    assert summary.mean_tool_call_accuracy == 1.0
    assert summary.mean_loop_iterations == 1.0
    assert summary.mean_planning_quality is None
    assert len(summary.results) == 2


async def test_run_agent_benchmark_handles_a_mixed_success_batch():
    async def invoke(task: str):
        if task == "fails":
            return {**SUCCESSFUL_STATE, "review_passed": False}
        return SUCCESSFUL_STATE

    tasks = [
        AgentBenchmarkTask(id="t1", task="ok"),
        AgentBenchmarkTask(id="t2", task="fails"),
    ]

    summary = await run_agent_benchmark(tasks, invoke)

    assert summary.task_success_rate == 0.5
