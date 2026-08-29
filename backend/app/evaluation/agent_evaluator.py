from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from langchain_core.language_models.chat_models import BaseChatModel

from backend.app.evaluation.cost_tracker import CostTracker, track_chat_model
from backend.app.evaluation.hallucination import hallucination_rate
from backend.app.evaluation.judging import judge_score
from backend.app.graphs.states.shared_state import SpecialistType

DEFAULT_AGENT_BENCHMARK_PATH = Path("tests/eval/fixtures/agent_benchmark.yaml")

PLANNING_QUALITY_JUDGE_PROMPT = (
    "You are grading a task-decomposition plan produced by an AI\n"
    "planner.\n\nTask: {task}\n\nPlan reasoning: {plan}\n\n"
    "Sub-tasks assigned (specialist: description): {sub_tasks}\n\n"
    "On a single line, output only a number between 0 and 1: how reasonable "
    "this decomposition is —\nright specialists for the work, sensible "
    "granularity (not too coarse, not needlessly split),\nno obviously missing "
    "step. Output nothing else.\n"
)

SupervisorInvoker = Callable[[str], Awaitable[dict[str, Any]]]


@dataclass
class AgentBenchmarkTask:
    id: str
    task: str
    expected_specialists: set[SpecialistType] = field(default_factory=set)


@dataclass
class AgentBenchmarkResult:
    task_id: str
    success: bool
    dispatched_specialists: set[SpecialistType]
    tool_call_accuracy: float
    planning_quality: float | None
    iteration_count: int
    hallucination_rate: float | None
    final_response: str


def load_agent_benchmark_tasks(
    path: Path = DEFAULT_AGENT_BENCHMARK_PATH,
) -> list[AgentBenchmarkTask]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [
        AgentBenchmarkTask(
            id=item["id"],
            task=item["task"],
            expected_specialists=set(item.get("expected_specialists", [])),
        )
        for item in data
    ]


def _tool_call_accuracy(actual: set[SpecialistType], expected: set[SpecialistType]) -> float:
    """Jaccard overlap between dispatched and expected specialists — penalizes
    both missing a needed specialist and dispatching an unnecessary one,
    symmetric rather than a one-sided precision or recall number.
    """
    union = actual | expected
    if not union:
        return 1.0
    return len(actual & expected) / len(union)


async def evaluate_agent_task(
    task: AgentBenchmarkTask,
    invoke_supervisor: SupervisorInvoker,
    *,
    judge: BaseChatModel | None = None,
    cost_tracker: CostTracker | None = None,
) -> AgentBenchmarkResult:
    """Run one benchmark task through the supervisor graph (via
    `invoke_supervisor`) and score it against §8.5's agent metric table.
    """
    state = await invoke_supervisor(task.task)

    sub_tasks = state.get("sub_tasks", [])
    dispatched = {t["specialist"] for t in sub_tasks}
    accuracy = _tool_call_accuracy(dispatched, task.expected_specialists)

    tracked_judge = judge
    if judge is not None and cost_tracker is not None:
        tracked_judge = track_chat_model(judge, cost_tracker, label="agent_evaluator")

    planning_score = None
    if tracked_judge is not None:
        sub_tasks_desc = "; ".join(f"{t['specialist']}: {t['description']}" for t in sub_tasks)
        planning_score = await judge_score(
            PLANNING_QUALITY_JUDGE_PROMPT.format(
                task=task.task,
                plan=state.get("current_plan") or "",
                sub_tasks=sub_tasks_desc or "(none)",
            ),
            tracked_judge,
        )

    final_response = state.get("final_response") or ""
    agent_context = "\n".join(r.get("summary", "") for r in state.get("agent_results", []))
    rate = await hallucination_rate(final_response, agent_context, tracked_judge)

    return AgentBenchmarkResult(
        task_id=task.id,
        success=bool(state.get("review_passed")),
        dispatched_specialists=dispatched,
        tool_call_accuracy=accuracy,
        planning_quality=planning_score,
        iteration_count=state.get("iteration_count", 0),
        hallucination_rate=rate,
        final_response=final_response,
    )


@dataclass
class AgentBenchmarkSummary:
    task_success_rate: float
    mean_tool_call_accuracy: float
    mean_planning_quality: float | None
    mean_loop_iterations: float
    mean_hallucination_rate: float | None
    results: list[AgentBenchmarkResult]


async def run_agent_benchmark(
    tasks: list[AgentBenchmarkTask],
    invoke_supervisor: SupervisorInvoker,
    *,
    judge: BaseChatModel | None = None,
    cost_tracker: CostTracker | None = None,
) -> AgentBenchmarkSummary:
    results = [
        await evaluate_agent_task(t, invoke_supervisor, judge=judge, cost_tracker=cost_tracker)
        for t in tasks
    ]

    planning_scores = [r.planning_quality for r in results if r.planning_quality is not None]
    hallucination_rates = [
        r.hallucination_rate for r in results if r.hallucination_rate is not None
    ]

    return AgentBenchmarkSummary(
        task_success_rate=(
            sum(1 for r in results if r.success) / len(results) if results else 0.0
        ),
        mean_tool_call_accuracy=(
            sum(r.tool_call_accuracy for r in results) / len(results) if results else 0.0
        ),
        mean_planning_quality=(
            sum(planning_scores) / len(planning_scores) if planning_scores else None
        ),
        mean_loop_iterations=(
            sum(r.iteration_count for r in results) / len(results) if results else 0.0
        ),
        mean_hallucination_rate=(
            sum(hallucination_rates) / len(hallucination_rates) if hallucination_rates else None
        ),
        results=results,
    )
