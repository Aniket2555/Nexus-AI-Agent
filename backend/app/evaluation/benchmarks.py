"""Combined benchmark runner (§8.5):

    python -m backend.app.evaluation.benchmarks [--report PATH] [--baseline PATH]

Runs the retrieval eval harness (`run_eval.py`, §2.7 — requires Qdrant and
Elasticsearch reachable) and, if `GROQ_API_KEY` is set, the agent benchmark
suite (`agent_evaluator.py`) against the real, compiled `supervisor_graph`
(Phase 6). Agent benchmarking is skipped, not failed, without a key — same
skip-not-fail posture `retrieval_metrics`/`faithfulness.py` already established
for judge-dependent metrics, since a supervisor run needs a real chat model at
every specialist, not just at one judge call.
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from langchain_core.messages import HumanMessage

from backend.app.config import get_settings
from backend.app.evaluation.agent_evaluator import (
    AgentBenchmarkSummary,
    load_agent_benchmark_tasks,
    run_agent_benchmark,
)
from backend.app.evaluation.cost_tracker import CostTracker
from backend.app.evaluation.run_eval import run as run_retrieval_eval
from backend.app.llm.provider import get_chat_model
from backend.app.observability.metrics import (
    apush_metrics,
    rag_context_precision,
    rag_context_recall,
    rag_faithfulness,
)

if TYPE_CHECKING:
    from backend.app.graphs.states.supervisor_state import SupervisorState


async def _invoke_supervisor(task: str) -> dict[str, Any]:
    from backend.app.graphs.supervisor import supervisor_graph

    initial_state = cast(
        "SupervisorState", {"messages": [HumanMessage(task)]}
    )
    result = await supervisor_graph.ainvoke(
        initial_state, config={"configurable": {"tenant_id": "eval", "user_id": "eval"}}
    )
    return dict(result)


def _summary_to_dict(summary: AgentBenchmarkSummary) -> dict[str, Any]:
    return {
        "task_success_rate": round(summary.task_success_rate, 4),
        "mean_tool_call_accuracy": round(summary.mean_tool_call_accuracy, 4),
        "mean_planning_quality": (
            round(summary.mean_planning_quality, 4)
            if summary.mean_planning_quality is not None
            else None
        ),
        "mean_loop_iterations": round(summary.mean_loop_iterations, 4),
        "mean_hallucination_rate": (
            round(summary.mean_hallucination_rate, 4)
            if summary.mean_hallucination_rate is not None
            else None
        ),
        "results": [
            {
                "task_id": r.task_id,
                "success": r.success,
                "dispatched_specialists": sorted(r.dispatched_specialists),
                "tool_call_accuracy": round(r.tool_call_accuracy, 4),
                "planning_quality": r.planning_quality,
                "iteration_count": r.iteration_count,
                "hallucination_rate": r.hallucination_rate,
            }
            for r in summary.results
        ],
    }


async def run(report_path: Path, baseline_path: Path | None = None) -> dict[str, Any]:
    settings = get_settings()
    cost_tracker = CostTracker()

    rag_report_path = report_path.with_stem(report_path.stem + "_rag")
    rag_report = await run_retrieval_eval(rag_report_path, baseline_path)

    for result in rag_report.get("results", []):
        rag_context_precision.observe(result["context_precision"])
        rag_context_recall.observe(result["context_recall"])
        if result.get("faithfulness") is not None:
            rag_faithfulness.observe(result["faithfulness"])
    await apush_metrics("nexus_rag_eval")

    agent_report = None
    if settings.groq_api_key:
        judge = get_chat_model()
        tasks = load_agent_benchmark_tasks()
        summary = await run_agent_benchmark(
            tasks, _invoke_supervisor, judge=judge, cost_tracker=cost_tracker
        )
        agent_report = _summary_to_dict(summary)
    else:
        print("GROQ_API_KEY not set — skipping the agent benchmark suite.")

    combined = {"retrieval": rag_report, "agent": agent_report, "cost": cost_tracker.summary()}

    report_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(f"Wrote {report_path}")

    if agent_report is not None:
        print(
            f"  agent task_success_rate={agent_report['task_success_rate']}"
            f" tool_call_accuracy={agent_report['mean_tool_call_accuracy']}"
            f" loop_iterations={agent_report['mean_loop_iterations']}"
        )
    print(f"  total eval cost: ${cost_tracker.total_cost_usd:.6f}")

    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("benchmark_report.json"))
    parser.add_argument("--baseline", type=Path, default=Path("tests/eval/baseline.json"))
    args = parser.parse_args()

    asyncio.run(run(args.report, args.baseline))


if __name__ == "__main__":
    main()
