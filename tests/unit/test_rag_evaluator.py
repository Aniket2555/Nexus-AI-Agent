from langchain_core.messages import AIMessage

from backend.app.evaluation.cost_tracker import CostTracker
from backend.app.evaluation.rag_evaluator import evaluate_rag_response


class _RoutingJudge:
    """Inspects the prompt to decide which canned response to return, rather
    than relying on call order — evaluate_rag_response calls the judge for
    several different metrics, and pattern-matching on each prompt's own
    wording (from faithfulness.py/hallucination.py's *_PROMPT templates) is
    more robust than hard-coding an exact sequence.
    """

    def __init__(self, relevance="0.8", faithfulness="0.9", correctness="0.7"):
        self._relevance = relevance
        self._faithfulness = faithfulness
        self._correctness = correctness
        self.call_count = 0

    async def ainvoke(self, prompt):
        self.call_count += 1
        text = prompt if isinstance(prompt, str) else str(prompt)

        if "grading whether retrieved context is relevant" in text:
            return AIMessage(self._relevance)
        if "grading whether an answer is faithful" in text:
            return AIMessage(self._faithfulness)
        if "grading whether a generated answer matches" in text:
            return AIMessage(self._correctness)
        if "Decompose the following answer" in text:
            return AIMessage("1. A claim.")
        if "checking factual claims against a context" in text:
            return AIMessage("1: SUPPORTED")

        raise AssertionError(f"unexpected prompt: {text[:80]}")


async def test_no_judge_and_no_golden_set_leaves_everything_unmeasured():
    result = await evaluate_rag_response("q", "a", "c", judge=None)

    assert result.context_precision is None
    assert result.context_recall is None
    assert result.relevance is None
    assert result.faithfulness is None
    assert result.groundedness is None
    assert result.answer_correctness is None


async def test_golden_set_ids_populate_context_precision_and_recall_without_a_judge():
    result = await evaluate_rag_response(
        "q", "a", "c",
        retrieved_chunk_ids=["x1", "x2"],
        expected_chunk_ids=["x1"],
        judge=None,
    )

    assert result.context_precision == 0.5
    assert result.context_recall == 1.0


async def test_judge_populates_relevance_faithfulness_and_groundedness():
    judge = _RoutingJudge()

    result = await evaluate_rag_response("q", "a", "c", judge=judge)

    assert result.relevance == 0.8
    assert result.faithfulness == 0.9
    assert result.groundedness == 1.0


async def test_reference_answer_populates_answer_correctness():
    judge = _RoutingJudge()

    result = await evaluate_rag_response("q", "a", "c", reference_answer="ref", judge=judge)

    assert result.answer_correctness == 0.7


async def test_cost_tracker_is_populated_when_a_judge_runs():
    judge = _RoutingJudge()
    tracker = CostTracker()

    await evaluate_rag_response("q", "a", "c", judge=judge, cost_tracker=tracker)

    assert tracker.call_count == judge.call_count
    assert tracker.total_cost_usd > 0
    assert "rag_evaluator" in tracker.cost_by_label
