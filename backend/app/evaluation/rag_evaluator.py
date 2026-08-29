from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel

from backend.app.evaluation import retrieval_metrics
from backend.app.evaluation.cost_tracker import CostTracker, track_chat_model
from backend.app.evaluation.faithfulness import answer_correctness, faithfulness, relevance
from backend.app.evaluation.hallucination import groundedness


@dataclass
class RAGEvaluationResult:
    """§8.5's RAG metric table, for one (question, answer, context) instance.
    Every field is `float | None`: `None` means "not measured" — either no judge
    was available, or the metric's precondition wasn't met (see each field's
    corresponding function's own docstring for why) — not "measured and zero."
    """

    question: str
    context_precision: float | None = None
    context_recall: float | None = None
    relevance: float | None = None
    faithfulness: float | None = None
    groundedness: float | None = None
    answer_correctness: float | None = None


async def evaluate_rag_response(
    question: str,
    answer: str,
    context: str,
    *,
    retrieved_chunk_ids: list[str] | None = None,
    expected_chunk_ids: list[str] | None = None,
    reference_answer: str | None = None,
    judge: BaseChatModel | None = None,
    cost_tracker: CostTracker | None = None,
) -> RAGEvaluationResult:
    """Score one RAG turn against every §8.5 metric its inputs support.

    `context_precision`/`context_recall` (retrieval_metrics.py, §2.7) only run
    when both `retrieved_chunk_ids` and `expected_chunk_ids` are given — they need
    a golden set, unlike the three judge-based metrics below, which work on any
    live (question, answer, context) with no ground truth at all.
    `answer_correctness` additionally needs `reference_answer`, which most of this
    project's golden set (§2.7's `golden_set.yaml`) doesn't carry — it only has
    `expected_chunk_ids`, so this is `None` for those unless the caller supplies
    one.
    """
    context_precision_score = None
    context_recall_score = None
    if retrieved_chunk_ids is not None and expected_chunk_ids is not None:
        context_precision_score = retrieval_metrics.context_precision(
            retrieved_chunk_ids, expected_chunk_ids
        )
        context_recall_score = retrieval_metrics.context_recall(
            retrieved_chunk_ids, expected_chunk_ids
        )

    tracked_judge = judge
    if judge is not None and cost_tracker is not None:
        tracked_judge = track_chat_model(judge, cost_tracker, label="rag_evaluator")

    correctness_score = None
    if reference_answer is not None:
        correctness_score = await answer_correctness(answer, reference_answer, tracked_judge)

    return RAGEvaluationResult(
        question,
        context_precision=context_precision_score,
        context_recall=context_recall_score,
        relevance=await relevance(question, context, tracked_judge),
        faithfulness=await faithfulness(answer, context, tracked_judge),
        groundedness=await groundedness(answer, context, tracked_judge),
        answer_correctness=correctness_score,
    )
