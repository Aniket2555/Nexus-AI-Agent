from langchain_core.language_models.chat_models import BaseChatModel

from backend.app.evaluation.judging import judge_score

FAITHFULNESS_JUDGE_PROMPT = (
    "You are grading whether an answer is faithful to its context.\n\n"
    "Context:\n{context}\n\nAnswer:\n{answer}\n\n"
    "On a single line, output only a number between 0 and 1: the fraction of "
    "factual\nclaims in the answer that are directly supported by the context. "
    "1 means every claim\nis supported; 0 means none are. Output nothing else.\n"
)

RELEVANCE_JUDGE_PROMPT = (
    "You are grading whether retrieved context is relevant to a question.\n\n"
    "Question:\n{question}\n\nRetrieved context:\n{context}\n\n"
    "On a single line, output only a number between 0 and 1: how relevant the "
    "context is\nto answering the question, regardless of whether it actually "
    "contains the answer.\n1 means highly relevant; 0 means completely unrelated. "
    "Output nothing else.\n"
)

ANSWER_CORRECTNESS_JUDGE_PROMPT = (
    "You are grading whether a generated answer matches a\n"
    "reference answer.\n\nReference answer:\n{reference}\n\nGenerated answer:\n{answer}\n\n"
    "On a single line, output only a number between 0 and 1: how well the "
    "generated answer\nmatches the reference answer's factual content. 1 means "
    "fully correct and complete;\n0 means wrong or contradicts the reference. "
    "Minor wording differences don't matter.\nOutput nothing else.\n"
)


async def faithfulness(
    answer: str, context: str, judge: BaseChatModel | None = None
) -> float | None:
    """LLM-judged faithfulness of `answer` to `context` — one holistic call. See
    `hallucination.py`'s `groundedness()` for the claim-by-claim version this
    doesn't catch partial hallucination as reliably as (one unsupported claim
    among ten supported ones can still read as "mostly faithful" to a single
    holistic judgment).
    """
    return await judge_score(
        FAITHFULNESS_JUDGE_PROMPT.format(context=context, answer=answer), judge
    )


async def relevance(
    question: str, context: str, judge: BaseChatModel | None = None
) -> float | None:
    """LLM-judged topical relevance of retrieved `context` to `question`.

    Distinct from `retrieval_metrics.context_precision`: that needs a golden set's
    `expected_chunk_ids` and only works in eval-harness mode. This works on any
    live query/context pair with no ground truth required, which is what makes it
    usable in `rag_evaluator.py` outside a fixed golden set.
    """
    return await judge_score(
        RELEVANCE_JUDGE_PROMPT.format(question=question, context=context), judge
    )


async def answer_correctness(
    answer: str, reference_answer: str, judge: BaseChatModel | None = None
) -> float | None:
    """LLM-judged match between a generated `answer` and a `reference_answer`.
    Only meaningful when a reference answer exists — most golden-set questions in
    this project only carry `expected_chunk_ids` (§2.7), not a reference answer, so
    this is `None` for those unless the caller supplies one.
    """
    return await judge_score(
        ANSWER_CORRECTNESS_JUDGE_PROMPT.format(reference=reference_answer, answer=answer), judge
    )
