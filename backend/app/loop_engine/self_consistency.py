import operator
from collections import Counter
from typing import Annotated, Any, TypedDict

from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.types import Send

from backend.app.config import get_settings
from backend.app.llm.provider import get_chat_model
from backend.app.utils.messages import as_text

SAMPLE_PROMPT = (
    "Answer the question below using only the provided context. Be concise — a "
    "short, direct answer, not an essay.\n\nContext:\n{context}\n\nQuestion:\n{question}\n"
)


class SampleState(TypedDict):
    """Per-branch input — deliberately narrow (just what one sample needs), distinct
    from the parent SelfConsistencyState so Send() only has to carry what each
    branch actually uses."""

    question: str
    context: str


class SelfConsistencyState(TypedDict):
    question: str
    context: str

    samples: Annotated[list[str], operator.add]
    final_answer: str
    vote_counts: dict[str, int]


async def sample_node(state: SampleState) -> dict[str, Any]:
    """One sample of N (a fan-out branch, dispatched by fan_out() below).

    temperature=0.7, not the pipeline's usual 0.1: at low temperature repeated
    samples of the same prompt come back nearly identical, and voting over
    identical samples doesn't reduce anything — the whole point of self-consistency
    is sampling genuinely different reasoning paths and seeing if they converge.
    """
    model = get_chat_model(temperature=0.7)
    prompt = SAMPLE_PROMPT.format(context=state["context"], question=state["question"])
    response = await model.ainvoke([SystemMessage(content=prompt)])
    return {"samples": [as_text(response)]}


def fan_out(state: SelfConsistencyState) -> list[Send]:
    """Dispatches `self_consistency_samples` parallel branches to sample_node via
    LangGraph's Send API — the graph-native fan-out mechanism (§3.2's "Fan-out/fan-in:
    N parallel branches -> voting node"), not a hand-rolled asyncio.gather loop.
    Under DECISIONS.md D4 this pattern *is* the learning value being demonstrated,
    so it's built on the framework primitive rather than reimplemented around it.
    """
    n = get_settings().self_consistency_samples
    return [
        Send("sample", {"question": state["question"], "context": state["context"]})
        for _ in range(n)
    ]


def _normalize(text: str) -> str:
    """Loose equality for voting: case/whitespace/trailing-punctuation differences
    between two samples that are substantively the same answer shouldn't split the
    vote between them."""
    return " ".join(text.lower().split()).strip(".,!? ")


async def vote_node(state: SelfConsistencyState) -> dict[str, Any]:
    """Majority vote over the samples, by normalized-text equality.

    Ties and all-distinct samples both resolve to `most_common(1)`'s first result,
    which is deterministic for a given `samples` list (Counter preserves insertion
    order among equal counts) — not randomly chosen.
    """
    samples = state.get("samples", [])
    if not samples:
        return {"final_answer": "", "vote_counts": {}}

    normalized = [_normalize(s) for s in samples]
    counts = Counter(normalized)
    winner_normalized, _ = counts.most_common(1)[0]
    winner = next(s for s, n in zip(samples, normalized, strict=True) if n == winner_normalized)

    return {"final_answer": winner, "vote_counts": dict(counts)}


builder = StateGraph(SelfConsistencyState)
builder.add_node("sample", sample_node)
builder.add_node("vote", vote_node)
builder.set_conditional_entry_point(fan_out, ["sample"])
builder.add_edge("sample", "vote")
builder.add_edge("vote", END)
self_consistency_graph = builder.compile()
