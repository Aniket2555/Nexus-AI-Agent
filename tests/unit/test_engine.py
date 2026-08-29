from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from backend.app.context_engine.engine import ContextEngine, merge_query_variant_results
from backend.app.context_engine.intent_detector import IntentDetector
from backend.app.context_engine.query_rewriter import QueryRewriter
from backend.app.context_engine.summarizer import ConversationSummarizer
from backend.app.context_engine.token_budget import TokenBudget


def _doc(chunk_id: str, rrf_score: float) -> dict:
    return {"id": chunk_id, "metadata": {"chunk_id": chunk_id}, "rrf_score": rrf_score}


def test_merge_keeps_the_higher_score_for_a_chunk_found_by_multiple_variants():
    variant_a = [_doc("shared", 0.02), _doc("only-a", 0.05)]
    variant_b = [_doc("shared", 0.09), _doc("only-b", 0.01)]

    merged = merge_query_variant_results([variant_a, variant_b])

    ids_to_scores = {d["id"]: d["rrf_score"] for d in merged}
    assert ids_to_scores["shared"] == 0.09
    assert set(ids_to_scores) == {"only-a", "only-b", "shared"}


def test_merge_sorts_descending_by_score():
    merged = merge_query_variant_results([[_doc("low", 0.01), _doc("high", 0.9)]])
    assert [d["id"] for d in merged] == ["high", "low"]


def test_merge_empty_input():
    assert merge_query_variant_results([]) == []
    assert merge_query_variant_results([[]]) == []


async def test_expand_query_returns_intent_and_variants():
    rewriter = QueryRewriter(chat_model=GenericFakeChatModel(messages=iter([AIMessage("v1\nv2")])))
    detector = IntentDetector()
    engine = ContextEngine(rewriter=rewriter, intent_detector=detector)

    intent, variants = await engine.expand_query("Write a Python function to sort a list")

    assert intent.intent == "code"
    assert intent.method == "rule"
    assert variants == ["v1", "v2"]


async def test_assemble_includes_summary_and_budgets_the_prompt():
    summarizer = ConversationSummarizer(
        chat_model=GenericFakeChatModel(messages=iter([AIMessage("Earlier: PTO discussion.")]))
    )
    engine = ContextEngine(summarizer=summarizer, token_budget=TokenBudget(10000))

    from backend.app.context_engine.intent_detector import IntentResult

    messages = [HumanMessage(f"m{i}") for i in range(15)]

    result = await engine.assemble(
        "what ships with nexus?",
        "NEXUS ships with dense retrieval.",
        messages,
        intent=IntentResult(intent="qa", confidence=1.0, method="rule"),
    )

    assert "Earlier: PTO discussion." in result.system_prompt
    assert result.memory_summary == "Earlier: PTO discussion."
    assert result.token_allocations["rag"] > 0
    assert sum(result.token_allocations.values()) <= 10000
