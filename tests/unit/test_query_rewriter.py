from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from backend.app.context_engine.query_rewriter import QueryRewriter


def _fake(*responses: str) -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=iter([AIMessage(r) for r in responses]))


async def test_hyde_returns_model_output():
    rewriter = QueryRewriter(chat_model=_fake("NEXUS uses Qdrant for dense retrieval."))
    result = await rewriter.hyde("what does nexus use for retrieval?")
    assert result == "NEXUS uses Qdrant for dense retrieval."


async def test_hyde_falls_back_to_original_query_on_empty_response():
    rewriter = QueryRewriter(chat_model=_fake(""))
    result = await rewriter.hyde("original question")
    assert result == "original question"


async def test_step_back_returns_model_output():
    rewriter = QueryRewriter(chat_model=_fake("What retrieval strategies exist in RAG systems?"))
    result = await rewriter.step_back("what does nexus use for retrieval?")
    assert result == "What retrieval strategies exist in RAG systems?"


async def test_multi_query_splits_lines_into_variants():
    response = "How does NEXUS retrieve data?\nWhat is NEXUS's retrieval method?\nNEXUS retrieval?"
    rewriter = QueryRewriter(chat_model=_fake(response))

    variants = await rewriter.multi_query("what does nexus use for retrieval?", n=3)

    assert len(variants) == 3
    assert all(v for v in variants)


async def test_multi_query_strips_bullets_and_blank_lines():
    rewriter = QueryRewriter(chat_model=_fake("- First variant\n\n* Second variant\n"))
    variants = await rewriter.multi_query("q", n=2)
    assert variants == ["First variant", "Second variant"]


async def test_multi_query_falls_back_to_original_on_empty_response():
    rewriter = QueryRewriter(chat_model=_fake(""))
    variants = await rewriter.multi_query("original question", n=3)
    assert variants == ["original question"]
