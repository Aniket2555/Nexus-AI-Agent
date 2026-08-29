import asyncio
import uuid
from dataclasses import asdict
from typing import Any, TypedDict

from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph

from backend.app.config import get_settings
from backend.app.graphs.states.shared_state import SpecialistResult
from backend.app.llm.provider import get_chat_model
from backend.app.memory.manager import get_memory_manager
from backend.app.rag.processing.citation_builder import build_citations, format_context_block
from backend.app.rag.processing.reranker import get_reranker
from backend.app.rag.retrieval.hybrid import HybridRetriever
from backend.app.security.content_boundary import wrap_untrusted_content
from backend.app.utils.messages import as_text

RESEARCH_PROMPT = (
    "You are the Research specialist in a multi-agent system.\n\n"
    "Answer the research question using ONLY the retrieved context below. Cite every\n"
    "claim inline as [Source: <file>, Page: <n>]. If the context does not contain\n"
    "enough to answer, say plainly what is missing rather than filling the gap from\n"
    "memory.\n\n{context}\n\nResearch question: {question}\n"
)


class ResearchState(TypedDict):
    question: str
    tenant_id: str
    user_id: str
    retrieved_docs: list[dict[str, Any]]
    answer: str
    citations: list[dict[str, Any]]


async def retrieve_node(state: ResearchState) -> dict[str, Any]:
    """Dense+sparse+graph+long-term-memory retrieval, reusing every retrieval layer
    built in Phases 1-4 rather than a research-specific search implementation —
    there's nothing "research" needs that the RAG chat pipeline doesn't already do.

    No query rewriting or reflection here (Phase 3's rewrite/reflect nodes stay
    specific to `rag_chat`'s single-turn conversational flow) — this specialist is
    invoked with an already-decomposed sub-task question from the supervisor's
    planner, which is a different kind of input than a raw user turn.
    """
    settings = get_settings()
    # HybridRetriever() and get_memory_manager() both lazily construct blocking
    # resources (embeddings model load, store/graph connections) on first call in
    # the process — threaded off so langgraph dev's BlockingError detector doesn't
    # trip, same reasoning as rag_chat.py's retrieve_node/memory_node.
    hybrid = await asyncio.to_thread(HybridRetriever)
    hybrid_docs = await hybrid.search(
        state["question"], tenant_id=state["tenant_id"], top_k=settings.retrieval_top_k
    )

    manager = await asyncio.to_thread(get_memory_manager)
    memory = await manager.recall(state["tenant_id"], state["user_id"], query=state["question"])

    candidates = hybrid_docs + memory.long_term_facts + memory.graph_context
    # rerank() is a plain sync, CPU-bound cross-encoder call — not awaitable, and
    # blocking if called directly from this async node. get_reranker() itself is
    # wrapped too, since its first call lazy-loads the CrossEncoder model.
    reranked = await asyncio.to_thread(lambda: get_reranker().rerank(state["question"], candidates))

    return {"retrieved_docs": reranked}


async def synthesize_node(state: ResearchState) -> dict[str, Any]:
    docs = state.get("retrieved_docs", [])
    citations = [asdict(c) for c in build_citations(docs)]
    context_block = wrap_untrusted_content(format_context_block(docs))

    prompt = RESEARCH_PROMPT.format(context=context_block, question=state["question"])
    response = await get_chat_model().ainvoke([SystemMessage(content=prompt)])

    return {"answer": as_text(response), "citations": citations}


builder = StateGraph(ResearchState)
builder.add_node("retrieve", retrieve_node)
builder.add_node("synthesize", synthesize_node)
builder.set_entry_point("retrieve")
builder.add_edge("retrieve", "synthesize")
builder.add_edge("synthesize", END)
research_graph = builder.compile()


async def run_research(
    task_id: str, question: str, tenant_id: str, user_id: str
) -> SpecialistResult:
    """Supervisor-facing entry point: runs the sub-graph, returns the common
    SpecialistResult shape dispatch (§6.2) appends to `agent_results`."""
    result = await research_graph.ainvoke(
        {"question": question, "tenant_id": tenant_id, "user_id": user_id}
    )

    return SpecialistResult(
        task_id=task_id,
        specialist="research",
        summary=result.get("answer", ""),
        success=bool(result.get("answer")),
        citations=result.get("citations", []),
        details={"query_id": str(uuid.uuid4())},
    )
