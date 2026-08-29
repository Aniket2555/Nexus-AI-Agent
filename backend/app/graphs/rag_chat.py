import asyncio
import time
from dataclasses import asdict
from typing import Any, Literal

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from backend.app.config import get_settings
from backend.app.context_engine.engine import ContextEngine, merge_query_variant_results
from backend.app.graphs.identity import identity_from_config
from backend.app.graphs.states.rag_chat_state import RAGChatState
from backend.app.llm.provider import get_chat_model
from backend.app.loop_engine.reflection import evaluate_response, should_retry
from backend.app.memory.manager import get_memory_manager
from backend.app.observability.metrics import apush_metrics, rag_retrieval_latency_seconds
from backend.app.rag.processing.broad_query import is_broad_document_query
from backend.app.rag.processing.citation_builder import build_citations, format_context_block
from backend.app.rag.processing.compressor import compress_context
from backend.app.rag.processing.deduplicator import deduplicate
from backend.app.rag.processing.reranker import get_reranker
from backend.app.rag.retrieval.dense import get_retriever
from backend.app.rag.retrieval.hybrid import HybridRetriever
from backend.app.security.content_boundary import wrap_untrusted_content
from backend.app.utils.messages import as_text

_context_engine = ContextEngine()


async def rewrite_node(state: RAGChatState, config: RunnableConfig) -> dict[str, Any]:
    """Classify intent and generate query rewrite variants (§3.1 stages 1-2), before
    retrieval runs."""
    query = as_text(state["messages"][-1])
    if not query:
        return {"query_text": "", "intent": None, "rewritten_queries": []}

    intent, variants = await _context_engine.expand_query(query)
    return {"query_text": query, "intent": asdict(intent), "rewritten_queries": variants}


async def retrieve_node(state: RAGChatState, config: RunnableConfig) -> dict[str, Any]:
    """Hybrid (dense + sparse, RRF-fused) retrieval, scoped to the caller's tenant.

    Searches the original query plus every rewrite variant from rewrite_node, then
    merges by chunk id (merge_query_variant_results) — a chunk two phrasings both
    surface is a stronger relevance signal than either search alone.

    Timed and pushed to the Pushgateway (§8.6 gap: "RAG Retrieval Latency ...
    Not wired to a metric") — this node runs inside `langgraph dev`'s own
    process (D2), the same "batch process, not a server" shape
    `supervisor.py`'s `finalize_node` already solves for agent metrics, so it's
    pushed here rather than scraped.
    """
    query = state.get("query_text", "")
    if not query:
        return {"retrieved_docs": []}

    tenant_id, _ = identity_from_config(config)
    settings = get_settings()

    variants = state.get("rewritten_queries") or []
    queries = [query] + [v for v in variants if v != query]

    # HybridRetriever() builds a DenseRetriever(), which constructs a
    # HuggingFaceEmbeddings model — a genuinely blocking (disk/network) load —
    # synchronously in __init__. Threaded off for the same BlockingError reason as
    # rerank_node/compress_node above. (test_rag_chat_graph.py's _StubHybridRetriever
    # takes no constructor args, confirming this call site is bare HybridRetriever(),
    # not the get_retriever()/get_sparse_retriever() singletons used elsewhere.)
    hybrid = await asyncio.to_thread(HybridRetriever)
    start = time.perf_counter()
    result_sets = [
        await hybrid.search(q, tenant_id=tenant_id, top_k=settings.retrieval_top_k)
        for q in queries
    ]
    rag_retrieval_latency_seconds.observe(time.perf_counter() - start)
    await apush_metrics("nexus_rag_chat")

    merged = merge_query_variant_results(result_sets)[: settings.retrieval_top_k]

    if is_broad_document_query(query):
        doc_ids = {d["metadata"]["doc_id"] for d in merged if d["metadata"].get("doc_id")}
        by_chunk_id = {d["metadata"].get("chunk_id", d["id"]): d for d in merged}
        for doc_id in doc_ids:
            for chunk in await get_retriever().get_all_chunks(doc_id, tenant_id=tenant_id):
                chunk["force_include"] = True
                by_chunk_id[chunk["metadata"].get("chunk_id", chunk["id"])] = chunk
        merged = list(by_chunk_id.values())

    return {"retrieved_docs": merged}


async def memory_node(state: RAGChatState, config: RunnableConfig) -> dict[str, Any]:
    """Enrich retrieved_docs with long-term memory and knowledge-graph context
    (§4.1, §4.3) before reranking judges everything together — graph- and
    memory-derived candidates go through the exact same cross-encoder scoring as
    document chunks, rather than being special-cased into the prompt separately.
    """
    query = state.get("query_text", "")
    if not query:
        return {}

    tenant_id, user_id = identity_from_config(config)
    # get_memory_manager()'s first call in the process constructs QdrantStore(...) /
    # KnowledgeGraph(), which load the embeddings model and open store connections —
    # blocking, same reasoning as retrieve_node's HybridRetriever() above.
    manager = await asyncio.to_thread(get_memory_manager)
    memory_context = await manager.recall(tenant_id, user_id, query=query)

    existing = state.get("retrieved_docs", [])
    enriched = existing + memory_context.long_term_facts + memory_context.graph_context

    return {"retrieved_docs": enriched, "recalled_memory": asdict(memory_context)}


async def rerank_node(state: RAGChatState) -> dict[str, Any]:
    """Cross-encoder rerank of the hybrid candidates down to `rerank_top_k`.

    Passes rewrite_node's variants too, not just the raw query — see reranker.py's
    `rerank()` docstring for why a meta/structural query needs this to find its own
    genuinely-relevant chunk.

    `rerank()` is a synchronous, CPU-bound cross-encoder call — run via
    `asyncio.to_thread` so it doesn't block the event loop (same reasoning as
    `apush_metrics`'s docstring: an unwrapped sync call directly inside an
    `async def` node trips `langgraph dev`'s BlockingError detector).
    """
    docs = state.get("retrieved_docs", [])
    if not docs:
        return {"retrieved_docs": []}

    variants = state.get("rewritten_queries") or []
    # get_reranker() itself is wrapped too, not just .rerank() — it lazy-loads the
    # CrossEncoder model on first call (blocking disk/network I/O), which would
    # otherwise run on the event loop before the thread handoff even happens.
    reranked = await asyncio.to_thread(
        lambda: get_reranker().rerank(state["query_text"], docs, query_variants=variants)
    )
    return {"retrieved_docs": reranked}


async def compress_node(state: RAGChatState) -> dict[str, Any]:
    """Deduplicate, assemble, and token-budget-compress the reranked chunks.

    Deduplication and compression are both "shrink the context" concerns (§2.3), and
    run back to back here rather than as separate graph nodes — but citations are
    still built from `retrieved_docs` *before* compression, so a citation always
    resolves to a real chunk even if compression trimmed its text.
    """
    docs = state.get("retrieved_docs", [])
    if not docs:
        return {
            "retrieved_docs": [],
            "compressed_context": "No matching content was found in the knowledge base.",
            "compression_stats": {
                "method": "none",
                "original_tokens": 0,
                "compressed_tokens": 0,
            },
            "citations": [],
        }

    embeddings_model = get_retriever().embeddings
    vectors = await embeddings_model.aembed_documents([d["content"] for d in docs])
    deduped = deduplicate(docs, vectors)

    citations = [asdict(c) for c in build_citations(deduped)]
    context_block = format_context_block(deduped)
    # Sync + potentially blocking (lazy-loads the LLMLingua-2 model on first call) —
    # same asyncio.to_thread reasoning as rerank_node above.
    result = await asyncio.to_thread(compress_context, context_block, state["query_text"])

    return {
        "retrieved_docs": deduped,
        "compressed_context": wrap_untrusted_content(result.text),
        "compression_stats": {
            "method": result.method,
            "original_tokens": result.original_tokens,
            "compressed_tokens": result.compressed_tokens,
        },
        "citations": citations,
    }


async def assemble_node(state: RAGChatState) -> dict[str, Any]:
    """Summarize older turns, allocate the token budget, and render the final system
    prompt (§3.1 stages 3, 5, 9) from the compressed context §2's nodes produced.
    """
    from backend.app.context_engine.intent_detector import IntentResult

    intent_dict = state.get("intent") or {
        "intent": "qa",
        "confidence": 0.0,
        "method": "fallback",
    }
    intent = IntentResult(**intent_dict)

    context_block = state.get("compressed_context") or "No matching content was found."

    assembled = await _context_engine.assemble(
        state.get("query_text", ""), context_block, state["messages"], intent=intent
    )

    return {
        "system_prompt": assembled.system_prompt,
        "token_allocations": assembled.token_allocations,
    }


async def generate_node(state: RAGChatState) -> dict[str, Any]:
    """Draft an answer, grounded in the assembled, citation-aware system prompt.

    Writes to `draft_answer`, not `messages` — see rag_chat_state.py's comment on
    why: this node reruns on a reflection retry, and appending a fresh AIMessage
    each time would leave every rejected draft visible in the thread instead of
    replacing them. Only finalize_node touches `messages`.
    """
    system_prompt = state.get("system_prompt") or ""
    if state.get("reflection_feedback"):
        system_prompt += (
            "\n\n## Revision needed\nYour previous attempt scored below the quality "
            f"threshold. Feedback: {state['reflection_feedback']}"
            "\nRevise your answer to address this."
        )

    response = await get_chat_model().ainvoke(
        [SystemMessage(content=system_prompt), *state["messages"]]
    )
    return {"draft_answer": response}


async def reflect_node(state: RAGChatState) -> dict[str, Any]:
    """Self-evaluate the draft answer against the retrieved context (§3.2).

    Passes `get_chat_model(temperature=0)` explicitly rather than letting
    evaluate_response() resolve its own default — both this node and generate_node
    then go through the same `rag_chat.get_chat_model` reference, so tests only need
    to stub one boundary to cover both the drafting and the judging call.
    """
    draft = state.get("draft_answer")
    response_text = as_text(draft) if draft is not None else ""
    context = state.get("compressed_context", "")

    judge = get_chat_model(temperature=0)
    result = await evaluate_response(response_text, context, chat_model=judge)

    return {
        "quality_score": result.quality_score,
        "reflection_feedback": result.feedback,
        "iteration_count": state.get("iteration_count", 0) + 1,
        "reflection_cost_usd": state.get("reflection_cost_usd", 0.0) + result.cost_usd,
    }


def reflect_edge(state: RAGChatState) -> Literal["generate", "finalize"]:
    decision = should_retry(
        state.get("iteration_count", 0),
        state.get("quality_score", 1.0),
        state.get("reflection_cost_usd", 0.0),
    )
    return "generate" if decision == "retry" else "finalize"


async def finalize_node(state: RAGChatState) -> dict[str, Any]:
    """Append the (possibly revised) draft to `messages` — the one place in the
    graph that happens, regardless of how many reflection iterations ran."""
    draft = state.get("draft_answer")
    if draft is not None:
        return {"messages": [draft]}
    return {}


async def remember_node(state: RAGChatState, config: RunnableConfig) -> dict[str, Any]:
    """Write path (§4.1, §4.2): store the completed exchange in long-term memory
    and extract entities/relationships into the knowledge graph, after the answer
    is final — not before, so a rejected reflection draft is never written to
    memory as if it were the accepted answer.
    """
    tenant_id, user_id = identity_from_config(config)
    await get_memory_manager().remember(tenant_id, user_id, messages=state["messages"])
    return {}


builder = StateGraph(RAGChatState)
builder.add_node("rewrite", rewrite_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("memory", memory_node)
builder.add_node("rerank", rerank_node)
builder.add_node("compress", compress_node)
builder.add_node("assemble", assemble_node)
builder.add_node("generate", generate_node)
builder.add_node("reflect", reflect_node)
builder.add_node("finalize", finalize_node)
builder.add_node("remember", remember_node)

builder.set_entry_point("rewrite")
builder.add_edge("rewrite", "retrieve")
builder.add_edge("retrieve", "memory")
builder.add_edge("memory", "rerank")
builder.add_edge("rerank", "compress")
builder.add_edge("compress", "assemble")
builder.add_edge("assemble", "generate")
builder.add_edge("generate", "reflect")
builder.add_conditional_edges("reflect", reflect_edge, ["generate", "finalize"])
builder.add_edge("finalize", "remember")
builder.add_edge("remember", END)

rag_graph = builder.compile()
