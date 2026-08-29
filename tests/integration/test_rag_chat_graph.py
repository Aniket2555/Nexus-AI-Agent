import pytest
from langchain_core.messages import AIMessage, HumanMessage

from backend.app.context_engine.engine import AssembledPrompt
from backend.app.context_engine.intent_detector import IntentResult
from backend.app.graphs import rag_chat

RETRIEVED = [
    {
        "id": "3f2b0c14-0000-5000-8000-000000000000",
        "content": "NEXUS ships with dense retrieval over Qdrant.",
        "metadata": {
            "source": "handbook.pdf",
            "page": 12,
            "chunk_id": "acme:handbook.pdf:p12:c0",
            "tenant_id": "acme",
        },
        "score": 0.9123456,
        "rrf_score": 0.0328,
        "rrf_sources": {"dense": 0.0164, "sparse": 0.0164},
    },
    {
        "id": "5a1c2d33-0000-5000-8000-000000000001",
        "content": "Hybrid search fuses dense and sparse results with reciprocal rank fusion.",
        "metadata": {
            "source": "handbook.pdf",
            "page": 13,
            "chunk_id": "acme:handbook.pdf:p13:c0",
            "tenant_id": "acme",
        },
        "score": 0.8811,
        "rrf_score": 0.0301,
        "rrf_sources": {"dense": 0.0164},
    },
]


class _StubHybridRetriever:
    instances: list["_StubHybridRetriever"] = []

    def __init__(self) -> None:
        self.calls = []
        _StubHybridRetriever.instances.append(self)

    async def search(self, query: str, *, tenant_id: str, top_k: int = 10):
        self.calls.append({"query": query, "tenant_id": tenant_id, "top_k": top_k})
        return RETRIEVED


class _StubReranker:
    """Pass-through: tags every doc with a rerank_score but keeps all of them, so the
    fixed RETRIEVED set stays predictable through to citations/compression.
    """

    def rerank(
        self, query: str, docs: list[dict], *, query_variants: list[str] | None = None
    ) -> list[dict]:
        return [{**d, "rerank_score": 0.9} for d in docs]


class _FakeEmbeddings:
    """Orthogonal one-hot-ish vectors: guarantees nothing collides in deduplicator's
    cosine-similarity check, so both RETRIEVED docs survive to citations.
    """

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0 if i == j else 0.0 for j in range(len(texts))] for i in range(len(texts))]


class _StubDenseRetriever:
    embeddings = _FakeEmbeddings()


class _StubContextEngine:
    """expand_query returns no *new* rewrite variants (just the original query), so
    retrieve_node's dedup-by-value collapses back to exactly one hybrid search call
    — keeping the existing single-call assertions meaningful without also having to
    fake query rewriting's LLM call in every test.
    """

    async def expand_query(self, query: str):
        return IntentResult(intent="qa", confidence=1.0, method="rule"), [query]

    async def assemble(self, query, context_block, messages, *, intent=None):
        return AssembledPrompt(
            system_prompt=f"SYSTEM PROMPT ({intent.intent}):\n{context_block}",
            intent=intent,
            memory_summary=None,
            token_allocations={"system": 10, "query": 5, "buffer": 10, "memory": 0, "rag": 50},
        )


def _is_judge_prompt(text: str) -> bool:
    """True only for loop_engine/reflection.py's REFLECTION_PROMPT, never for a
    generate_node draft prompt. A plain "SCORE:" substring check is NOT safe here:
    on a retry, generate_node embeds the previous judge's full feedback text (which
    itself contains "SCORE:") into its own revision prompt — so that check would
    misclassify the second draft call as a judge call. This phrase is unique to the
    reflection prompt's instructions and is never echoed back into anything else.
    """
    return "Evaluate the following response for quality" in text


class _RoutingFakeChatModel:
    """Distinguishes the reflection judge's call from the drafting call by prompt
    content, so the reflection loop (§3.2, wired into every turn) passes on the
    first attempt by default — tests that don't specifically exercise retry
    behavior shouldn't have to also fake 3 rounds of it to get one answer through.
    """

    async def ainvoke(self, messages) -> AIMessage:
        text = messages[0].content if messages else ""
        if _is_judge_prompt(text):
            return AIMessage("SCORE: 1.0\nFEEDBACK: grounded and complete")
        return AIMessage("Grounded answer.")


class _StubMemoryManager:
    """No long-term/graph facts, and remember() is a no-op — memory_node/
    remember_node are exercised for real (they run as real graph nodes), only the
    Qdrant/Neo4j-backed manager underneath them is stubbed."""

    def __init__(self) -> None:
        self.remembered = []

    async def recall(self, tenant_id: str, user_id: str, *, query: str):
        from backend.app.memory.manager import MemoryContext

        return MemoryContext([], [])

    async def remember(self, tenant_id: str, user_id: str, *, messages) -> None:
        self.remembered.append({"tenant_id": tenant_id, "user_id": user_id, "messages": messages})


@pytest.fixture
def stub(monkeypatch):
    """Exercise the real compiled graph with every network/model boundary stubbed:
    HybridRetriever (Qdrant+Elasticsearch), the reranker (cross-encoder), the
    embeddings model (used by compress_node's dedup step), the context engine
    (query rewriting, intent detection, summarization, prompt assembly), the
    memory manager (long-term/graph recall + the write-back path), and the chat
    model (both the drafting call and the reflection judge call — both go through
    rag_chat.get_chat_model). Everything in between — dedup, citation building,
    compression's "already fits" path, the reflect/finalize routing — runs for real.
    """
    _StubHybridRetriever.instances.clear()
    monkeypatch.setattr(rag_chat, "HybridRetriever", _StubHybridRetriever)
    monkeypatch.setattr(rag_chat, "get_reranker", lambda: _StubReranker())
    monkeypatch.setattr(rag_chat, "get_retriever", lambda: _StubDenseRetriever())
    monkeypatch.setattr(rag_chat, "_context_engine", _StubContextEngine())

    stub_memory = _StubMemoryManager()
    monkeypatch.setattr(rag_chat, "get_memory_manager", lambda: stub_memory)
    monkeypatch.setattr(rag_chat, "get_chat_model", lambda *a, **k: _RoutingFakeChatModel())

    return stub_memory


async def test_citations_carry_real_page_and_chunk_id(stub):
    """Locks in the Phase 1 citation regression, now through hybrid+rerank+compress."""
    result = await rag_chat.rag_graph.ainvoke(
        {"messages": [HumanMessage("what ships with nexus?")]},
        {"configurable": {"tenant_id": "acme"}},
    )

    chunk_ids = {c["chunk_id"] for c in result["citations"]}
    assert chunk_ids == {"acme:handbook.pdf:p12:c0", "acme:handbook.pdf:p13:c0"}
    assert all(c["page"] in (12, 13) for c in result["citations"])


async def test_tenant_comes_from_config_not_state(stub):
    """A caller must not be able to read another tenant by editing state."""
    await rag_chat.rag_graph.ainvoke(
        {"messages": [HumanMessage("hello")]},
        {"configurable": {"tenant_id": "acme"}},
    )

    assert _StubHybridRetriever.instances[0].calls[0]["tenant_id"] == "acme"


async def test_multimodal_content_blocks_are_flattened_to_text(stub):
    """Agent Chat UI can send a list of typed blocks; embedding a list would raise."""
    await rag_chat.rag_graph.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=[
                        {"type": "text", "text": "describe this"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,AAA"},
                        },
                    ]
                )
            ]
        },
        {"configurable": {"tenant_id": "acme"}},
    )

    assert _StubHybridRetriever.instances[0].calls[0]["query"] == "describe this"


async def test_answer_is_appended_to_messages(stub):
    result = await rag_chat.rag_graph.ainvoke(
        {"messages": [HumanMessage("q")]},
        {"configurable": {"tenant_id": "acme"}},
    )

    assert isinstance(result["messages"][-1], AIMessage)


async def test_recalled_memory_is_surfaced_in_state(stub):
    result = await rag_chat.rag_graph.ainvoke(
        {"messages": [HumanMessage("what ships with nexus?")]},
        {"configurable": {"tenant_id": "acme"}},
    )

    assert result["recalled_memory"] == {"long_term_facts": [], "graph_context": []}


async def test_recalled_memory_reflects_real_recall_output(stub):
    """Same as above, but with non-empty recall() output, to prove this isn't
    just an empty-dict coincidence."""
    fact = {
        "id": "memory:acme:u1:k1",
        "content": "The user prefers concise answers.",
        "metadata": {"source": "long_term_memory", "key": "k1"},
        "score": 0.87,
    }

    async def recall_with_a_fact(tenant_id, user_id, *, query):
        from backend.app.memory.manager import MemoryContext

        return MemoryContext([fact], [])

    stub.recall = recall_with_a_fact

    result = await rag_chat.rag_graph.ainvoke(
        {"messages": [HumanMessage("what does the user prefer?")]},
        {"configurable": {"tenant_id": "acme"}},
    )

    assert result["recalled_memory"] == {"long_term_facts": [fact], "graph_context": []}

    chunk_ids = {c["chunk_id"] for c in result["citations"]}
    assert "memory:acme:u1:k1" in chunk_ids

    assert result["messages"][-1].content == "Grounded answer."


async def test_compressed_context_is_populated_and_not_truncated_below_budget(stub):
    result = await rag_chat.rag_graph.ainvoke(
        {"messages": [HumanMessage("what ships with nexus?")]},
        {"configurable": {"tenant_id": "acme"}},
    )

    assert "NEXUS ships with dense retrieval over Qdrant." in result["compressed_context"]
    assert result["compression_stats"]["method"] == "none"


async def test_compressed_context_is_wrapped_in_the_untrusted_content_boundary(stub):
    """§5.5: retrieved content must be delimited and labelled as data, not
    instructions, before it reaches the model — verified through the real
    compiled graph, not just wrap_untrusted_content() in isolation."""
    result = await rag_chat.rag_graph.ainvoke(
        {"messages": [HumanMessage("what ships with nexus?")]},
        {"configurable": {"tenant_id": "acme"}},
    )

    assert result["compressed_context"].startswith("<untrusted_content>")
    assert result["compressed_context"].rstrip().endswith("</untrusted_content>")
    assert "NEXUS ships with dense retrieval over Qdrant." in result["compressed_context"]


async def test_system_prompt_is_assembled_from_the_compressed_context(stub):
    result = await rag_chat.rag_graph.ainvoke(
        {"messages": [HumanMessage("what ships with nexus?")]},
        {"configurable": {"tenant_id": "acme"}},
    )

    assert result["system_prompt"].startswith("SYSTEM PROMPT (qa):")
    assert "NEXUS ships with dense retrieval over Qdrant." in result["system_prompt"]
    assert result["token_allocations"]["rag"] > 0


async def test_rewrite_variants_trigger_one_hybrid_search_per_variant_and_merge(stub, monkeypatch):
    """With real (non-identical) rewrite variants, retrieve_node must search each one
    and merge_query_variant_results must combine them — not just use the first.
    """

    class _MultiVariantEngine:
        async def expand_query(self, query: str):
            return (
                IntentResult(intent="qa", confidence=1.0, method="rule"),
                [query, "a different phrasing", "yet another phrasing"],
            )

        async def assemble(self, query, context_block, messages, *, intent=None):
            return AssembledPrompt(
                system_prompt=context_block,
                intent=intent,
                memory_summary=None,
                token_allocations={"system": 1, "query": 1, "buffer": 1, "memory": 1, "rag": 1},
            )

    monkeypatch.setattr(rag_chat, "_context_engine", _MultiVariantEngine())

    result = await rag_chat.rag_graph.ainvoke(
        {"messages": [HumanMessage("what ships with nexus?")]},
        {"configurable": {"tenant_id": "acme"}},
    )

    queries_searched = {c["query"] for c in _StubHybridRetriever.instances[0].calls}
    assert queries_searched == {
        "what ships with nexus?",
        "a different phrasing",
        "yet another phrasing",
    }

    chunk_ids = {c["chunk_id"] for c in result["citations"]}
    assert chunk_ids == {"acme:handbook.pdf:p12:c0", "acme:handbook.pdf:p13:c0"}


async def test_intent_is_populated_from_rewrite_node(stub):
    result = await rag_chat.rag_graph.ainvoke(
        {"messages": [HumanMessage("what ships with nexus?")]},
        {"configurable": {"tenant_id": "acme"}},
    )

    assert result["intent"]["intent"] == "qa"
    assert result["intent"]["method"] == "rule"


class _ImprovingFakeChatModel:
    """First draft scores low, second (revised) draft scores high — the judge's
    score is keyed on which draft text appears in its own prompt (REFLECTION_PROMPT
    embeds the draft verbatim), not on call order, so this stays correct regardless
    of how many times each node happens to be invoked.
    """

    def __init__(self) -> None:
        self.draft_calls = 0

    async def ainvoke(self, messages) -> AIMessage:
        text = messages[0].content if messages else ""

        if _is_judge_prompt(text):
            if "Weak answer." in text:
                return AIMessage("SCORE: 0.2\nFEEDBACK: too vague, revise")
            return AIMessage("SCORE: 0.9\nFEEDBACK: much better")

        self.draft_calls += 1
        return AIMessage("Weak answer." if self.draft_calls == 1 else "Improved answer.")


async def test_reflection_retries_a_low_quality_draft_and_finalizes_the_improved_one(
    stub, monkeypatch
):
    model = _ImprovingFakeChatModel()

    monkeypatch.setattr(rag_chat, "get_chat_model", lambda *a, **k: model)

    result = await rag_chat.rag_graph.ainvoke(
        {"messages": [HumanMessage("what ships with nexus?")]},
        {"configurable": {"tenant_id": "acme"}},
    )

    assert result["messages"][-1].content == "Improved answer."
    assert result["iteration_count"] == 2
    assert result["quality_score"] == 0.9

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(ai_messages) == 1


async def test_reflection_stops_at_max_iterations_even_if_quality_stays_low(stub, monkeypatch):
    class _AlwaysLowScoreModel:
        async def ainvoke(self, messages) -> AIMessage:
            text = messages[0].content if messages else ""
            if _is_judge_prompt(text):
                return AIMessage("SCORE: 0.1\nFEEDBACK: still not good enough")
            return AIMessage("Persistently mediocre answer.")

    monkeypatch.setattr(rag_chat, "get_chat_model", lambda *a, **k: _AlwaysLowScoreModel())

    result = await rag_chat.rag_graph.ainvoke(
        {"messages": [HumanMessage("what ships with nexus?")]},
        {"configurable": {"tenant_id": "acme"}},
    )

    assert result["iteration_count"] == 3
    assert result["messages"][-1].content == "Persistently mediocre answer."

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(ai_messages) == 1


async def test_no_hits_produces_a_graceful_no_context_answer(stub, monkeypatch):
    async def _empty_search(self, query: str, tenant_id: str, top_k: int = 10):
        return []

    monkeypatch.setattr(_StubHybridRetriever, "search", _empty_search)

    result = await rag_chat.rag_graph.ainvoke(
        {"messages": [HumanMessage("anything?")]},
        {"configurable": {"tenant_id": "acme"}},
    )

    assert result["citations"] == []
    assert "No matching content" in result["compressed_context"]


async def test_memory_node_merges_long_term_and_graph_facts_into_retrieved_docs(stub, monkeypatch):
    """Graph- and memory-derived candidates must reach rerank alongside document
    chunks, not bypass it — they go through the same relevance judgment."""
    from backend.app.memory.manager import MemoryContext

    class _EnrichingMemoryManager:
        async def recall(self, tenant_id: str, user_id: str, *, query: str) -> MemoryContext:
            return MemoryContext(
                [
                    {
                        "id": "memory:acme:default:fact-1",
                        "content": "The user previously asked about retrieval.",
                        "metadata": {"source": "long_term_memory"},
                        "score": 0.5,
                    }
                ],
                [
                    {
                        "id": "kg:NEXUS:USES:Groq",
                        "content": "NEXUS USES Groq: for chat",
                        "metadata": {"source": "knowledge_graph"},
                        "score": 1.0,
                    }
                ],
            )

        async def remember(self, tenant_id, user_id, *, messages) -> None:
            return None

    monkeypatch.setattr(rag_chat, "get_memory_manager", lambda: _EnrichingMemoryManager())

    result = await rag_chat.rag_graph.ainvoke(
        {"messages": [HumanMessage("what ships with nexus?")]},
        {"configurable": {"tenant_id": "acme"}},
    )

    sources = {c.get("chunk_id") or c.get("source") for c in result["citations"]}

    assert len(result["citations"]) == 4
    assert sources


async def test_remember_node_is_called_with_the_final_messages(stub):
    result = await rag_chat.rag_graph.ainvoke(
        {"messages": [HumanMessage("what ships with nexus?")]},
        {"configurable": {"tenant_id": "acme", "user_id": "user-42"}},
    )

    assert len(stub.remembered) == 1
    call = stub.remembered[0]
    assert call["tenant_id"] == "acme"
    assert call["user_id"] == "user-42"
    assert call["messages"][-1].content == result["messages"][-1].content


async def test_remember_defaults_user_id_when_not_provided(stub):
    await rag_chat.rag_graph.ainvoke(
        {"messages": [HumanMessage("hello")]},
        {"configurable": {"tenant_id": "acme"}},
    )

    assert stub.remembered[0]["user_id"] == "default"
