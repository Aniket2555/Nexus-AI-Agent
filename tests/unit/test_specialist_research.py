from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from backend.app.graphs.specialists import research as research_module
from backend.app.memory.manager import MemoryContext

RETRIEVED = [
    {
        "id": "point-1",
        "content": "NEXUS uses Groq for chat and local embeddings for retrieval.",
        "metadata": {"source": "handbook.pdf", "page": 3, "chunk_id": "acme:handbook.pdf:p3:c0"},
        "score": 0.9,
    }
]


class _StubHybrid:
    def __init__(self, docs):
        self.docs = docs
        self.calls = []

    async def search(self, query, tenant_id, top_k=10):
        self.calls.append({"query": query, "tenant_id": tenant_id})
        return self.docs


class _StubMemoryManager:
    async def recall(self, tenant_id, user_id, *, query):
        return MemoryContext(long_term_facts=[], graph_context=[])


class _StubReranker:
    def rerank(self, query, docs):
        return docs


async def test_retrieve_node_merges_hybrid_and_memory_results(monkeypatch):
    stub_hybrid = _StubHybrid(RETRIEVED)
    monkeypatch.setattr(research_module, "HybridRetriever", lambda: stub_hybrid)
    monkeypatch.setattr(research_module, "get_memory_manager", lambda: _StubMemoryManager())
    monkeypatch.setattr(research_module, "get_reranker", lambda: _StubReranker())

    result = await research_module.retrieve_node(
        {"question": "what does nexus use for chat?", "tenant_id": "acme", "user_id": "u1"}
    )

    assert result["retrieved_docs"] == RETRIEVED
    assert stub_hybrid.calls[0]["tenant_id"] == "acme"


async def test_synthesize_node_produces_answer_and_citations(monkeypatch):
    model = GenericFakeChatModel(messages=iter([AIMessage("NEXUS uses Groq for chat.")]))
    monkeypatch.setattr(research_module, "get_chat_model", lambda: model)

    result = await research_module.synthesize_node(
        {"question": "q", "tenant_id": "acme", "user_id": "u1", "retrieved_docs": RETRIEVED}
    )

    assert result["answer"] == "NEXUS uses Groq for chat."
    assert result["citations"][0]["chunk_id"] == "acme:handbook.pdf:p3:c0"


async def test_run_research_returns_a_specialist_result(monkeypatch):
    stub_hybrid = _StubHybrid(RETRIEVED)
    monkeypatch.setattr(research_module, "HybridRetriever", lambda: stub_hybrid)
    monkeypatch.setattr(research_module, "get_memory_manager", lambda: _StubMemoryManager())
    monkeypatch.setattr(research_module, "get_reranker", lambda: _StubReranker())

    model = GenericFakeChatModel(messages=iter([AIMessage("Grounded research answer.")]))
    monkeypatch.setattr(research_module, "get_chat_model", lambda: model)

    result = await research_module.run_research(
        task_id="task-1", question="what does nexus use?", tenant_id="acme", user_id="u1"
    )

    assert result["task_id"] == "task-1"
    assert result["specialist"] == "research"
    assert result["success"] is True
    assert result["summary"] == "Grounded research answer."
    assert len(result["citations"]) == 1
