import uuid
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AnyMessage

from backend.app.config import get_settings
from backend.app.memory.knowledge_graph import KnowledgeGraph, extract_entities_and_relationships
from backend.app.memory.longterm import QdrantStore
from backend.app.rag.retrieval.dense import get_retriever
from backend.app.rag.retrieval.graph_rag import GraphRetriever
from backend.app.utils.messages import as_text


@dataclass
class MemoryContext:
    long_term_facts: list[dict[str, Any]] = field(default_factory=list)
    graph_context: list[dict[str, Any]] = field(default_factory=list)


class MemoryManager:
    """Orchestrates the memory layers from §4.1's table that actually need new
    code. Working memory has no fetch step — it *is* the caller's `messages` list
    (D4's reuse decision) — and conversation memory doesn't either: persistence is
    the LangGraph Platform's job (D6), and the summarization/compaction half is
    already Phase 3's `ContextEngine` → `ConversationSummarizer`, wired into
    `rag_chat.py`'s `assemble_node`. Calling it a second time here would mean two
    separate summarization LLM calls per turn producing two summaries that could
    disagree — so this manager only owns the two layers Phase 4 actually adds:
    long-term memory and the knowledge graph.
    """

    def __init__(
        self, store: QdrantStore | None = None, kg: KnowledgeGraph | None = None
    ) -> None:
        self.store = store or QdrantStore(embeddings=get_retriever().embeddings)
        self.kg = kg or KnowledgeGraph()

    async def recall(self, tenant_id: str, user_id: str, query: str) -> MemoryContext:
        """Fetch long-term memory and graph context, scoped to this tenant+user.
        Independent of each other — a slow or unavailable one doesn't block the
        other (graph_rag.py already degrades gracefully on its own; an unreachable
        long-term store is not similarly guarded, since Qdrant is already a hard
        dependency for primary document retrieval in the same turn — there's no
        graceful "partial" RAG chat without it).
        """
        settings = get_settings()
        long_term_items = await self.store.asearch(
            ("memory", tenant_id, user_id), query=query, limit=settings.graph_rag_top_k
        )
        long_term_facts = [
            {
                "id": f"memory:{tenant_id}:{user_id}:{item.key}",
                "content": item.value.get("text", ""),
                "metadata": {"source": "long_term_memory", "key": item.key},
                "score": item.score if item.score is not None else 1.0,
            }
            for item in long_term_items
        ]

        graph_context = await GraphRetriever(kg=self.kg).search(
            query, tenant_id=tenant_id, top_k=settings.graph_rag_top_k
        )

        return MemoryContext(long_term_facts=long_term_facts, graph_context=graph_context)

    async def remember(
        self, tenant_id: str, user_id: str, messages: list[AnyMessage]
    ) -> None:
        """Write path, called after a turn completes: store the exchange in
        long-term memory and extract entities/relationships into the graph.

        One more LLM call (entity extraction) per turn, on top of what generation
        and reflection already cost (§3's cost-tradeoff callout) — real, not free.
        """
        text = "\n".join(f"{m.type}: {as_text(m)}" for m in messages[-2:] if as_text(m))
        if not text:
            return

        await self.store.aput(("memory", tenant_id, user_id), str(uuid.uuid4()), {"text": text})

        extraction = await extract_entities_and_relationships(text)
        if extraction.entities:
            await self.kg.upsert_entities(extraction.entities, tenant_id=tenant_id)
        if extraction.relationships:
            await self.kg.upsert_relationships(extraction.relationships, tenant_id=tenant_id)


_manager: MemoryManager | None = None


def get_memory_manager() -> MemoryManager:
    """Process-wide singleton — same rationale as dense.py's get_retriever():
    avoids reloading the embeddings model and reopening store/graph connections
    per turn."""
    global _manager
    if _manager is None:
        _manager = MemoryManager()
    return _manager
