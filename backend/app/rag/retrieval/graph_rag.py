import logging
from typing import Any

from backend.app.memory.knowledge_graph import KnowledgeGraph, extract_entities_and_relationships

logger = logging.getLogger(__name__)


class GraphRetriever:
    """Wraps knowledge-graph entity lookups into the same {id, content, metadata,
    score} shape dense.py/sparse.py return, so a graph-derived fact competes for
    relevance in the reranker exactly like a document chunk, rather than being
    special-cased into the prompt separately.
    """

    def __init__(self, kg: KnowledgeGraph | None = None) -> None:
        self.kg = kg or KnowledgeGraph()

    async def search(
        self, query: str, *, tenant_id: str, top_k: int = 5
    ) -> list[dict[str, Any]]:
        try:
            extraction = await extract_entities_and_relationships(query)
            docs: list[dict[str, Any]] = []
            for entity in extraction.entities[:3]:  # capped: each is a Neo4j round trip
                facts = await self.kg.query_about(entity.name, tenant_id=tenant_id, limit=top_k)
                docs.extend(self._facts_to_docs(entity.name, facts))
            return docs[:top_k]
        except Exception:
            logger.warning("Graph retrieval unavailable; continuing without it.")
            return []

    def _facts_to_docs(self, entity_name: str, facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        docs = []
        for fact in facts:
            docs.append(
                {
                    "id": f"kg:{entity_name}:{fact.get('relationship', '')}:{fact.get('target', '')}",
                    "content": fact.get("description", ""),
                    "metadata": {
                        "source": "knowledge_graph",
                        "entity": entity_name,
                        "related_entity": fact.get("target"),
                        "relationship": fact.get("relationship"),
                    },
                    # Graph facts are exact, not similarity-ranked — a fixed score
                    # leaves it to the reranker to judge actual relevance.
                    "score": 1.0,
                }
            )
        return docs
