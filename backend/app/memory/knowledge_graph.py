import asyncio
import json
import logging
import re
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_neo4j import Neo4jGraph

from backend.app.config import get_settings
from backend.app.llm.provider import get_chat_model
from backend.app.utils.messages import as_text

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = (
    "Extract entities and relationships from the text below.\n\n"
    "Allowed entity types: {node_types}\n"
    "Allowed relationship types: {edge_types}\n\n"
    "Respond with ONLY valid JSON, no markdown fences, no commentary, in exactly "
    "this shape:\n"
    "{{\n"
    '  "entities": [{{"name": "...", "type": "...", "description": "..."}}],\n'
    '  "relationships": [{{"source": "...", "target": "...", "type": "...", '
    '"description": "..."}}]\n'
    "}}\n\n"
    "Only extract what the text actually states or clearly implies — do not invent "
    "entities.\n"
    "`source`/`target` in relationships must be `name` values from the entities "
    'list. If nothing\nqualifies, return {{"entities": [], "relationships": []}}.\n\n'
    "Text:\n{text}\n"
)

_CODE_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


@dataclass
class Entity:
    name: str
    type: str
    description: str = ""


@dataclass
class Relationship:
    source: str
    target: str
    type: str
    description: str = ""


@dataclass
class ExtractionResult:
    entities: list[Entity]
    relationships: list[Relationship]


def _parse_extraction(raw: str) -> ExtractionResult:
    """Parse the LLM's JSON response, dropping (not raising on) anything that
    doesn't fit the allowed vocabulary — one malformed entity in a batch of ten
    shouldn't discard the other nine. Returns an empty result, not an exception, if
    the response isn't valid JSON at all: extraction feeding the graph is a
    best-effort enrichment step, not something a chat turn should fail over.
    """
    settings = get_settings()
    text = _CODE_FENCE.sub("", raw.strip())
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Entity extraction returned unparseable JSON: %r", raw[:200])
        return ExtractionResult(entities=[], relationships=[])

    entities: list[Entity] = []
    for item in data.get("entities", []):
        if not isinstance(item, dict) or item.get("type") not in settings.kg_node_types:
            continue
        name = item.get("name")
        if not name:
            continue
        entities.append(
            Entity(name=name, type=item["type"], description=item.get("description", ""))
        )

    valid_names = {e.name for e in entities}
    relationships: list[Relationship] = []
    for item in data.get("relationships", []):
        if not isinstance(item, dict) or item.get("type") not in settings.kg_edge_types:
            continue
        source, target = item.get("source"), item.get("target")
        if source not in valid_names or target not in valid_names:
            continue
        relationships.append(
            Relationship(
                source=source,
                target=target,
                type=item["type"],
                description=item.get("description", ""),
            )
        )

    return ExtractionResult(entities=entities, relationships=relationships)


async def extract_entities_and_relationships(
    text: str, *, chat_model: BaseChatModel | None = None
) -> ExtractionResult:
    """LLM-based extraction (§4.2) — the genuinely custom part of the knowledge
    graph per DECISIONS.md D4; the connection/query layer below reuses Neo4jGraph.
    """
    settings = get_settings()
    model = chat_model or get_chat_model(temperature=0)
    prompt = EXTRACTION_PROMPT.format(
        node_types=", ".join(settings.kg_node_types),
        edge_types=", ".join(settings.kg_edge_types),
        text=text,
    )
    response = await model.ainvoke(prompt)
    return _parse_extraction(as_text(response))


class KnowledgeGraph:
    """Neo4j connection/query layer, reusing `langchain_neo4j.Neo4jGraph` rather
    than hand-writing driver/session management (DECISIONS.md D4). Every node
    carries a generic `:Entity` label plus its specific type label (e.g.
    `:Entity:Project`) and a `tenant_id` property — queries always filter on
    `tenant_id`, the same multi-tenancy discipline as Qdrant's payload filter
    (Phase 1) and this store's own namespace scoping (§4.1).

    `Neo4jGraph.query()` is sync-only (no `aquery`); wrapped in `asyncio.to_thread`
    so it doesn't block the event loop when called from an async node.
    """

    def __init__(self, graph: Neo4jGraph | None = None) -> None:
        self._graph = graph

    @property
    def graph(self) -> Neo4jGraph:
        if self._graph is None:
            settings = get_settings()
            self._graph = Neo4jGraph(
                settings.neo4j_url,
                settings.neo4j_user,
                settings.neo4j_password,
                refresh_schema=False,
            )
        return self._graph

    async def _query(self, cypher: str, params: dict) -> list[dict]:
        # `self.graph` (not just `.query()`) must be resolved inside the thread too —
        # the property lazily constructs Neo4jGraph(...) on first access, which opens
        # a blocking driver connection. Accessing it before to_thread() runs (e.g.
        # `self.graph.query` as a bound-method argument) would evaluate that
        # construction on the event loop instead.
        return await asyncio.to_thread(lambda: self.graph.query(cypher, params))

    async def upsert_entities(self, entities: list[Entity], tenant_id: str) -> None:
        for entity in entities:
            await self._query(
                f"MERGE (e:Entity:{entity.type} {{name: $name, tenant_id: $tenant_id}}) "
                "ON CREATE SET e.description = $description "
                "ON MATCH SET e.description = coalesce($description, e.description)",
                {
                    "name": entity.name,
                    "tenant_id": tenant_id,
                    "description": entity.description or None,
                },
            )

    async def upsert_relationships(
        self, relationships: list[Relationship], tenant_id: str
    ) -> None:
        for rel in relationships:
            await self._query(
                "MATCH (a:Entity {name: $source, tenant_id: $tenant_id}), "
                "(b:Entity {name: $target, tenant_id: $tenant_id}) "
                f"MERGE (a)-[r:{rel.type} {{tenant_id: $tenant_id}}]->(b) "
                "ON CREATE SET r.description = $description",
                {
                    "source": rel.source,
                    "target": rel.target,
                    "tenant_id": tenant_id,
                    "description": rel.description or None,
                },
            )

    async def query_about(
        self, entity_name: str, tenant_id: str, *, limit: int = 20
    ) -> list[dict]:
        """"What do we know about X?" — every direct relationship, either direction."""
        return await self._query(
            "MATCH (e:Entity {name: $name, tenant_id: $tenant_id})-[r]-(related:Entity) "
            "WHERE related.tenant_id = $tenant_id "
            "RETURN e.name AS entity, type(r) AS relationship, related.name AS "
            "related_entity, labels(related) AS related_labels, "
            "r.description AS relationship_description LIMIT $limit",
            {"name": entity_name, "tenant_id": tenant_id, "limit": limit},
        )

    async def multi_hop_traverse(
        self,
        entity_names: list[str],
        tenant_id: str,
        *,
        max_hops: int = 2,
        limit: int = 20,
    ) -> list[dict]:
        """"How are X and Y related?" and general multi-hop context expansion.

        `max_hops` is interpolated, not parameterized — Cypher's variable-length
        relationship syntax (`[*1..N]`) requires a literal integer, not a bind
        parameter, in every Neo4j version. Safe here because `max_hops` comes from
        `settings.graph_rag_max_hops` (an int), never from request-controlled text.
        """
        if not entity_names:
            return []
        return await self._query(
            "UNWIND $names AS start_name MATCH path = "
            f"(e:Entity {{name: start_name, tenant_id: $tenant_id}})-[*1..{int(max_hops)}]"
            "-(related:Entity {tenant_id: $tenant_id}) WHERE related.name <> start_name "
            "RETURN related.name AS name, labels(related) AS labels, "
            "min(length(path)) AS hops ORDER BY hops LIMIT $limit",
            {"names": entity_names, "tenant_id": tenant_id, "limit": limit},
        )
