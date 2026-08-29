from langchain_core.messages import AIMessage

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

from backend.app.memory.knowledge_graph import (
    Entity,
    KnowledgeGraph,
    Relationship,
    _parse_extraction,
    extract_entities_and_relationships,
)

VALID_RESPONSE = """```json
{
  "entities": [
    {"name": "NEXUS", "type": "Project", "description": "An agentic AI platform"},
    {"name": "Groq", "type": "Tool", "description": "LLM inference provider"}
  ],
  "relationships": [
    {"source": "NEXUS", "target": "Groq", "type": "USES", "description": "for chat"}
  ]
}
```"""


def test_parses_fenced_json():
    result = _parse_extraction(VALID_RESPONSE)

    assert {e.name for e in result.entities} == {"NEXUS", "Groq"}
    assert result.relationships[0].source == "NEXUS"


def test_drops_entities_with_a_type_outside_the_allowed_vocabulary():
    raw = '{"entities": [{"name": "Bogus", "type": "NotARealType"}], "relationships": []}'

    result = _parse_extraction(raw)

    assert result.entities == []


def test_drops_relationships_referencing_an_entity_not_in_this_extraction():
    raw = """{
    "entities": [{"name": "NEXUS", "type": "Project"}],
    "relationships": [{"source": "NEXUS", "target": "Ghost", "type": "USES"}]
    }"""

    result = _parse_extraction(raw)

    assert result.relationships == []


def test_drops_relationships_with_a_type_outside_the_allowed_vocabulary():
    raw = """{
    "entities": [{"name": "A", "type": "Tool"}, {"name": "B", "type": "Tool"}],
    "relationships": [{"source": "A", "target": "B", "type": "NOT_A_REAL_EDGE"}]
    }"""

    result = _parse_extraction(raw)

    assert result.relationships == []


def test_unparseable_json_returns_an_empty_result_not_an_exception():
    result = _parse_extraction("this is not json at all")

    assert result.entities == []


def test_missing_entity_name_is_skipped():
    raw = '{"entities": [{"type": "Project"}], "relationships": []}'

    result = _parse_extraction(raw)

    assert result.entities == []


async def test_extract_entities_and_relationships_calls_the_model():
    model = GenericFakeChatModel(messages=iter([AIMessage(VALID_RESPONSE)]))

    result = await extract_entities_and_relationships("some text", chat_model=model)

    assert len(result.entities) == 2


class _FakeNeo4jGraph:
    def __init__(self) -> None:
        self.calls = []

    def query(self, cypher: str, params: dict) -> list[dict]:
        self.calls.append((cypher, params))
        return []


async def test_upsert_entities_merges_with_type_as_label_and_tenant_scoped():
    fake = _FakeNeo4jGraph()
    kg = KnowledgeGraph(graph=fake)

    await kg.upsert_entities(
        [Entity(name="NEXUS", type="Project", description="desc")], tenant_id="acme"
    )

    cypher, params = fake.calls[0]
    assert "MERGE (e:Entity:Project" in cypher
    assert params == {"name": "NEXUS", "tenant_id": "acme", "description": "desc"}


async def test_upsert_relationships_uses_type_as_relationship_label():
    fake = _FakeNeo4jGraph()
    kg = KnowledgeGraph(graph=fake)

    await kg.upsert_relationships(
        [Relationship(source="NEXUS", target="Groq", type="USES", description="for chat")],
        tenant_id="acme",
    )

    cypher, params = fake.calls[0]
    assert "MERGE (a)-[r:USES" in cypher
    assert params["source"] == "NEXUS"
    assert params["target"] == "Groq"
    assert params["tenant_id"] == "acme"


async def test_query_about_scopes_to_tenant():
    fake = _FakeNeo4jGraph()
    kg = KnowledgeGraph(graph=fake)

    await kg.query_about("NEXUS", tenant_id="acme")

    cypher, params = fake.calls[0]
    assert params["name"] == "NEXUS"
    assert params["tenant_id"] == "acme"
    assert "tenant_id" in cypher


async def test_multi_hop_traverse_interpolates_max_hops_as_a_literal():
    fake = _FakeNeo4jGraph()
    kg = KnowledgeGraph(graph=fake)

    await kg.multi_hop_traverse(["NEXUS"], tenant_id="acme", max_hops=3)

    cypher, params = fake.calls[0]
    assert "[*1..3]" in cypher
    assert params["names"] == ["NEXUS"]


async def test_multi_hop_traverse_with_no_entities_does_not_query():
    fake = _FakeNeo4jGraph()
    kg = KnowledgeGraph(graph=fake)

    result = await kg.multi_hop_traverse([], tenant_id="acme")

    assert result == []
    assert fake.calls == []
