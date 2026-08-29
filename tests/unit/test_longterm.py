import math
from dataclasses import dataclass, field

import pytest
from langgraph.store.base import MatchCondition

from backend.app.memory.longterm import QdrantStore, _matches, _point_id, _text_for_indexing


@dataclass
class _Point:
    id: str
    vector: list[float]
    payload: dict


@dataclass
class _FakeQdrantClient:
    points: dict[str, _Point] = field(default_factory=dict)
    collections: set[str] = field(default_factory=set)

    async def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.collections

    async def create_collection(self, collection_name, **kwargs) -> None:
        self.collections.add(collection_name)

    async def create_payload_index(self, *args, **kwargs) -> None:
        return None

    async def retrieve(self, collection_name, ids) -> list[_Point]:
        return [self.points[i] for i in ids if i in self.points]

    async def delete(self, collection_name, points_selector) -> None:
        for point_id in points_selector.points:
            self.points.pop(point_id, None)

    async def upsert(self, collection_name, points, wait: bool = True) -> None:
        for p in points:
            self.points[p.id] = _Point(p.id, p.vector, payload=p.payload)

    def _matches_filter(self, payload: dict, qfilter) -> bool:
        if qfilter is None:
            return True
        for cond in qfilter.must:
            value = payload
            for part in cond.key.split("."):
                value = value.get(part) if isinstance(value, dict) else None
            if value != cond.match.value:
                return False
        return True

    async def query_points(self, collection_name, query, query_filter=None, limit=10, offset=0, **kwargs):
        def cosine(a, b):
            dot = sum(x * y for x, y in zip(a, b, strict=True))
            na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
            return dot / (na * nb) if na and nb else 0.0

        candidates = [
            p for p in self.points.values() if self._matches_filter(p.payload, query_filter)
        ]
        scored = sorted(candidates, key=lambda p: cosine(query, p.vector), reverse=True)
        page = scored[offset : offset + limit]

        @dataclass
        class _ScoredPoint:
            id: str
            payload: dict
            score: float

        @dataclass
        class _Response:
            points: list

        scored_points = [_ScoredPoint(p.id, p.payload, cosine(query, p.vector)) for p in page]
        return _Response(points=scored_points)

    async def scroll(self, collection_name, scroll_filter=None, limit=10, offset=None, **kwargs):
        candidates = [
            p for p in self.points.values() if self._matches_filter(p.payload, scroll_filter)
        ]
        candidates.sort(key=lambda p: p.id)

        start = offset or 0
        end = start + limit
        if isinstance(start, int):
            page = candidates[start:end]
        else:
            page = candidates[:limit]

        has_more = isinstance(start, int) and end < len(candidates)
        next_offset = end if has_more else None

        return page, next_offset

    async def close(self) -> None:
        return None


class _FakeEmbeddings:
    """Deterministic 4-dim embedding: one-hot on a hash of the text, so unrelated
    texts are orthogonal and don't need a real model."""

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    async def aembed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        bucket = hash(text) % 4
        vec = [0.0, 0.0, 0.0, 0.0]
        vec[bucket] = 1.0
        return vec


@pytest.fixture
async def store():
    s = QdrantStore(
        "test-memory", embeddings=_FakeEmbeddings(), client=_FakeQdrantClient(), vector_size=4
    )
    await s.ensure_collection()
    return s


def test_point_id_is_deterministic():
    a = _point_id(("memory", "acme"), "fact-1")
    b = _point_id(("memory", "acme"), "fact-1")
    c = _point_id(("memory", "acme"), "fact-2")

    assert a == b
    assert a != c


def test_text_for_indexing_false_returns_none():
    assert _text_for_indexing({"text": "hello"}, False) is None


def test_text_for_indexing_explicit_fields():
    result = _text_for_indexing({"a": "one", "b": "two", "c": "three"}, ["a", "c"])
    assert result == "one three"


def test_text_for_indexing_prefers_text_field():
    result = _text_for_indexing({"text": "hello", "other": "x"}, None)
    assert result == "hello"


def test_text_for_indexing_falls_back_to_json():
    result = _text_for_indexing({"foo": "bar"}, None)
    assert "foo" in result and "bar" in result


def test_matches_prefix():
    cond = MatchCondition(match_type="prefix", path=("memory", "acme"))
    assert _matches(("memory", "acme", "user-1"), cond) is True
    assert _matches(("memory", "globex", "user-1"), cond) is False


def test_matches_prefix_with_wildcard():
    cond = MatchCondition(match_type="prefix", path=("memory", "*"))
    assert _matches(("memory", "acme", "user-1"), cond) is True
    assert _matches(("memory", "globex", "user-1"), cond) is True


def test_matches_suffix():
    cond = MatchCondition(match_type="suffix", path=("user-1",))
    assert _matches(("memory", "acme", "user-1"), cond) is True


async def test_put_then_get_round_trips(store):
    await store.aput(("memory", "acme"), "fact-1", {"text": "hello world"})

    item = await store.aget(("memory", "acme"), "fact-1")

    assert item.value == {"text": "hello world"}
    assert item.namespace == ("memory", "acme")


async def test_get_missing_key_returns_none(store):
    assert await store.aget(("memory", "acme"), "nope") is None


async def test_overwrite_preserves_created_at(store):
    await store.aput(("memory", "acme"), "fact-1", {"text": "v1"})
    first = await store.aget(("memory", "acme"), "fact-1")

    await store.aput(("memory", "acme"), "fact-1", {"text": "v2"})
    second = await store.aget(("memory", "acme"), "fact-1")

    assert second.created_at == first.created_at
    assert second.value == {"text": "v2"}


async def test_delete_removes_the_item(store):
    await store.aput(("memory", "acme"), "fact-1", {"text": "hello"})
    await store.adelete(("memory", "acme"), "fact-1")

    assert await store.aget(("memory", "acme"), "fact-1") is None


async def test_search_is_scoped_to_namespace_prefix(store):
    await store.aput(("memory", "acme"), "fact-1", {"text": "acme secret"})
    await store.aput(("memory", "globex"), "fact-1", {"text": "globex secret"})

    results = await store.asearch(("memory", "acme"))

    assert len(results) == 1
    assert results[0].value == {"text": "acme secret"}


async def test_search_with_query_ranks_by_similarity(store):
    await store.aput(("memory",), "a", {"text": "apple"})
    await store.aput(("memory",), "b", {"text": "banana"})

    results = await store.asearch(("memory",), query="apple")

    assert results[0].key == "a"
    assert results[0].score is not None


async def test_list_namespaces_returns_distinct_namespaces(store):
    await store.aput(("memory", "acme", "user-1"), "f1", {"text": "x"})
    await store.aput(("memory", "acme", "user-2"), "f1", {"text": "y"})

    namespaces = await store.alist_namespaces(prefix=("memory", "acme"))

    assert set(namespaces) == {("memory", "acme", "user-1"), ("memory", "acme", "user-2")}


async def test_index_false_item_is_gettable_but_not_semantically_searchable(store):
    await store.aput(("memory",), "raw", {"text": "should not be searchable"}, index=False)

    assert await store.aget(("memory",), "raw") is not None

    results = await store.asearch(("memory",), query="should not be searchable")

    assert results[0].score == 0.0
