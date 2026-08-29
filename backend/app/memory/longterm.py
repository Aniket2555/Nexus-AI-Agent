import asyncio
import json
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    MatchCondition,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)
from qdrant_client import AsyncQdrantClient, models

from backend.app.config import get_settings

MEMORY_NAMESPACE = uuid.UUID("6ba7b812-9dad-11d1-80b4-00c04fd430c8")


def _point_id(namespace: tuple[str, ...], key: str) -> str:
    return str(uuid.uuid5(MEMORY_NAMESPACE, f"{'/'.join(namespace)}:{key}"))


def _text_for_indexing(value: dict[str, Any], index: Any) -> str | None:
    """What gets embedded for semantic search, per BaseStore's `index` contract.

    `index=False` -> not searchable (caller opted out). `index=[<fields>]` -> only
    those top-level fields, concatenated. `index=None` (default) -> a "text" or
    "content" field if the caller happens to use one of those conventional names,
    else the whole value JSON-serialized so *something* reasonable is searchable
    without the caller having to think about it.
    """
    if index is False:
        return None
    if isinstance(index, list):
        parts = [str(value[field]) for field in index if field in value]
        text = " ".join(parts).strip()
        return text or None
    if "text" in value:
        return str(value["text"])
    if "content" in value:
        return str(value["content"])
    return json.dumps(value, sort_keys=True, default=str)


def _matches(namespace: tuple[str, ...], condition: MatchCondition) -> bool:
    """Prefix/suffix namespace matching with single-segment "*" wildcards.

    Scope decision: LangGraph's full NamespacePath spec allows richer per-segment
    patterns; this implements the common case (exact segment or "*" wildcard per
    position) rather than a complete pattern-matching engine, since `list_namespaces`
    is a minor path for this store's actual usage (get/put/search with concrete
    namespaces dominate — see manager.py).
    """
    path = condition.path
    if condition.match_type == "prefix":
        if len(path) > len(namespace):
            return False
        segment = namespace[: len(path)]
    else:
        if len(path) > len(namespace):
            return False
        segment = namespace[len(namespace) - len(path) :]
    return all(p == "*" or p == n for p, n in zip(path, segment, strict=True))


class QdrantStore(BaseStore):
    """`langgraph.store.base.BaseStore` backed by Qdrant (§4.1's "reuse the
    interface, build the backend" — `BaseStore` is the framework's cross-thread
    memory primitive; this is the storage implementation behind it).

    Only `batch`/`abatch` are actually abstract on `BaseStore` — `get`/`put`/
    `delete`/`search`/`list_namespaces` are convenience wrappers already implemented
    in terms of them, so this class only needs to dispatch on the four `Op` types.
    """

    def __init__(
        self,
        collection_name: str | None = None,
        embeddings: Embeddings | None = None,
        client: AsyncQdrantClient | None = None,
        vector_size: int | None = None,
    ) -> None:
        settings = get_settings()
        self.collection_name = collection_name or settings.qdrant_memory_collection
        self.vector_size = vector_size or settings.embedding_dimensions
        self.max_namespace_depth = settings.memory_max_namespace_depth
        self.client = client or AsyncQdrantClient(url=settings.qdrant_url)
        self.embeddings = embeddings or HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )

    async def close(self) -> None:
        await self.client.close()

    async def ensure_collection(self) -> None:
        """Idempotent. Called once at startup — never per request."""
        if await self.client.collection_exists(self.collection_name):
            return
        await self.client.create_collection(
            self.collection_name,
            vectors_config=models.VectorParams(
                size=self.vector_size, distance=models.Distance.COSINE
            ),
        )
        for i in range(self.max_namespace_depth):
            await self.client.create_payload_index(
                self.collection_name,
                field_name=f"ns_{i}",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    def _namespace_payload(self, namespace: tuple[str, ...]) -> dict[str, str]:
        if len(namespace) > self.max_namespace_depth:
            raise ValueError(
                f"Namespace depth {len(namespace)} exceeds "
                f"memory_max_namespace_depth={self.max_namespace_depth}: {namespace}"
            )
        return {f"ns_{i}": part for i, part in enumerate(namespace)}

    async def _do_get(self, op: GetOp) -> Item | None:
        points = await self.client.retrieve(
            self.collection_name, ids=[_point_id(op.namespace, op.key)]
        )
        if not points:
            return None
        payload = points[0].payload
        return Item(
            value=payload["value"],
            key=payload["key"],
            namespace=tuple(payload["namespace"]),
            created_at=datetime.fromisoformat(payload["created_at"]),
            updated_at=datetime.fromisoformat(payload["updated_at"]),
        )

    async def _do_put(self, op: PutOp) -> None:
        point_id = _point_id(op.namespace, op.key)
        if op.value is None:
            await self.client.delete(
                self.collection_name,
                points_selector=models.PointIdsList(points=[point_id]),
            )
            return

        text = _text_for_indexing(op.value, op.index)
        if text:
            vector = (await self.embeddings.aembed_documents([text]))[0]
        else:
            vector = [0.0] * self.vector_size

        existing = await self.client.retrieve(self.collection_name, ids=[point_id])
        now = datetime.now(UTC).isoformat()
        if existing:
            created_at = existing[0].payload["created_at"]
        else:
            created_at = now

        payload = {
            "namespace": list(op.namespace),
            "key": op.key,
            "value": op.value,
            "created_at": created_at,
            "updated_at": now,
            **self._namespace_payload(op.namespace),
        }
        await self.client.upsert(
            self.collection_name,
            wait=True,
            points=[models.PointStruct(id=point_id, vector=vector, payload=payload)],
        )

    async def _do_search(self, op: SearchOp) -> list[SearchItem]:
        must = [
            models.FieldCondition(key=f"ns_{i}", match=models.MatchValue(value=part))
            for i, part in enumerate(op.namespace_prefix)
        ]
        for field, value in (op.filter or {}).items():
            must.append(
                models.FieldCondition(
                    key=f"value.{field}", match=models.MatchValue(value=value)
                )
            )
        query_filter = models.Filter(must=must) if must else None

        scores: dict[Any, float] = {}
        if op.query:
            vector = await self.embeddings.aembed_query(op.query)
            response = await self.client.query_points(
                self.collection_name,
                query=vector,
                query_filter=query_filter,
                limit=op.limit,
                offset=op.offset,
                with_payload=True,
            )
            points = response.points
            scores = {p.id: float(p.score) for p in points}
        else:
            points, _ = await self.client.scroll(
                self.collection_name,
                scroll_filter=query_filter,
                limit=op.limit,
                offset=op.offset,
                with_payload=True,
            )

        return [
            SearchItem(
                namespace=tuple(p.payload["namespace"]),
                key=p.payload["key"],
                value=p.payload["value"],
                created_at=datetime.fromisoformat(p.payload["created_at"]),
                updated_at=datetime.fromisoformat(p.payload["updated_at"]),
                score=scores.get(p.id),
            )
            for p in points
        ]

    async def _do_list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        namespaces: set[tuple[str, ...]] = set()
        next_offset = None
        while True:
            points, next_offset = await self.client.scroll(
                self.collection_name,
                limit=256,
                offset=next_offset,
                with_payload=["namespace"],
            )
            namespaces.update(tuple(p.payload["namespace"]) for p in points)
            if next_offset is None:
                break

        if op.max_depth is not None:
            namespaces = {ns[: op.max_depth] for ns in namespaces}
        if op.match_conditions:
            namespaces = {
                ns for ns in namespaces if all(_matches(ns, c) for c in op.match_conditions)
            }

        return sorted(namespaces)[op.offset : op.offset + op.limit]

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        results: list[Result] = []
        for op in ops:
            if isinstance(op, GetOp):
                results.append(await self._do_get(op))
            elif isinstance(op, PutOp):
                await self._do_put(op)
                results.append(None)
            elif isinstance(op, SearchOp):
                results.append(await self._do_search(op))
            elif isinstance(op, ListNamespacesOp):
                results.append(await self._do_list_namespaces(op))
            else:
                raise TypeError(f"Unsupported store op: {type(op)}")
        return results

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        """Sync entry point, for BaseStore's non-async callers. Assumes no event
        loop is already running in this thread — calling this from inside async
        code should use `abatch`/`aget`/`aput`/`asearch` instead.
        """
        return asyncio.run(self.abatch(ops))
