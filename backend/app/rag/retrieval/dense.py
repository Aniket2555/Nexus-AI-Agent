import uuid
from collections.abc import Sequence
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import AsyncQdrantClient, models

from backend.app.config import get_settings

# Fixed namespace so uuid5(chunk_id) is reproducible across processes and restarts.
# The builtin hash() cannot be used for point ids: PYTHONHASHSEED randomises string
# hashing per process, so re-ingesting a document would mint fresh ids every time and
# duplicate every chunk instead of upserting over it.
NEXUS_NAMESPACE = uuid.UUID("6ba7b811-9dad-11d1-80b4-00c04fd430c8")

# Local model, no per-request API cap — batched anyway so one huge PDF doesn't hold
# thousands of chunks in memory for a single encode() call.
EMBED_BATCH_SIZE = 128


class DenseRetriever:
    """Dense vector search over Qdrant, scoped per tenant.

    Async client throughout: the sync QdrantClient blocks the event loop when called
    from an async node, which matters against a 50+ concurrent user target.
    """

    def __init__(
        self,
        collection_name: str | None = None,
        embeddings: Embeddings | None = None,
    ) -> None:
        settings = get_settings()
        self.collection_name = collection_name or settings.qdrant_collection
        self.vector_size = settings.embedding_dimensions
        self.client = AsyncQdrantClient(url=settings.qdrant_url)
        # Injectable so tests can swap in a fake embeddings client instead of loading
        # the real model. HuggingFaceEmbeddings needs no API key (DECISIONS.md D7) —
        # the model is downloaded once from the HF Hub on first use and then runs
        # locally on CPU, so this constructor does not raise for missing credentials
        # the way the previous OpenAIEmbeddings default did.
        self.embeddings = embeddings or HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            encode_kwargs={"normalize_embeddings": True},  # required for COSINE distance
        )

    async def close(self) -> None:
        """Release the underlying HTTP client. See SparseRetriever.close() — same
        rationale, symmetric API."""
        await self.client.close()

    async def ensure_collection(self) -> None:
        """Idempotent. Called once at startup — never per request."""
        if await self.client.collection_exists(self.collection_name):
            return
        await self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.vector_size, distance=models.Distance.COSINE
            ),
        )
        # Payload indexes must exist before the collection carries real data —
        # without them every tenant-filtered search degrades to a full scan.
        for field in ("tenant_id", "metadata.doc_id"):
            await self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    async def delete_document(self, doc_id: str) -> None:
        """Drop every chunk of a document before re-ingesting it.

        uuid5 ids make re-ingest overwrite chunk-for-chunk, but a shorter new version
        would leave the tail of the old one behind as orphans.
        """
        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.doc_id",
                            match=models.MatchValue(value=doc_id),
                        )
                    ]
                )
            ),
        )

    async def upsert_chunks(self, chunks: Sequence[dict[str, Any]]) -> int:
        if not chunks:
            return 0
        upserted = 0
        for start in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[start : start + EMBED_BATCH_SIZE]
            vectors = await self.embeddings.aembed_documents(
                [chunk["content"] for chunk in batch]
            )
            await self.client.upsert(
                collection_name=self.collection_name,
                wait=True,
                points=[
                    models.PointStruct(
                        id=str(uuid.uuid5(NEXUS_NAMESPACE, chunk["chunk_id"])),
                        vector=vector,
                        payload={
                            "content": chunk["content"],
                            # Duplicated out of metadata so it can carry a payload index.
                            "tenant_id": chunk["metadata"]["tenant_id"],
                            "metadata": chunk["metadata"],
                        },
                    )
                    for chunk, vector in zip(batch, vectors, strict=True)
                ],
            )
            upserted += len(batch)
        return upserted

    async def get_all_chunks(self, doc_id: str, *, tenant_id: str) -> list[dict[str, Any]]:
        """Every chunk of one document, in page/chunk order — a plain filtered
        scroll, no vector search or relevance scoring involved.

        Exists for broad "list everything in the document" queries (§rag_chat.py's
        `retrieve_node`), where per-chunk cross-encoder relevance is the wrong tool:
        a reranker scores "is this chunk relevant to the question," not "does this
        chunk belong to the document the user wants full coverage of" — a document
        with N topically-distinct sections (e.g. one page per course) can legitimately
        have some of those sections score below threshold against a generic "give me
        everything" phrasing even though every one of them is exactly what's wanted.
        Bypassing rerank's relevance filter for this case, rather than lowering the
        threshold further, keeps the threshold meaningful for actual relevance
        judgments elsewhere.
        """
        points, next_offset = [], None
        query_filter = models.Filter(
            must=[
                models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id)),
                models.FieldCondition(
                    key="metadata.doc_id", match=models.MatchValue(value=doc_id)
                ),
            ]
        )
        while True:
            batch, next_offset = await self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=256,
                offset=next_offset,
                with_payload=True,
            )
            points.extend(batch)
            if next_offset is None:
                break
        chunks = [
            {
                "id": str(point.id),
                "content": point.payload.get("content", ""),
                "metadata": point.payload.get("metadata", {}),
                "score": 1.0,  # not a relevance score — this bypasses ranking entirely
            }
            for point in points
        ]
        chunks.sort(
            key=lambda c: (c["metadata"].get("page") or 0, c["metadata"].get("chunk_index") or 0)
        )
        return chunks

    async def search(
        self, query: str, *, tenant_id: str, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Tenant-scoped dense search.

        No score_threshold in Phase 1. The old default of 0.3 was a magic number
        presented as a relevance gate; it gets calibrated in Phase 2 once the eval
        harness (2.7) can measure what it actually costs in recall.
        """
        vector = await self.embeddings.aembed_query(query)
        # .search() is deprecated in qdrant-client >= 1.10 in favour of query_points().
        response = await self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=top_k,
            with_payload=True,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="tenant_id", match=models.MatchValue(value=tenant_id)
                    )
                ]
            ),
        )
        return [
            {
                "id": str(point.id),
                "content": point.payload.get("content", ""),
                "metadata": point.payload.get("metadata", {}),
                "score": float(point.score),
            }
            for point in response.points
        ]


_retriever: DenseRetriever | None = None


def get_retriever() -> DenseRetriever:
    """Process-wide singleton.

    Constructing DenseRetriever inside the retrieve node built a fresh HTTP client and
    reloaded the local sentence-transformers model into memory on every single query,
    and re-ran the collection check each time — against a <500 ms p95 retrieval target.
    """
    global _retriever
    if _retriever is None:
        _retriever = DenseRetriever()
    return _retriever
