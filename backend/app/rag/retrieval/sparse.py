from typing import Any

from elasticsearch import AsyncElasticsearch

from backend.app.config import get_settings


class SparseRetriever:
    """BM25 keyword search over Elasticsearch, scoped per tenant.

    Mirrors DenseRetriever's method shape (ensure_index/upsert_chunks/
    delete_document/search) so HybridRetriever can drive both without
    special-casing either. Chunk ids are used directly as Elasticsearch document
    ids (no uuid5 needed — ES ids are arbitrary strings), which means dense and
    sparse mint *different* native ids for the same chunk; hybrid.py's fusion keys
    on metadata.chunk_id, not either store's own id, to account for this.
    """

    def __init__(self, index_name: str | None = None) -> None:
        settings = get_settings()
        self.index_name = index_name or settings.elasticsearch_index
        self.client = AsyncElasticsearch(settings.elasticsearch_url)

    async def close(self) -> None:
        await self.client.close()

    async def ensure_index(self) -> None:
        if await self.client.indices.exists(index=self.index_name):
            return
        await self.client.indices.create(
            index=self.index_name,
            mappings={
                "properties": {
                    "content": {"type": "text"},
                    "tenant_id": {"type": "keyword"},
                    "metadata": {"type": "object"},
                }
            },
        )

    async def delete_document(self, doc_id: str) -> None:
        await self.client.delete_by_query(
            index=self.index_name,
            query={"term": {"metadata.doc_id": doc_id}},
            conflicts="proceed",
        )

    async def upsert_chunks(self, chunks: list[dict[str, Any]]) -> int:
        if not chunks:
            return 0
        for chunk in chunks:
            await self.client.index(
                index=self.index_name,
                id=chunk["chunk_id"],
                document={
                    "content": chunk["content"],
                    "tenant_id": chunk["metadata"]["tenant_id"],
                    "metadata": chunk["metadata"],
                },
            )
        return len(chunks)

    async def search(
        self, query: str, *, tenant_id: str, top_k: int = 5
    ) -> list[dict[str, Any]]:
        response = await self.client.search(
            index=self.index_name,
            query={
                "bool": {
                    "must": [{"match": {"content": query}}],
                    "filter": [{"term": {"tenant_id": tenant_id}}],
                }
            },
            size=top_k,
        )
        return [
            {
                "id": hit["_id"],
                "content": hit["_source"].get("content", ""),
                "metadata": hit["_source"].get("metadata", {}),
                "score": float(hit["_score"]),
            }
            for hit in response["hits"]["hits"]
        ]


_sparse_retriever: SparseRetriever | None = None


def get_sparse_retriever() -> SparseRetriever:
    """Process-wide singleton — same reasoning as dense.py's get_retriever():
    avoid reopening an AsyncElasticsearch client and re-running ensure_index()
    on every call site that needs one.
    """
    global _sparse_retriever
    if _sparse_retriever is None:
        _sparse_retriever = SparseRetriever()
    return _sparse_retriever
