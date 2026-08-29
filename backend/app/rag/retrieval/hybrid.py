from typing import Any

from backend.app.rag.retrieval.dense import DenseRetriever
from backend.app.rag.retrieval.sparse import SparseRetriever

RRF_K = 60


class HybridRetriever:
    """Reciprocal Rank Fusion of dense (Qdrant) and sparse (Elasticsearch) search.

    RRF combines *ranks*, not raw scores — cosine similarity and BM25 scores live on
    incomparable scales, so fusing the two directly would let whichever retriever
    happens to produce larger numbers dominate regardless of relevance.
    """

    def __init__(
        self,
        dense: DenseRetriever | None = None,
        sparse: SparseRetriever | None = None,
    ) -> None:
        self.dense = dense or DenseRetriever()
        self.sparse = sparse or SparseRetriever()

    async def search(
        self, query: str, *, tenant_id: str, top_k: int = 5
    ) -> list[dict[str, Any]]:
        # Over-fetch from each side: RRF's value is in letting a chunk both
        # retrievers ranked respectably outrank one either side ranked highest, and
        # that only works if there's a pool of candidates larger than top_k to fuse.
        fetch_k = top_k * 2
        dense_results = await self.dense.search(query, tenant_id=tenant_id, top_k=fetch_k)
        sparse_results = await self.sparse.search(query, tenant_id=tenant_id, top_k=fetch_k)

        # Keyed on metadata.chunk_id, never a store's native id — Qdrant mints a
        # uuid5 and Elasticsearch uses the raw chunk_id directly, so the same chunk
        # has two different native ids across the two stores. Keying fusion on the
        # native id would treat the "same" chunk as two different documents and
        # defeat the entire point of RRF.
        scores: dict[str, float] = {}
        sources: dict[str, set[str]] = {}
        docs_by_id: dict[str, dict[str, Any]] = {}

        for rank, doc in enumerate(dense_results):
            chunk_id = doc["metadata"].get("chunk_id", doc["id"])
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
            sources.setdefault(chunk_id, set()).add("dense")
            docs_by_id.setdefault(chunk_id, doc)

        for rank, doc in enumerate(sparse_results):
            chunk_id = doc["metadata"].get("chunk_id", doc["id"])
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
            sources.setdefault(chunk_id, set()).add("sparse")
            docs_by_id.setdefault(chunk_id, doc)

        ranked_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:top_k]
        results = []
        for chunk_id in ranked_ids:
            doc = dict(docs_by_id[chunk_id])
            doc["rrf_score"] = round(scores[chunk_id], 6)
            doc["rrf_sources"] = sorted(sources[chunk_id])
            results.append(doc)
        return results
