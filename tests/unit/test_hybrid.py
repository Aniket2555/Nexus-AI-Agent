from backend.app.rag.retrieval.hybrid import HybridRetriever


def _doc(chunk_id: str, doc_id: str = "d") -> dict:
    return {"id": doc_id, "content": chunk_id, "metadata": {"chunk_id": chunk_id}, "score": 1.0}


class _FakeDense:
    def __init__(self, results: list[dict]) -> None:
        self.results = results
        self.calls = []

    async def search(self, query: str, *, tenant_id: str, top_k: int = 10):
        self.calls.append({"query": query, "tenant_id": tenant_id, "top_k": top_k})
        return self.results


class _FakeSparse:
    def __init__(self, results: list[dict]) -> None:
        self.results = results
        self.calls = []

    async def search(self, query: str, *, tenant_id: str, top_k: int = 10):
        self.calls.append({"query": query, "tenant_id": tenant_id, "top_k": top_k})
        return self.results


async def test_a_document_found_by_both_retrievers_outranks_a_single_source_hit():
    """The whole point of RRF: consensus between dense and sparse should win over
    a document either retriever alone ranked first."""
    dense = _FakeDense([_doc("only-dense"), _doc("both")])
    sparse = _FakeSparse([_doc("only-sparse"), _doc("both")])
    hybrid = HybridRetriever(dense=dense, sparse=sparse)

    results = await hybrid.search("q", tenant_id="acme", top_k=3)

    ids = [r["metadata"]["chunk_id"] for r in results]
    assert ids[0] == "both"


async def test_fusion_keys_on_chunk_id_not_native_store_id():
    """Dense and sparse mint different native ids for the same chunk (uuid5 vs raw
    chunk_id) — fusion must still recognise them as the same document."""
    dense = _FakeDense([_doc("shared", doc_id="qdrant-uuid-1")])
    sparse = _FakeSparse([_doc("shared", doc_id="es-id-1")])
    hybrid = HybridRetriever(dense=dense, sparse=sparse)

    results = await hybrid.search("q", tenant_id="acme", top_k=5)

    assert len(results) == 1
    assert results[0]["rrf_sources"].keys() == {"dense", "sparse"}


async def test_tenant_id_and_top_k_forwarded_to_both_retrievers():
    dense = _FakeDense([])
    sparse = _FakeSparse([])
    hybrid = HybridRetriever(dense=dense, sparse=sparse)

    await hybrid.search("q", tenant_id="acme", top_k=5)

    assert dense.calls[0]["tenant_id"] == "acme"
    assert sparse.calls[0]["tenant_id"] == "acme"
    assert dense.calls[0]["top_k"] == 10
    assert sparse.calls[0]["top_k"] == 10


async def test_result_capped_at_top_k():
    docs = [_doc(f"c{i}") for i in range(10)]
    dense = _FakeDense(docs)
    sparse = _FakeSparse([])
    hybrid = HybridRetriever(dense=dense, sparse=sparse)

    results = await hybrid.search("q", tenant_id="acme", top_k=3)

    assert len(results) == 3
