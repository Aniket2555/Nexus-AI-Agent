from backend.app.rag.processing.reranker import Reranker


class _FakeModel:
    """Deterministic stand-in for CrossEncoder — no model download needed."""

    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.pairs_seen = []

    def predict(self, pairs):
        self.pairs_seen = list(pairs)
        return self.scores


def _docs(n: int) -> list[dict]:
    return [{"content": f"doc {i}", "metadata": {"chunk_id": f"c{i}"}} for i in range(n)]


def test_scores_are_used_as_is_not_re_sigmoided():
    """Regression test: an earlier version applied a second sigmoid on top of scores
    that were already sigmoid-scaled by the model, compressing (0.96, 0.00002) down
    to (0.72, 0.5) and destroying almost all separation. predict() output must pass
    through unchanged.
    """
    model = _FakeModel([0.9613, 0.0])
    reranker = Reranker(model=model)
    reranker.score_threshold = 0.0
    reranker.top_k = 2

    out = reranker.rerank("q", _docs(2))

    assert out[0]["rerank_score"] == 0.9613
    assert out[1]["rerank_score"] == 0.0


def test_below_threshold_results_are_dropped():
    model = _FakeModel([0.9, 0.1])
    reranker = Reranker(model=model)
    reranker.score_threshold = 0.3
    reranker.top_k = 5

    out = reranker.rerank("q", _docs(2))

    assert len(out) == 1
    assert out[0]["metadata"]["chunk_id"] == "c0"


def test_results_sorted_descending_by_score():
    model = _FakeModel([0.2, 0.9, 0.5])
    reranker = Reranker(model=model)
    reranker.score_threshold = 0.0
    reranker.top_k = 5

    out = reranker.rerank("q", _docs(3))

    assert [d["rerank_score"] for d in out] == [0.9, 0.5, 0.2]


def test_result_capped_at_top_k():
    model = _FakeModel([0.9, 0.8, 0.7, 0.6])
    reranker = Reranker(model=model)
    reranker.score_threshold = 0.0
    reranker.top_k = 2

    out = reranker.rerank("q", _docs(4))

    assert len(out) == 2


def test_empty_input_returns_empty_without_calling_model():
    model = _FakeModel([])
    reranker = Reranker(model=model)

    assert reranker.rerank("q", []) == []


def test_query_variants_take_the_max_score_per_doc():
    """The real bug this closes: a meta/structural query ('list the content in
    course 1 section') scores a genuinely-relevant chunk far below threshold on its
    own, but a rewritten variant phrased as a direct question scores it well above —
    each doc should get credit for whichever phrasing scored it best, not just the
    original query.
    """
    model = _FakeModel([0.05, 0.2, 0.9, 0.1])
    reranker = Reranker(model=model)
    reranker.score_threshold = 0.15
    reranker.top_k = 5

    out = reranker.rerank("q", _docs(2), query_variants=["variant"])

    assert [d["metadata"]["chunk_id"] for d in out] == ["c1", "c0"]
    assert [d["rerank_score"] for d in out] == [0.9, 0.2]


def _doc_with_source(chunk_id: str, source: str) -> dict:
    return {"content": f"content {chunk_id}", "metadata": {"chunk_id": chunk_id, "source": source}}


def test_memory_cannot_crowd_out_every_document_slot():
    """The real bug: MemoryManager.remember() stores raw past turns as long-term
    memory facts with no cap, and a fact containing the query almost verbatim
    scores near-perfectly against a repeat/rephrase of it -- comfortably beating a
    real document chunk answering the same question from actual source material.
    Left unchecked, five memory facts can occupy all five top_k slots and a real,
    above-threshold document chunk never reaches generate_node at all.
    """
    docs = [
        _doc_with_source("doc-a", "handbook.pdf"),
        _doc_with_source("doc-b", "handbook.pdf"),
        _doc_with_source("mem-1", "long_term_memory"),
        _doc_with_source("mem-2", "long_term_memory"),
        _doc_with_source("mem-3", "long_term_memory"),
        _doc_with_source("mem-4", "long_term_memory"),
        _doc_with_source("mem-5", "long_term_memory"),
    ]
    model = _FakeModel([0.18, 0.11, 0.98, 0.86, 0.51, 0.51, 0.44])
    reranker = Reranker(model=model)
    reranker.score_threshold = 0.1
    reranker.top_k = 5

    out = reranker.rerank("q", docs)

    sources = [d["metadata"]["source"] for d in out]
    assert sources.count("handbook.pdf") == 2, (
        f"both above-threshold document chunks must survive, not just whichever scored highest overall: got {sources}"
    )
    assert len(out) == 5


def test_force_include_bypasses_threshold_and_top_k():
    """The broad-document-query fix (rag/processing/broad_query.py,
    retrieve_node's expansion): a chunk tagged force_include must survive even
    when it scores below score_threshold and even when it would otherwise be
    pushed out by the top_k cap - retrieve_node only sets this flag when it has
    already decided the whole document belongs in context, not based on this
    chunk's individual relevance.
    """
    docs = [
        {**_docs(1)[0], "force_include": True},
        _doc_with_source("doc-b", "handbook.pdf"),
        _doc_with_source("doc-c", "handbook.pdf"),
        _doc_with_source("doc-d", "handbook.pdf"),
        _doc_with_source("doc-e", "handbook.pdf"),
        _doc_with_source("doc-f", "handbook.pdf"),
    ]
    model = _FakeModel([0.01, 0.9, 0.8, 0.7, 0.6, 0.5])
    reranker = Reranker(model=model)
    reranker.score_threshold = 0.3
    reranker.top_k = 2

    out = reranker.rerank("q", docs)

    forced_ids = [d["metadata"]["chunk_id"] for d in out if d.get("force_include")]
    assert forced_ids == ["c0"], "the force_include doc must survive despite scoring 0.01"
    assert len(out) == reranker.top_k + 1, "forced doc is additive, not counted against top_k"


def test_no_variants_behaves_exactly_as_before():
    """query_variants=None (the default) must reduce to the original single-query
    behavior — this is what every pre-existing caller/test relies on.
    """
    model = _FakeModel([0.9, 0.1])
    reranker = Reranker(model=model)
    reranker.score_threshold = 0.3
    reranker.top_k = 5

    out = reranker.rerank("q", _docs(2))

    assert len(out) == 1
    assert out[0]["metadata"]["chunk_id"] == "c0"
