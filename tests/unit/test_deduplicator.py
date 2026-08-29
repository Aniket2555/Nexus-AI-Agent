import pytest

from backend.app.rag.processing.deduplicator import deduplicate


def _doc(chunk_id: str) -> dict:
    return {"content": chunk_id, "metadata": {"chunk_id": chunk_id}}


def test_near_identical_embeddings_collapse_to_the_first():
    docs = [_doc("a"), _doc("b")]
    embeddings = [[1.0, 0.0, 0.0], [0.999, 0.001, 0.0]]

    kept = deduplicate(docs, embeddings)

    assert [d["metadata"]["chunk_id"] for d in kept] == ["a"]


def test_dissimilar_embeddings_are_both_kept():
    docs = [_doc("a"), _doc("b")]
    embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

    kept = deduplicate(docs, embeddings)

    assert [d["metadata"]["chunk_id"] for d in kept] == ["a", "b"]


def test_earlier_ranked_doc_wins_ties_broken_by_input_order():
    """Docs arrive already ranked (post-rerank); a later near-duplicate must lose to
    an earlier, higher-ranked one, never the reverse."""
    docs = [_doc("higher-ranked"), _doc("lower-ranked-duplicate")]
    embeddings = [[1.0, 0.0], [1.0, 0.0]]

    kept = deduplicate(docs, embeddings)

    assert [d["metadata"]["chunk_id"] for d in kept] == ["higher-ranked"]


def test_empty_input():
    assert deduplicate([], []) == []


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        deduplicate([_doc("a")], [[1.0], [2.0]])
