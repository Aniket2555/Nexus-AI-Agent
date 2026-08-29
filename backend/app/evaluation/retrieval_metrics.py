"""Golden-set-only retrieval metrics (§2.7): both need `expected_chunk_ids`, so
neither works outside a fixed golden set — see `faithfulness.py` for the
judge-based metrics (`faithfulness`, `relevance`, `answer_correctness`) that work
on any live query with no ground truth required, and which used to live here
before Phase 8 grew the full metric suite this module was deliberately kept thin
ahead of.
"""


def context_precision(retrieved_chunk_ids: list[str], expected_chunk_ids: list[str]) -> float:
    """Of what was retrieved, how much was actually relevant?"""
    if not retrieved_chunk_ids:
        return 0.0

    expected = set(expected_chunk_ids)
    hits = sum(1 for cid in retrieved_chunk_ids if cid in expected)
    return hits / len(retrieved_chunk_ids)


def context_recall(retrieved_chunk_ids: list[str], expected_chunk_ids: list[str]) -> float:
    """Of what should have been retrieved, how much was found?"""
    if not expected_chunk_ids:
        return 1.0

    retrieved = set(retrieved_chunk_ids)
    hits = sum(1 for cid in expected_chunk_ids if cid in retrieved)
    return hits / len(expected_chunk_ids)
