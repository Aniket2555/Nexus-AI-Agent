from typing import Any

import numpy as np

from backend.app.config import get_settings


def deduplicate(docs: list[dict[str, Any]], embeddings: list[list[float]]) -> list[dict[str, Any]]:
    """Drop near-duplicate chunks by cosine similarity of their embeddings.

    `docs` and `embeddings` must be the same length and order — the caller (rerank or
    generate node) is responsible for producing the embedding for each surviving
    chunk. Kept greedily in `docs` order, so higher-ranked chunks always win over
    later near-duplicates, matching the ranking already established by rerank.

    This runs post-rerank on a handful of chunks (top_k, typically 5), not over the
    corpus at ingest time — the O(n^2) comparison is fine at this size.
    """
    if len(docs) != len(embeddings):
        raise ValueError(f"docs ({len(docs)}) and embeddings ({len(embeddings)}) length mismatch")

    if not docs:
        return []

    threshold = get_settings().dedup_similarity_threshold
    matrix = np.array(embeddings, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = matrix / norms

    kept_indices = []
    for i in range(len(docs)):
        is_duplicate = any(
            float(np.dot(normalized[i], normalized[j])) >= threshold for j in kept_indices
        )
        if not is_duplicate:
            kept_indices.append(i)

    return [docs[i] for i in kept_indices]
