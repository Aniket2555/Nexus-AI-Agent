from dataclasses import dataclass
from typing import Any


@dataclass
class Citation:
    """One resolved source reference, carrying the full provenance chain
    (§2.4): original document -> chunk -> retrieval score -> rerank score.
    """

    source: str
    page: int | str | None
    chunk_id: str
    retrieval_score: float | None
    rerank_score: float | None

    def format(self) -> str:
        """Renders exactly the format the system prompt asks the model to emit:
        `[Source: filename.pdf, Page 12]`.
        """
        page = self.page if self.page is not None else "?"
        return f"[Source: {self.source}, Page {page}]"


def build_citations(docs: list[dict[str, Any]]) -> list[Citation]:
    """Resolve retrieved/reranked chunks into citations.

    Reads `chunk_id` out of `metadata` — never off a store's own id — because that is
    the one field guaranteed to be present and identical across Qdrant and
    Elasticsearch results (dense.py and sparse.py mint different native ids for the
    same chunk; see hybrid.py's fusion key).
    """
    citations = []
    for doc in docs:
        metadata = doc.get("metadata", {})
        citations.append(
            Citation(
                source=metadata.get("source", "unknown"),
                page=metadata.get("page"),
                chunk_id=metadata.get("chunk_id", doc.get("id", "unknown")),
                retrieval_score=_round_or_none(doc.get("score")),
                rerank_score=_round_or_none(doc.get("rerank_score")),
            )
        )
    return citations


def _round_or_none(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


def format_context_block(docs: list[dict[str, Any]]) -> str:
    """Assemble the `[Source: ..., Page: ...]\\ncontent` blocks the generate node feeds
    to the model, in the same shape Phase 1's generate_node used inline.
    """
    if not docs:
        return "No matching content was found in the knowledge base."
    blocks = []
    for doc in docs:
        metadata = doc.get("metadata", {})
        page = metadata.get("page", "?")
        source = metadata.get("source", "unknown")
        blocks.append(f"[Source: {source}, Page: {page}]\n{doc['content']}")
    return "\n\n".join(blocks)
