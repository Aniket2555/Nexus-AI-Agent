from functools import lru_cache
from typing import Any

from sentence_transformers import CrossEncoder

from backend.app.config import get_settings

# BGE-reranker-v2-m3: local, free, no API key — matches the Phase 1 embeddings choice
# (DECISIONS.md D7) of keeping the whole retrieval stack key-less. The doc's other
# option, Cohere Rerank, is a hosted paid API and was dropped for the same reason
# OpenAI embeddings were: it would reintroduce a second required key.
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


@lru_cache(maxsize=2)
def _load_model(model_name: str) -> CrossEncoder:
    """Process-wide cache. Loading a cross-encoder is a multi-hundred-MB weight load —
    doing it per request would dominate the retrieval latency budget."""
    return CrossEncoder(model_name)


class Reranker:
    """Cross-encoder reranking of hybrid search results.

    A cross-encoder scores (query, passage) pairs jointly, which is far more accurate
    than the cosine/BM25 similarity used to generate candidates — but too slow to run
    over a whole corpus, which is why it only re-scores the top-N hybrid candidates
    rather than replacing retrieval.
    """

    def __init__(self, model_name: str | None = None, model: Any = None) -> None:
        settings = get_settings()
        # Injectable so tests can supply a fake model exposing .predict(pairs) instead
        # of downloading the real multi-hundred-MB cross-encoder — same pattern as
        # DenseRetriever's injectable `embeddings` in Phase 1.
        self.model = model or _load_model(model_name or DEFAULT_RERANKER_MODEL)
        self.score_threshold = settings.rerank_score_threshold
        self.top_k = settings.rerank_top_k

    def rerank(
        self,
        query: str,
        docs: list[dict[str, Any]],
        *,
        query_variants: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Re-score and re-order `docs` by joint relevance to `query`.

        Results below `score_threshold` are dropped rather than merely deprioritized —
        a hybrid candidate that scores low under the cross-encoder is very likely not
        relevant, and passing it through wastes generation-time context budget.

        `query_variants` (Phase 3's rewrite_node output) are scored too, per doc, and
        the MAX across the original query and every variant wins. This exists because
        the cross-encoder is trained to score "does this passage directly answer this
        question," which structural/meta queries ("list the content in course 1
        section") score poorly on even against the exactly-right passage (~0.03-0.11
        measured live) — a rewritten variant phrased more like a direct question
        ("What topics are covered in the Course 1 section?") scores meaningfully
        higher against the same passage without changing what's actually being asked.
        Without this, query rewriting only ever helped candidate generation (§2.1's
        merge_query_variant_results), never reranking — the exact same
        chunk-could-be-found-but-still-gets-filtered-out gap this closes.
        """
        if not docs:
            return []
        queries = [query, *(query_variants or [])]
        # One predict() call across the full (query, doc) cross-product rather than
        # one per variant — CrossEncoder.predict() batches internally, and this keeps
        # it to a single model invocation regardless of how many variants exist.
        pairs = [(q, doc["content"]) for doc in docs for q in queries]
        raw_scores = self.model.predict(pairs)
        n = len(queries)
        scored = []
        for i, doc in enumerate(docs):
            doc_scores = raw_scores[i * n : (i + 1) * n]
            # CrossEncoder.predict() already applies the model's own Sigmoid
            # activation for BAAI/bge-reranker-v2-m3 (confirmed via model.activation_fn
            # at load time) — scores arrive pre-scaled to [0, 1]. Applying a second
            # sigmoid here was tried and verified wrong: it compressed a real
            # (0.96, 0.00002) pair down to (0.72, 0.5), destroying almost all of the
            # separation between a clearly relevant and clearly irrelevant passage.
            scored.append({**doc, "rerank_score": round(float(max(doc_scores)), 4)})
        scored.sort(key=lambda d: d["rerank_score"], reverse=True)

        # force_include (retrieve_node's broad-document-query expansion, rag/
        # processing/broad_query.py) bypasses the threshold and the top_k cap
        # entirely — still scored above, for citation ordering, but never dropped
        # for scoring low. compress_node's token-budget compression is what keeps
        # an expanded document from blowing the context window, not this filter.
        forced = [d for d in scored if d.get("force_include")]
        rest = [d for d in scored if not d.get("force_include")]
        kept_rest = [d for d in rest if d["rerank_score"] >= self.score_threshold]
        selected_rest = _reserve_document_slots(kept_rest, self.top_k)

        combined = forced + selected_rest
        combined.sort(key=lambda d: d["rerank_score"], reverse=True)
        return combined


# Sources that aren't an actual retrieved document — memory_node (§4.3) merges
# these into the same candidate pool documents flow through, at metadata.source ==
# "long_term_memory" (memory/manager.py's recall()) or "knowledge_graph"
# (rag/retrieval/graph_rag.py) — everything else is a real dense/sparse chunk.
_NON_DOCUMENT_SOURCES = frozenset({"long_term_memory", "knowledge_graph"})


def _reserve_document_slots(kept: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    """Guarantee real document chunks a fair shot at the final top_k, not just
    whichever source scored highest.

    Real bug this closes: `MemoryManager.remember()` stores the raw text of every
    past turn ("human: <query>\\nai: <answer>") as a long-term memory fact, with no
    cap and no distinction between a grounded answer and a refusal. On a repeated or
    rephrased question, that fact contains the query almost verbatim and scores
    near-perfectly against it (0.98 measured live) — comfortably beating a real
    document chunk answering the same question from actual source material (0.11-0.18
    measured live for the same query). Once enough turns about one topic have
    accumulated, memory facts alone can fill every slot in `top_k`, and the source
    documents disappear from `generate_node`'s context entirely — even though they
    scored well above `score_threshold` and would otherwise have made the cut.

    This doesn't stop memory from winning on merit (a genuinely more relevant memory
    fact still outranks a weak document match within its reserved half); it only
    stops non-document sources from occupying *every* slot when real document
    matches exist and passed the threshold.
    """
    if len(kept) <= top_k:
        return kept
    doc_slots = -(-top_k // 2)  # ceil(top_k / 2) — documents get first claim on odd counts
    docs = [d for d in kept if d.get("metadata", {}).get("source") not in _NON_DOCUMENT_SOURCES]
    non_docs = [d for d in kept if d.get("metadata", {}).get("source") in _NON_DOCUMENT_SOURCES]
    if not docs or not non_docs:
        return kept[:top_k]  # only one source type present — nothing to balance
    chosen_docs = docs[:doc_slots]
    chosen_non_docs = non_docs[: top_k - len(chosen_docs)]
    combined = chosen_docs + chosen_non_docs
    combined.sort(key=lambda d: d["rerank_score"], reverse=True)
    return combined


_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    """Process-wide singleton, same rationale as the retriever singletons."""
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker
