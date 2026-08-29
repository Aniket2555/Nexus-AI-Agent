"""Retrieval eval harness runner (§2.7).

    python -m backend.app.evaluation.run_eval [--report PATH] [--baseline PATH]

Scores the current pipeline (hybrid retrieve -> rerank) against the golden set fixed
over the fixed sample corpus, writes a JSON report, and — if a baseline exists —
prints the delta. Requires Qdrant and Elasticsearch reachable at the configured URLs;
does not require GROQ_API_KEY (faithfulness is skipped, not failed, without one — see
retrieval_metrics.faithfulness).
"""

import argparse
import asyncio
import json
import statistics
from pathlib import Path
from typing import Any

from backend.app.config import get_settings
from backend.app.evaluation.faithfulness import faithfulness
from backend.app.evaluation.golden_set import Corpus, GoldenQuestion, load_corpus, load_golden_set
from backend.app.evaluation.retrieval_metrics import context_precision, context_recall
from backend.app.llm.provider import get_chat_model
from backend.app.rag.processing.chunker import chunk_pages
from backend.app.rag.processing.citation_builder import format_context_block
from backend.app.rag.processing.reranker import Reranker
from backend.app.rag.retrieval.dense import DenseRetriever
from backend.app.rag.retrieval.hybrid import HybridRetriever
from backend.app.rag.retrieval.sparse import SparseRetriever

EVAL_COLLECTION = "nexus_eval_v1"


async def _seed_corpus(corpus: Corpus, dense: DenseRetriever, sparse: SparseRetriever) -> int:
    """Ingest the fixed sample corpus into eval-scoped stores. Idempotent — safe to
    run every time the harness runs rather than requiring a separate seed step.
    """
    chunks = chunk_pages(
        corpus.pages, doc_id=corpus.doc_id, source=corpus.source, tenant_id=corpus.tenant_id
    )

    await dense.delete_document(corpus.doc_id)
    await sparse.delete_document(corpus.doc_id)

    await dense.upsert_chunks(chunks)
    await sparse.upsert_chunks(chunks)

    return len(chunks)


async def _score_question(
    question: GoldenQuestion,
    tenant_id: str,
    hybrid: HybridRetriever,
    reranker: Reranker,
    judge: Any,
) -> dict[str, Any]:
    settings = get_settings()
    hybrid_docs = await hybrid.search(question.question, tenant_id=tenant_id, top_k=settings.retrieval_top_k)

    reranked = reranker.rerank(question.question, hybrid_docs)
    retrieved_ids = [d["metadata"].get("chunk_id", d["id"]) for d in reranked]

    both_sources = sum(1 for d in hybrid_docs if len(d.get("rrf_sources", {})) == 2)

    precision = context_precision(retrieved_ids, question.expected_chunk_ids)
    recall = context_recall(retrieved_ids, question.expected_chunk_ids)

    result = {
        "id": question.id,
        "question": question.question,
        "expected_chunk_ids": question.expected_chunk_ids,
        "retrieved_chunk_ids": retrieved_ids,
        "context_precision": round(precision, 4),
        "context_recall": round(recall, 4),
        "hybrid_candidates": len(hybrid_docs),
        "hybrid_dual_source_hits": both_sources,
    }

    if judge is not None:
        context = format_context_block(reranked)
        answer_response = await judge.ainvoke(
            f"Answer using only this context:\n{context}\n\nQuestion: {question.question}"
        )
        answer_text = (
            answer_response.content
            if isinstance(answer_response.content, str)
            else str(answer_response.content)
        )
        score = await faithfulness(answer_text, context, judge)
        result["faithfulness"] = score

    return result


async def run(report_path: Path, baseline_path: Path | None = None) -> dict[str, Any]:
    settings = get_settings()
    corpus = load_corpus()
    questions = load_golden_set()

    dense = DenseRetriever(collection_name=EVAL_COLLECTION)
    sparse = SparseRetriever(index_name=EVAL_COLLECTION)

    try:
        await dense.ensure_collection()
        await sparse.ensure_index()

        chunk_count = await _seed_corpus(corpus, dense, sparse)

        hybrid = HybridRetriever(dense=dense, sparse=sparse)
        reranker = Reranker()
        judge = get_chat_model() if settings.groq_api_key else None

        results = [
            await _score_question(q, corpus.tenant_id, hybrid, reranker, judge) for q in questions
        ]
    finally:
        await dense.close()
        await sparse.close()

    faithfulness_scores = [r["faithfulness"] for r in results if r.get("faithfulness") is not None]
    mean_faithfulness = round(statistics.mean(faithfulness_scores), 4) if faithfulness_scores else None

    report = {
        "corpus_chunks": chunk_count,
        "question_count": len(results),
        "mean_context_precision": round(statistics.mean(r["context_precision"] for r in results), 4),
        "mean_context_recall": round(statistics.mean(r["context_recall"] for r in results), 4),
        "mean_faithfulness": mean_faithfulness,
        "faithfulness_scored_count": len(faithfulness_scores),
        "results": results,
    }

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"Wrote {report_path} - precision={report['mean_context_precision']}"
        f" recall={report['mean_context_recall']} faithfulness={report['mean_faithfulness']}"
    )

    if baseline_path and baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        for metric in ("mean_context_precision", "mean_context_recall"):
            delta = report[metric] - baseline[metric]
            print(f"  delta {metric}: {delta:+.4f} (baseline {baseline[metric]})")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("eval_report.json"))
    parser.add_argument("--baseline", type=Path, default=Path("tests/eval/baseline.json"))
    args = parser.parse_args()

    asyncio.run(run(args.report, args.baseline))


if __name__ == "__main__":
    main()
