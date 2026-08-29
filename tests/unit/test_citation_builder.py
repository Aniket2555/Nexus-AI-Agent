from backend.app.rag.processing.citation_builder import (
    build_citations,
    format_context_block,
)


def _doc(source="handbook.pdf", page=12, chunk_id="acme:handbook.pdf:p12:c0", **extra):
    return {
        "id": "point-id",
        "content": "some content",
        "metadata": {"source": source, "page": page, "chunk_id": chunk_id},
        **extra,
    }


def test_citation_resolves_from_metadata_not_native_id():
    """The Phase 1 regression this locks in: citations must read chunk_id out of
    metadata, never fall back to a store's native point id."""
    citations = build_citations([_doc()])

    assert citations[0].chunk_id == "acme:handbook.pdf:p12:c0"
    assert citations[0].chunk_id != "point-id"


def test_citation_carries_retrieval_and_rerank_scores():
    citations = build_citations([_doc(score=0.9123456, rerank_score=0.87654)])

    assert citations[0].retrieval_score == 0.9123
    assert citations[0].rerank_score == 0.8765


def test_citation_format_matches_system_prompt_contract():
    citations = build_citations([_doc(source="handbook.pdf", page=12)])
    assert citations[0].format() == "[Source: handbook.pdf, Page 12]"


def test_missing_page_renders_as_question_mark():
    citations = build_citations([_doc(page=None)])
    assert citations[0].format() == "[Source: handbook.pdf, Page ?]"


def test_format_context_block_empty_docs():
    assert format_context_block([]) == "No matching content was found in the knowledge base."


def test_format_context_block_includes_source_and_page():
    block = format_context_block([_doc()])
    assert "[Source: handbook.pdf, Page: 12]" in block
    assert "some content" in block
