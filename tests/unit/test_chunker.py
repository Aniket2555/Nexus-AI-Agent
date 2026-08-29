from backend.app.rag.processing.chunker import chunk_pages

KW = {"doc_id": "acme:handbook.pdf", "source": "handbook.pdf", "tenant_id": "acme"}


def test_page_number_survives_into_metadata():
    """The regression that made every citation say 'Page 1'."""
    chunks = chunk_pages([(1, "alpha text"), (7, "beta text")], **KW)
    assert {c["metadata"]["page"] for c in chunks} == {1, 7}


def test_oversized_page_is_split_rather_than_emitted_whole():
    """A single huge paragraph must still be split.

    Emitted whole it would exceed the embedding model's 8191-token input limit and
    fail the entire ingest, which is what the hand-rolled paragraph splitter did.
    """
    page = " ".join(["word"] * 20000)
    chunks = chunk_pages([(1, page)], **KW)

    assert len(chunks) > 1
    assert all(len(c["content"]) < 20000 for c in chunks)


def test_chunk_ids_are_stable_across_runs():
    """Ingestion idempotency depends on this: uuid5(chunk_id) must not drift."""
    page = [(1, "sentence. " * 500)]
    assert [c["chunk_id"] for c in chunk_pages(page, **KW)] == [
        c["chunk_id"] for c in chunk_pages(page, **KW)
    ]


def test_chunk_id_is_reachable_from_metadata():
    """Citations read chunk_id out of metadata, not off the chunk top level."""
    chunks = chunk_pages([(3, "some content")], **KW)
    assert chunks[0]["metadata"]["chunk_id"] == chunks[0]["chunk_id"]
    assert chunks[0]["metadata"]["chunk_id"].endswith(":p3:c0")


def test_blank_pages_produce_no_chunks():
    """Scanned PDFs are full of empty text layers; they must not become empty vectors."""
    assert chunk_pages([(1, "   \n\n  ")], **KW) == []
    assert len(chunk_pages([(1, "  "), (2, "real content")], **KW)) == 1
