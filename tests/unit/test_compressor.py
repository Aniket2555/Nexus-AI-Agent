from backend.app.rag.processing import compressor


def test_context_within_budget_is_returned_unchanged():
    result = compressor.compress_context("short context", "q", token_budget=1000)

    assert result.method == "none"
    assert result.text == "short context"
    assert result.original_tokens == result.compressed_tokens


def test_falls_back_to_extractive_when_llmlingua_unavailable(monkeypatch):
    monkeypatch.setattr(compressor, "_load_llmlingua", lambda: None)

    long_context = " ".join([f"Sentence number {i} about NEXUS retrieval." for i in range(200)])
    result = compressor.compress_context(long_context, "NEXUS retrieval", token_budget=50)

    assert result.method == "extractive"
    assert result.compressed_tokens <= result.original_tokens
    assert result.compressed_tokens > 0


def test_extractive_prefers_query_relevant_sentences():
    context = "The weather today is sunny. NEXUS uses Qdrant for dense retrieval. Bananas are a good source of potassium."

    compressed = compressor._extractive_compress(context, "what does nexus use for retrieval?", 15)

    assert "Qdrant" in compressed
    assert "Bananas" not in compressed


def test_extractive_preserves_original_sentence_order():
    context = "First sentence about NEXUS. Second sentence about NEXUS. Third sentence about NEXUS."

    compressed = compressor._extractive_compress(context, "NEXUS", 100)

    assert compressed.index("First") < compressed.index("Second") < compressed.index("Third")


def test_extractive_handles_empty_context():
    assert compressor._extractive_compress("", "q", 100) == ""
