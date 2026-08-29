from backend.app.rag.processing.broad_query import is_broad_document_query


def test_detects_all_content_phrasing():
    assert is_broad_document_query("write all the content in the pdf")
    assert is_broad_document_query("list all the course and there content")


def test_detects_everything_phrasing():
    assert is_broad_document_query("list everything in the document")
    assert is_broad_document_query("tell me everything")


def test_detects_entire_whole_complete_full_phrasing():
    assert is_broad_document_query("summarize the entire pdf")
    assert is_broad_document_query("give me the whole document")
    assert is_broad_document_query("what's the complete content of this file")


def test_does_not_flag_a_specific_question():
    """The real bug this whole module exists to fix (§reranker.py) is a per-chunk
    relevance scorer being the wrong tool for full-coverage queries — it must not
    fire on an ordinary, specific question, or every question would bypass relevance
    filtering and defeat the point of reranking.
    """
    assert not is_broad_document_query("what is Course 1 about?")
    assert not is_broad_document_query("how many lectures does Course 3 have?")
    assert not is_broad_document_query("what is LangGraph used for?")


def test_case_insensitive():
    assert is_broad_document_query("WRITE ALL THE CONTENT IN THE PDF")
