from backend.app.rag.ingestion.pdf_parser import _render_table


def test_render_table_produces_markdown():
    rows = [["Name", "Amount"], ["Widget", "$10"], ["Gadget", "$25"]]
    rendered = _render_table(rows)
    lines = rendered.splitlines()

    assert lines[0] == "| Name | Amount |"
    assert lines[1] == "| --- | --- |"
    assert "| Widget | $10 |" in rendered
    assert "| Gadget | $25 |" in rendered


def test_render_table_handles_none_cells():
    """pdfplumber returns None for a merged/empty cell, not an empty string."""
    rows = [["Name", "Amount"], ["Widget", None]]
    rendered = _render_table(rows)
    assert "| Widget |  |" in rendered


def test_render_table_pads_ragged_rows():
    """A row shorter than the header (a missed merged cell) must not misalign columns."""
    rows = [["A", "B", "C"], ["only-one"]]
    rendered = _render_table(rows)
    assert "| only-one |  |  |" in rendered


def test_render_table_too_small_returns_empty():
    assert _render_table([["only header"]]) == ""
    assert _render_table([]) == ""
