import io

import openpyxl
import pandas as pd

from backend.app.rag.ingestion.excel_processor import extract_excel_pages


def _make_xlsx(sheets: dict[str, list[list]]) -> bytes:
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, rows in sheets.items():
        ws = wb.create_sheet(name)
        for row in rows:
            ws.append(row)

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def test_schema_and_data_present_for_single_sheet():
    payload = _make_xlsx({"Budget": [["item", "cost"], ["widget", 10], ["gadget", 25]]})

    pages = extract_excel_pages("budget.xlsx", payload)

    assert len(pages) == 1
    page_number, text = pages[0]
    assert page_number == 1
    assert "Sheet: Budget" in text
    assert "item" in text
    assert "widget" in text or "10" in text


def test_numeric_columns_get_a_statistical_summary():
    payload = _make_xlsx({"Sheet1": [["item", "cost"], ["a", 10], ["b", 20], ["c", 30]]})

    _, text = extract_excel_pages("f.xlsx", payload)[0]

    assert "Statistical summary" in text
    assert "mean" in text


def test_one_page_per_sheet_in_workbook_order():
    payload = _make_xlsx({"First": [["a"], [1]], "Second": [["b"], [2]]})

    pages = extract_excel_pages("f.xlsx", payload)

    assert [p for p, _ in pages] == [1, 2]
    assert "Sheet: First" in pages[0][1]
    assert "Sheet: Second" in pages[1][1]


def test_empty_sheet_produces_no_page():
    payload = _make_xlsx({"Empty": [], "Real": [["a"], [1]]})

    pages = extract_excel_pages("f.xlsx", payload)

    assert len(pages) == 1
    assert "Real" in pages[0][1]


def test_csv_is_treated_as_a_single_implicit_sheet():
    df = pd.DataFrame({"item": ["widget"], "cost": [10]})
    payload = df.to_csv(index=False).encode("utf-8")

    pages = extract_excel_pages("f.csv", payload)

    assert len(pages) == 1
    assert "Sheet: Sheet1" in pages[0][1]
    assert "widget" in pages[0][1]
