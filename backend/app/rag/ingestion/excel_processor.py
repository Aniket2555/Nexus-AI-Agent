import io

import pandas as pd


def extract_excel_pages(filename: str, payload: bytes) -> list[tuple[int, str]]:
    """One "page" per sheet (CSVs are a single implicit sheet), each rendered as a
    schema summary, a numeric statistical summary, and the row data itself — the
    three things §2.5 calls for: schema extraction, statistical summary, cell content.

    All three go in the same text block rather than separate chunks so that "what's
    the average of column X" and "what's in row 5" both land in a retrievable chunk
    without needing a table-aware retriever.
    """
    if filename.lower().endswith(".csv"):
        sheets = {"Sheet1": pd.read_csv(io.BytesIO(payload))}
    else:
        sheets = pd.read_excel(io.BytesIO(payload), sheet_name=None)

    pages: list[tuple[int, str]] = []
    for sheet_index, (sheet_name, df) in enumerate(sheets.items(), 1):
        if df.empty:
            continue
        pages.append((sheet_index, _render_sheet(sheet_name, df)))
    return pages


def _render_sheet(sheet_name: str, df: pd.DataFrame) -> str:
    schema = "\n".join(f"- {col} ({dtype})" for col, dtype in df.dtypes.items())
    numeric = df.select_dtypes(include="number")
    stats = numeric.describe().to_string() if not numeric.empty else "(no numeric columns)"
    rows = df.to_csv(index=False)
    return (
        f"Sheet: {sheet_name}"
        f"\n\nColumns:\n{schema}"
        f"\n\nStatistical summary:\n{stats}"
        f"\n\nData:\n{rows}"
    )
