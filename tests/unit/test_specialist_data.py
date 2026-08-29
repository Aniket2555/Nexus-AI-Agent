from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from backend.app.graphs.specialists import data as data_module
from backend.app.graphs.specialists.data import _extract_sql, execute_node
from backend.app.mcp.tools.custom.sql import UnsafeQueryError


def test_extract_sql_strips_markdown_fences():
    assert _extract_sql("```sql\nSELECT * FROM documents\n```") == "SELECT * FROM documents"


def test_extract_sql_passes_through_bare_sql_unchanged():
    assert _extract_sql("SELECT 1") == "SELECT 1"


async def test_get_schema_summary_groups_columns_by_table(monkeypatch):
    async def fake_query(query=None):
        return [
            {"table_name": "documents", "column_name": "id", "data_type": "integer"},
            {"table_name": "documents", "column_name": "name", "data_type": "text"},
        ]

    monkeypatch.setattr(data_module, "run_readonly_query", fake_query)

    summary = await data_module.get_schema_summary()

    assert "documents: id (integer), name (text)" in summary


async def test_get_schema_summary_handles_no_tables(monkeypatch):
    async def fake_query(query=None):
        return []

    monkeypatch.setattr(data_module, "run_readonly_query", fake_query)

    summary = await data_module.get_schema_summary()

    assert "no tables" in summary.lower()


async def test_execute_node_reports_unsafe_query_as_an_error_not_an_exception(monkeypatch):
    async def fake_query(query=None):
        raise UnsafeQueryError("Only SELECT/WITH statements are allowed.")

    monkeypatch.setattr(data_module, "run_readonly_query", fake_query)

    result = await execute_node({"sql_query": "DROP TABLE documents"})

    assert result["rows"] == []
    assert "Only SELECT" in result["error"]


async def test_execute_node_returns_rows_on_success(monkeypatch):
    async def fake_query(query=None):
        return [{"count": 42}]

    monkeypatch.setattr(data_module, "run_readonly_query", fake_query)

    result = await execute_node({"sql_query": "SELECT count(*) FROM documents"})

    assert result["rows"] == [{"count": 42}]
    assert result["error"] is None


async def test_run_data_returns_a_specialist_result(monkeypatch):
    async def fake_query(query=None):
        if "information_schema" in query:
            return [{"table_name": "documents", "column_name": "id", "data_type": "integer"}]
        return [{"count": 42}]

    monkeypatch.setattr(data_module, "run_readonly_query", fake_query)

    messages = [AIMessage("SELECT count(*) FROM documents"), AIMessage("There are 42 rows.")]
    model = GenericFakeChatModel(messages=iter(messages))
    monkeypatch.setattr(data_module, "get_chat_model", lambda temperature=0.1: model)

    result = await data_module.run_data("task-1", "how many documents are there?")

    assert result["task_id"] == "task-1"
    assert result["specialist"] == "data"
    assert result["success"] is True
    assert result["summary"] == "There are 42 rows."
    assert result["details"]["row_count"] == 1
