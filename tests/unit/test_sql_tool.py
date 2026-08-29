import pytest

from backend.app.mcp.tools.custom.sql import UnsafeQueryError, _validate_read_only


@pytest.mark.parametrize(
    "query",
    [
        "SELECT * FROM documents",
        "  select id from documents  ",
        "WITH recent AS (SELECT * FROM documents) SELECT * FROM recent",
        "SELECT * FROM documents;",
    ],
)
def test_allows_select_and_with_statements(query):
    _validate_read_only(query)


@pytest.mark.parametrize(
    "query",
    [
        "DROP TABLE documents",
        "DELETE FROM documents",
        "UPDATE documents SET name = 'x'",
        "INSERT INTO documents VALUES (1)",
        "ALTER TABLE documents ADD COLUMN x TEXT",
        "TRUNCATE documents",
        "GRANT ALL ON documents TO public",
        "CREATE TABLE evil (id INT)",
    ],
)
def test_rejects_mutating_statements(query):
    with pytest.raises(UnsafeQueryError):
        _validate_read_only(query)


def test_rejects_statement_stacking():
    with pytest.raises(UnsafeQueryError, match="Multiple statements"):
        _validate_read_only("SELECT 1; DROP TABLE documents")


def test_rejects_mutating_keyword_hidden_inside_a_select():
    """A keyword-based check has to look at the whole statement, not just the
    leading token — a CTE or subquery could smuggle a mutation in."""
    with pytest.raises(UnsafeQueryError):
        _validate_read_only("WITH x AS (DELETE FROM documents RETURNING *) SELECT * FROM x")
