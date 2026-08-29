from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from backend.app.graphs.specialists import code as code_module
from backend.app.graphs.specialists.code import _extract_code_block, validate_edge, validate_node


def test_extract_code_block_pulls_language_and_body():
    text = "Here's the approach.\n\n```python\ndef add(a, b):\n    return a + b\n```\n"

    language, body = _extract_code_block(text)

    assert language == "python"
    assert body == "def add(a, b):\n    return a + b"


def test_extract_code_block_returns_empty_when_no_fence_present():
    assert _extract_code_block("just prose, no code") == ("", "")


async def test_validate_node_accepts_syntactically_valid_python():
    result = await validate_node({"response_text": "```python\ndef f():\n    return 1\n```"})

    assert result["syntax_valid"] is True
    assert result["syntax_error"] is None


async def test_validate_node_rejects_invalid_python_syntax():
    result = await validate_node({"response_text": "```python\ndef f(:\n    return 1\n```"})

    assert result["syntax_valid"] is False
    assert result["syntax_error"] is not None


async def test_validate_node_does_not_execute_anything():
    """Regression guard for the deliberate scope decision: even destructive-looking
    code must only be parsed, never run."""
    dangerous = "```python\nimport os\nos.system('this must never actually run')\n```"

    result = await validate_node({"response_text": dangerous})

    assert result["syntax_valid"] is True
    assert "os.system" in result["code"]


async def test_validate_node_skips_syntax_check_for_non_python():
    result = await validate_node({"response_text": "```javascript\nconst x = 1;\n```"})

    assert result["language"] == "javascript"
    assert result["syntax_valid"] is True


async def test_validate_node_reports_missing_code_block():
    result = await validate_node({"response_text": "I did not include any code."})

    assert result["syntax_valid"] is False
    assert "No code block" in result["syntax_error"]


def test_validate_edge_routes_valid_python_to_execute():
    state = {"language": "python", "syntax_valid": True}

    assert validate_edge(state) == "execute"


def test_validate_edge_skips_execution_for_invalid_syntax():
    state = {"language": "python", "syntax_valid": False}

    assert validate_edge(state) == "skip"


def test_validate_edge_skips_execution_for_non_python():
    state = {"language": "javascript", "syntax_valid": True}

    assert validate_edge(state) == "skip"


async def test_run_code_executes_syntax_valid_python_via_the_sandbox(monkeypatch):
    """§7.1: real execution now happens — mock `run_python` (the sandbox call)
    rather than Docker itself, so this stays a fast unit test; the real,
    Docker-backed round trip is covered in tests/integration.
    """
    model = GenericFakeChatModel(
        messages=iter([AIMessage("```python\ndef add(a, b):\n    return a + b\n```")])
    )
    monkeypatch.setattr(code_module, "get_chat_model", lambda: model)

    async def fake_run_python(code):
        assert "def add" in code
        return {
            "stdout": "",
            "stderr": "",
            "exit_code": 0,
            "timed_out": False,
            "truncated": False,
            "success": True,
        }

    monkeypatch.setattr(code_module, "run_python", fake_run_python)

    result = await code_module.run_code("task-1", "write an add function")

    assert result["task_id"] == "task-1"
    assert result["specialist"] == "code"
    assert result["success"] is True
    assert result["details"]["executed"] is True
    assert "Phase 7" in result["details"]["note"]


async def test_run_code_does_not_execute_syntax_invalid_python(monkeypatch):
    model = GenericFakeChatModel(messages=iter([AIMessage("```python\ndef f(:\n```")]))
    monkeypatch.setattr(code_module, "get_chat_model", lambda: model)

    async def fail_if_called(code):
        raise AssertionError("run_python must not be called for invalid syntax")

    monkeypatch.setattr(code_module, "run_python", fail_if_called)

    result = await code_module.run_code("task-1", "write broken code")

    assert result["success"] is False
    assert result["details"]["executed"] is False
