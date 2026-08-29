from backend.app.mcp.tools.custom import python_repl as python_repl_module
from backend.app.mcp.tools.custom import shell as shell_module


async def test_python_repl_formats_successful_output(monkeypatch):
    async def fake_run(language, code, *, timeout_seconds=None):
        assert language == "python"
        return {
            "stdout": "42\n",
            "stderr": "",
            "exit_code": 0,
            "timed_out": False,
            "truncated": False,
            "success": True,
        }

    monkeypatch.setattr(python_repl_module, "run", fake_run)

    output = await python_repl_module.python_repl.ainvoke({"code": "print(42)"})

    assert "exit_code=0" in output
    assert "42" in output
    assert "stderr" not in output


async def test_python_repl_surfaces_timeout_in_the_formatted_output(monkeypatch):
    async def fake_run(language, code, *, timeout_seconds=None):
        return {
            "stdout": "",
            "stderr": "",
            "exit_code": None,
            "timed_out": True,
            "truncated": False,
            "success": False,
        }

    monkeypatch.setattr(python_repl_module, "run", fake_run)

    output = await python_repl_module.python_repl.ainvoke({"code": "while True: pass"})

    assert "timed out" in output


async def test_shell_formats_successful_output(monkeypatch):
    async def fake_run(language, code, *, timeout_seconds=None):
        assert language == "shell"
        return {
            "stdout": "hi\n",
            "stderr": "",
            "exit_code": 0,
            "timed_out": False,
            "truncated": False,
            "success": True,
        }

    monkeypatch.setattr(shell_module, "run", fake_run)

    output = await shell_module.shell.ainvoke({"command": "echo hi"})

    assert "hi" in output
    assert "exit_code=0" in output


async def test_shell_reports_truncated_output(monkeypatch):
    async def fake_run(language, code, *, timeout_seconds=None):
        return {
            "stdout": "aaaaaaaaaa",
            "stderr": "",
            "exit_code": 0,
            "timed_out": False,
            "truncated": True,
            "success": True,
        }

    monkeypatch.setattr(shell_module, "run", fake_run)

    output = await shell_module.shell.ainvoke({"command": "yes"})

    assert "truncated" in output
