"""Exercises the real MCP client layer (§5.1) against a real MCP server —
tests/fixtures/mcp_echo_server.py, a Python server needing no Node.js/npx (this
project's environment has neither — verified directly, see servers.yaml's note).
Real stdio transport, real protocol handshake, real tool discovery and invocation.
"""

import sys
from pathlib import Path

import pytest

from backend.app.mcp.client import get_client

_SERVER_PATH = str(Path(__file__).parent.parent / "fixtures" / "mcp_echo_server.py")


@pytest.fixture
async def echo_tools():
    client = get_client(
        extra_configs={
            "echo": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [_SERVER_PATH],
            }
        }
    )
    tools = await client.get_tools(server_name="echo")
    return {t.name: t for t in tools}


async def test_get_tools_discovers_every_tool_the_server_declares(echo_tools):
    assert set(echo_tools) == {"echo", "add", "fail"}


async def test_tool_schema_is_derived_from_the_server_not_hand_written(echo_tools):
    """§5.1: "the adapter derives Pydantic args schemas from each server's declared
    schema" — verified by checking the add tool's schema matches its real Python
    signature (a: int, b: int), which this test never states by hand.
    """
    schema = echo_tools["add"].args

    assert schema["a"]["type"] == "integer"
    assert schema["b"]["type"] == "integer"


async def test_calling_a_tool_returns_the_real_servers_computation(echo_tools):
    result = await echo_tools["add"].ainvoke({"a": 3, "b": 4})

    assert result[0]["text"] == "7"


async def test_echo_tool_round_trips_arbitrary_text(echo_tools):
    result = await echo_tools["echo"].ainvoke({"text": "hello real mcp server"})

    assert result[0]["text"] == "hello real mcp server"


async def test_a_failing_tool_call_surfaces_as_content_not_an_exception(echo_tools):
    """MultiServerMCPClient's default (handle_tool_errors=True) converts a tool-side
    exception into an "Error executing tool ..." text block instead of raising —
    verified directly, since assuming either behavior without checking would be a
    real bug in anything built on top of this (a bare try/except here would never
    fire; policy.py's tests rely on knowing this).
    """
    result = await echo_tools["fail"].ainvoke({"message": "boom"})

    assert "Error executing tool fail" in result[0]["text"]
    assert "boom" in result[0]["text"]


def test_load_server_configs_omits_servers_missing_required_env(monkeypatch):
    from backend.app.mcp.client import load_server_configs

    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    configs = load_server_configs()

    assert "github" not in configs
    assert "filesystem" in configs


def test_load_server_configs_includes_github_once_token_is_set(monkeypatch):
    from backend.app.mcp.client import load_server_configs

    monkeypatch.setenv("GITHUB_TOKEN", "test-token-123")

    configs = load_server_configs()

    assert "github" in configs
    assert configs["github"]["headers"]["Authorization"] == "Bearer test-token-123"


def test_workspace_dir_defaults_when_env_var_unset(monkeypatch):
    from backend.app.mcp.client import load_server_configs

    monkeypatch.delenv("WORKSPACE_DIR", raising=False)

    configs = load_server_configs()

    assert "./data/workspace" in configs["filesystem"]["args"]


def test_workspace_dir_env_var_overrides_the_default(monkeypatch):
    from backend.app.mcp.client import load_server_configs

    monkeypatch.setenv("WORKSPACE_DIR", "/custom/workspace")

    configs = load_server_configs()

    assert "/custom/workspace" in configs["filesystem"]["args"]
