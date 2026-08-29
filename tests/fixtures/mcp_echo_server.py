"""A minimal real MCP server for tests/integration/test_mcp_client.py — needs no
Node.js/npx (this project's environment has neither), just the `mcp` Python SDK,
so the integration test can exercise a real stdio transport end to end.
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("echo")


@mcp.tool()
def echo(text: str) -> str:
    """Echo back the given text."""
    return text


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@mcp.tool()
def fail(message: str) -> str:
    """Always raises, to exercise the client's tool-error content handling."""
    raise RuntimeError(message)


if __name__ == "__main__":
    mcp.run(transport="stdio")
