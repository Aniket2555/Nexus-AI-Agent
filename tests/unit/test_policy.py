import asyncio

import pytest

from backend.app.mcp.policy import (
    AgentToolPolicy,
    RateLimitExceededError,
    RedisRateLimiter,
    ToolNotAllowedError,
    ToolTimeoutError,
    enforce_and_invoke,
)


class _FakeTool:
    """Enough of BaseTool's surface for enforce_and_invoke: a name and an async
    ainvoke(). Not a real MCP tool — policy.py's job is enforcement around
    whatever tool it's handed, verified separately (against a real one) in
    tests/integration/test_mcp_client.py."""

    def __init__(self, name: str, delay: float = 0.0, result: str = "ok"):
        self.name = name
        self.delay = delay
        self.result = result
        self.calls = []

    async def ainvoke(self, tool_input: dict):
        self.calls.append(tool_input)
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.result


class _AlwaysAllowLimiter:
    async def allow(self, key: str, tokens: float = 1.0) -> bool:
        return True


class _AlwaysDenyLimiter:
    async def allow(self, key: str, tokens: float = 1.0) -> bool:
        return False


def test_check_allowed_passes_for_an_allowed_tool():
    policy = AgentToolPolicy("research", allowed_tools=frozenset({"search"}))
    policy.check_allowed("search")


def test_check_allowed_raises_for_a_disallowed_tool():
    policy = AgentToolPolicy("research", allowed_tools=frozenset({"search"}))
    with pytest.raises(ToolNotAllowedError):
        policy.check_allowed("delete_file")


def test_requires_approval_true_only_for_write_tools():
    policy = AgentToolPolicy(
        "code",
        allowed_tools=frozenset({"read_file", "write_file"}),
        write_tools=frozenset({"write_file"}),
    )

    assert policy.requires_approval("write_file") is True
    assert policy.requires_approval("read_file") is False


async def test_enforce_and_invoke_calls_an_allowed_tool():
    tool = _FakeTool("search", result="found it")
    policy = AgentToolPolicy("research", allowed_tools=frozenset({"search"}))

    result = await enforce_and_invoke(tool, {"q": "x"}, policy, _AlwaysAllowLimiter())

    assert result == "found it"
    assert tool.calls == [{"q": "x"}]


async def test_enforce_and_invoke_blocks_a_disallowed_tool_before_calling_it():
    tool = _FakeTool("delete_file")
    policy = AgentToolPolicy("research", allowed_tools=frozenset({"search"}))

    with pytest.raises(ToolNotAllowedError):
        await enforce_and_invoke(tool, {}, policy, _AlwaysAllowLimiter())

    assert tool.calls == []


async def test_enforce_and_invoke_raises_on_rate_limit():
    tool = _FakeTool("search")
    policy = AgentToolPolicy("research", allowed_tools=frozenset({"search"}))

    with pytest.raises(RateLimitExceededError):
        await enforce_and_invoke(tool, {}, policy, _AlwaysDenyLimiter())

    assert tool.calls == []


async def test_enforce_and_invoke_times_out_a_slow_tool():
    tool = _FakeTool("slow_search", delay=0.2)
    policy = AgentToolPolicy("research", allowed_tools=frozenset({"slow_search"}))

    with pytest.raises(ToolTimeoutError):
        await enforce_and_invoke(tool, {}, policy, _AlwaysAllowLimiter(), timeout=0.05)


async def test_enforce_and_invoke_allows_a_slow_tool_within_its_timeout():
    tool = _FakeTool("search", delay=0.05, result="done")
    policy = AgentToolPolicy("research", allowed_tools=frozenset({"search"}))

    result = await enforce_and_invoke(tool, {}, policy, _AlwaysAllowLimiter(), timeout=1.0)

    assert result == "done"


def test_rate_limiter_construction_never_requires_a_reachable_redis():
    """redis.asyncio's client is lazy — verified directly against a live server —
    but this locks in that constructing a RedisRateLimiter specifically doesn't
    force a connection attempt, regardless of that client-library detail."""
    limiter = RedisRateLimiter(capacity=5, refill_per_second=1.0)

    assert limiter.capacity == 5
    assert limiter.refill_per_second == 1.0
