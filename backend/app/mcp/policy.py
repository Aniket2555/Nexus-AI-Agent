import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import redis.asyncio as redis
from langchain_core.tools import BaseTool

from backend.app.config import get_settings
from backend.app.observability.metrics import apush_metrics, rate_limit_tokens_remaining


class ToolNotAllowedError(Exception):
    """An agent attempted a tool outside its allowlist (§5.1/§5.5)."""


class RateLimitExceededError(Exception):
    """A (agent, tool) pair exceeded its configured rate limit."""


class ToolTimeoutError(Exception):
    """A tool call exceeded its timeout."""


@dataclass
class AgentToolPolicy:
    """Per-agent tool access rules — enforced here, in code every tool call passes
    through, not by asking the model nicely in a system prompt (§5.1: "Enforced in
    policy.py, not by prompt").
    """

    agent_name: str
    allowed_tools: frozenset[str]
    write_tools: frozenset[str] = field(default_factory=frozenset)

    def check_allowed(self, tool_name: str) -> None:
        if tool_name not in self.allowed_tools:
            raise ToolNotAllowedError(
                f"Agent {self.agent_name!r} is not allowed to call tool {tool_name!r}. "
                f"Allowed: {sorted(self.allowed_tools)}"
            )

    def requires_approval(self, tool_name: str) -> bool:
        return tool_name in self.write_tools


_TOKEN_BUCKET_SCRIPT = """
local bucket_key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_per_second = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local requested = tonumber(ARGV[4])

local bucket = redis.call('HMGET', bucket_key, 'tokens', 'timestamp')
local tokens = tonumber(bucket[1])
local timestamp = tonumber(bucket[2])

if tokens == nil then
  tokens = capacity
  timestamp = now
end

local elapsed = math.max(0, now - timestamp)
tokens = math.min(capacity, tokens + elapsed * refill_per_second)

local allowed = 0
if tokens >= requested then
  tokens = tokens - requested
  allowed = 1
end

redis.call('HMSET', bucket_key, 'tokens', tokens, 'timestamp', now)
redis.call('EXPIRE', bucket_key, 3600)
-- tostring(tokens), not the bare number: Redis's Lua-to-RESP conversion floors
-- numeric table elements to integers, which would silently truncate a
-- fractional remaining-tokens value (e.g. 2.5 -> 2) before Python ever sees it.
return {allowed, tostring(tokens)}
"""


def _parse_rate_key(key: str) -> tuple[str, str]:
    """Split `enforce_and_invoke`'s `f"tool_rate:{agent_name}:{tool.name}"` key
    back into its two label values for the Gauge below.

    `.allow()` also accepts an arbitrary caller-supplied key (it's plain,
    graph-agnostic Python — see `enforce_and_invoke`'s docstring), so a key that
    doesn't follow that convention degrades to a single "unknown" `tool_name`
    label rather than raising.
    """
    parts = key.split(":", 2)
    if len(parts) == 3 and parts[0] == "tool_rate":
        return parts[1], parts[2]
    return key, "unknown"


class RedisRateLimiter:
    """Token-bucket rate limiting, backed by Redis so the limit is shared across
    processes — an in-memory counter wouldn't be, and this project already runs
    FastAPI and `langgraph dev` as separate processes (§4.1's cache-placement note
    hit the same issue).
    """

    def __init__(
        self,
        redis_client: "redis.Redis | None" = None,
        capacity: int | None = None,
        refill_per_second: float | None = None,
    ) -> None:
        settings = get_settings()
        self._redis = redis_client
        self.capacity = capacity if capacity is not None else settings.tool_rate_limit_capacity
        self.refill_per_second = (
            refill_per_second
            if refill_per_second is not None
            else settings.tool_rate_limit_refill_per_second
        )

    @property
    def redis(self) -> "redis.Redis":
        if self._redis is None:
            self._redis = redis.from_url(get_settings().redis_url)
        return self._redis

    async def allow(self, key: str, tokens: float = 1.0) -> bool:
        allowed_raw, tokens_remaining_raw = await self.redis.eval(
            _TOKEN_BUCKET_SCRIPT,
            1,
            key,
            self.capacity,
            self.refill_per_second,
            time.time(),
            tokens,
        )

        agent_name, tool_name = _parse_rate_key(key)
        rate_limit_tokens_remaining.labels(agent_name, tool_name).set(
            float(tokens_remaining_raw)
        )
        await apush_metrics("nexus_rate_limiter")

        return bool(int(allowed_raw))

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()


async def enforce_and_invoke(
    tool: BaseTool,
    tool_input: dict[str, Any],
    policy: AgentToolPolicy,
    limiter: RedisRateLimiter,
    *,
    timeout: float | None = None,
) -> Any:
    """The one path a tool call should go through: allowlist check, then rate
    limit, then a timeout-bounded invocation. Approval-gating (§5.5's write-action
    confirmation) is deliberately not folded in here — `interrupt()` only makes
    sense called from inside a LangGraph node, and this function is plain,
    graph-agnostic Python so it's usable (and testable) outside a graph too. A
    caller wires `policy.requires_approval(tool.name)` to `interrupt()` itself —
    see `backend/app/loop_engine/human_approval.py`'s pattern (§3.3), reused as-is
    rather than building a second approval mechanism here.
    """
    policy.check_allowed(tool.name)

    settings = get_settings()
    rate_key = f"tool_rate:{policy.agent_name}:{tool.name}"
    if not await limiter.allow(rate_key):
        raise RateLimitExceededError(
            f"Rate limit exceeded for agent {policy.agent_name!r} calling tool {tool.name!r}"
        )

    call_timeout = timeout if timeout is not None else settings.tool_call_timeout_seconds
    try:
        return await asyncio.wait_for(tool.ainvoke(tool_input), timeout=call_timeout)
    except TimeoutError as exc:
        raise ToolTimeoutError(f"Tool {tool.name!r} timed out after {call_timeout}s") from exc
