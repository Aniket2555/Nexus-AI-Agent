"""Per-API-key rate limiting for the FastAPI-served, non-agent API (Phase 8).

Reuses `backend/app/mcp/policy.py`'s `RedisRateLimiter` as-is rather than
building a second token-bucket implementation — that primitive is already
generic over the bucket *key*; §5.1 built it keyed as
`f"tool_rate:{agent}:{tool}"` per (agent, tool) pair, this module only
changes the key shape to one bucket per resolved API key (falling back to
tenant_id for the DEBUG-bypass path, which has no real key — see auth.py).
"""

import logging
from typing import Annotated

from fastapi import Depends, HTTPException, status

from backend.app.api.middleware.auth import Principal, require_api_key
from backend.app.mcp.policy import RedisRateLimiter

logger = logging.getLogger(__name__)

_RATE_KEY_PREFIX = "api_rate"

_limiter: RedisRateLimiter | None = None


def get_api_rate_limiter() -> RedisRateLimiter:
    global _limiter
    if _limiter is None:
        _limiter = RedisRateLimiter()
    return _limiter


def _rate_limit_key(principal: Principal) -> str:
    return f"{_RATE_KEY_PREFIX}:{principal['api_key'] or principal['tenant_id']}"


async def enforce_api_rate_limit(
    principal: Annotated[Principal, Depends(require_api_key)],
    limiter: Annotated[RedisRateLimiter, Depends(get_api_rate_limiter)],
) -> Principal:
    """Authenticate (via `require_api_key`) then rate-limit the resolved
    principal. Endpoints depend on this instead of `require_api_key` directly
    when they want both checks; it returns the same `Principal`, so it's a
    drop-in replacement wherever only auth was wired before.

    Raises 429 with a JSON body (`{"detail": "..."}`, matching this project's
    existing `HTTPException(status, message)` house style — see
    `documents.py`) when the bucket for this key is empty.
    """
    if not await limiter.allow(_rate_limit_key(principal)):
        logger.info("Rate limit exceeded for tenant %r", principal["tenant_id"])
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"Rate limit exceeded for tenant {principal['tenant_id']!r}. Try again shortly.",
        )

    return principal
