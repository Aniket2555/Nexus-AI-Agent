"""Unit tests for backend/app/api/middleware/rate_limit.py.

`RedisRateLimiter` itself is already covered against a fake in
tests/unit/test_policy.py; here we only need a fake `.allow()` to verify
`enforce_api_rate_limit`'s own logic (key construction, 429 on deny) — no
live Redis, same "construction never requires a reachable Redis" property
test_policy.py already locks in for the underlying primitive.
"""

import pytest
from fastapi import HTTPException

from backend.app.api.middleware.auth import Principal
from backend.app.api.middleware.rate_limit import _rate_limit_key as rate_limit_key
from backend.app.api.middleware.rate_limit import (
    enforce_api_rate_limit,
    get_api_rate_limiter,
)
from backend.app.mcp.policy import RedisRateLimiter


class _FakeLimiter:
    def __init__(self, allowed: bool = True):
        self.allowed = allowed
        self.calls = []

    async def allow(self, key: str, tokens: float = 1.0) -> bool:
        self.calls.append(key)
        return self.allowed


def _principal(tenant_id: str = "acme", api_key: str | None = "sk-good") -> Principal:
    return Principal(tenant_id=tenant_id, user_id="u1", api_key=api_key)


def test_rate_limit_key_is_built_from_the_api_key_when_present():
    key = rate_limit_key(_principal(tenant_id="acme", api_key="sk-good"))
    assert key == "api_rate:sk-good"


def test_rate_limit_key_falls_back_to_tenant_id_on_the_debug_bypass_path():
    """api_key is None only on auth.py's DEBUG-bypass path — must still get a
    (shared, tenant-scoped) bucket rather than skipping rate limiting."""
    key = rate_limit_key(_principal(tenant_id="default", api_key=None))
    assert key == "api_rate:default"


async def test_allows_the_request_through_when_the_bucket_has_capacity():
    principal = _principal()
    limiter = _FakeLimiter(allowed=True)

    result = await enforce_api_rate_limit(principal, limiter)

    assert result == principal
    assert limiter.calls == ["api_rate:sk-good"]


async def test_raises_429_with_a_clear_json_body_when_the_bucket_is_empty():
    principal = _principal(tenant_id="acme")
    limiter = _FakeLimiter(allowed=False)

    with pytest.raises(HTTPException) as exc_info:
        await enforce_api_rate_limit(principal, limiter)

    assert exc_info.value.status_code == 429
    assert isinstance(exc_info.value.detail, str)
    assert "acme" in exc_info.value.detail


async def test_two_different_tenants_are_rate_limited_independently():
    limiter = _FakeLimiter(allowed=True)

    await enforce_api_rate_limit(_principal(api_key="sk-acme"), limiter)
    await enforce_api_rate_limit(_principal(api_key="sk-globex"), limiter)

    assert limiter.calls == ["api_rate:sk-acme", "api_rate:sk-globex"]


def test_get_api_rate_limiter_is_a_process_wide_singleton():
    from backend.app.api.middleware import rate_limit as rate_limit_module

    rate_limit_module._limiter = None
    try:
        first = get_api_rate_limiter()
        second = get_api_rate_limiter()

        assert first is second
        assert isinstance(first, RedisRateLimiter)
    finally:
        rate_limit_module._limiter = None
