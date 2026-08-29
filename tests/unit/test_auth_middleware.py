"""Unit tests for backend/app/api/middleware/auth.py.

No live Redis or Postgres needed — `require_api_key` only ever touches
`Settings.api_keys`, monkeypatched here via a fake settings object, same
pattern `tests/unit/test_network_policy.py` and `tests/unit/test_egress.py`
already use elsewhere in this suite.
"""

from typing import Annotated, Any

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from backend.app.api.middleware import auth as auth_module
from backend.app.api.middleware.auth import Principal, require_api_key


class _FakeSettings:
    def __init__(self, api_keys: dict[str, dict[str, str]] | None = None, debug: bool = True):
        self.api_keys = api_keys or {}
        self.debug = debug


def _patch_settings(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> None:
    monkeypatch.setattr(auth_module, "get_settings", lambda: _FakeSettings(**kwargs))


async def test_bypasses_auth_with_a_default_principal_when_no_keys_configured_and_debug(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_settings(monkeypatch, debug=True)

    principal = await require_api_key(api_key=None)

    assert principal == {"tenant_id": "default", "user_id": "default", "api_key": None}


async def test_rejects_every_request_when_no_keys_configured_and_debug_is_off(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_settings(monkeypatch, debug=False)

    with pytest.raises(HTTPException) as exc_info:
        await require_api_key(api_key=None)

    assert exc_info.value.status_code == 401


async def test_rejects_a_missing_header_once_keys_are_configured(monkeypatch: pytest.MonkeyPatch):
    _patch_settings(
        monkeypatch,
        api_keys={"sk-good": {"tenant_id": "acme", "user_id": "u1", "name": "n"}},
        debug=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await require_api_key(api_key=None)

    assert exc_info.value.status_code == 401


async def test_rejects_an_unrecognized_key_even_in_debug_mode(monkeypatch: pytest.MonkeyPatch):
    """The DEBUG bypass only fires when *no* keys are configured at all — once
    NEXUS_API_KEYS has even one entry, a wrong key must still be rejected."""
    _patch_settings(
        monkeypatch,
        api_keys={"sk-good": {"tenant_id": "acme", "user_id": "u1", "name": "n"}},
        debug=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await require_api_key(api_key="sk-wrong")

    assert exc_info.value.status_code == 401


async def test_resolves_the_configured_tenant_and_user_for_a_valid_key(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_settings(
        monkeypatch,
        api_keys={"sk-good": {"tenant_id": "acme", "user_id": "u1", "name": "Acme key"}},
        debug=True,
    )

    principal = await require_api_key(api_key="sk-good")

    assert principal == {"tenant_id": "acme", "user_id": "u1", "api_key": "sk-good"}


async def test_two_different_keys_resolve_to_two_different_tenants(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_settings(
        monkeypatch,
        api_keys={
            "sk-acme": {"tenant_id": "acme", "user_id": "u1", "name": "Acme"},
            "sk-globex": {"tenant_id": "globex", "user_id": "u2", "name": "Globex"},
        },
        debug=True,
    )

    acme = await require_api_key(api_key="sk-acme")
    globex = await require_api_key(api_key="sk-globex")

    assert acme["tenant_id"] == "acme"
    assert globex["tenant_id"] == "globex"
    assert acme["tenant_id"] != globex["tenant_id"]


def test_x_api_key_header_is_read_by_the_real_dependency_wiring(monkeypatch: pytest.MonkeyPatch):
    """Exercises the actual `APIKeyHeader` extraction (header name `X-API-Key`),
    not just the plain function call above."""
    _patch_settings(
        monkeypatch,
        api_keys={"sk-good": {"tenant_id": "acme", "user_id": "u1", "name": "n"}},
        debug=True,
    )

    app = FastAPI()

    @app.get("/protected")
    async def protected(principal: Annotated[Principal, Depends(require_api_key)]) -> dict:
        return dict(principal)

    client = TestClient(app)

    missing = client.get("/protected")
    assert missing.status_code == 401

    wrong = client.get("/protected", headers={"X-API-Key": "sk-wrong"})
    assert wrong.status_code == 401

    ok = client.get("/protected", headers={"X-API-Key": "sk-good"})
    assert ok.status_code == 200
    assert ok.json() == {"tenant_id": "acme", "user_id": "u1", "api_key": "sk-good"}
