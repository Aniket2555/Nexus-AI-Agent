"""Simple API-key authentication for the non-agent HTTP API (Phase 8).

A deliberate choice, not a placeholder: DECISIONS.md settled on a plain
`X-API-Key` header rather than JWT/OAuth for this API — there is no browser
login flow to support, only server-to-server / script callers, and a header
lookup against a configured map is the entire mechanism a bearer-token or
signed-JWT scheme would otherwise need a library for.

`Settings.api_keys` (parsed from the `NEXUS_API_KEYS` env var — see
config.py) maps each key to the `{tenant_id, user_id, name}` it resolves to.
`require_api_key` is a FastAPI dependency that validates the header against
that map and returns the resolved principal; wire it into any endpoint that
must not run as an unauthenticated, caller-supplied tenant_id (documents.py's
upload endpoint used to take `tenant_id` as a plain, unauthenticated request
parameter — anyone could claim to be any tenant. This replaces that).

Scope boundary: this covers only the FastAPI-served, non-agent API
(documents.py, workflows.py). It deliberately does NOT add auth to
`langgraph dev` itself — that's a separate LangGraph Platform server this
project has never put behind auth (Phases.md's Phase 1 scope note), and
doing so is out of scope here. A future authenticated frontend would resolve
its own API key against this same mechanism server-side first, then pass the
*resolved* tenant_id into the graph run's `config.configurable`, the same
value documents.py already forwards to ingestion.
"""

import logging
from typing import TypedDict

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from backend.app.config import get_settings

logger = logging.getLogger(__name__)

API_KEY_HEADER_NAME = "X-API-Key"

_api_key_scheme = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)

_DEBUG_BYPASS_TENANT_ID = "default"
_DEBUG_BYPASS_USER_ID = "default"


class Principal(TypedDict):
    """The authenticated identity a request resolves to."""

    tenant_id: str
    user_id: str
    api_key: str | None


async def require_api_key(api_key: str | None = Security(_api_key_scheme)) -> Principal:
    """Validate `X-API-Key` against `settings.api_keys` and return the resolved
    principal. Raises 401 on a missing or unrecognized key.

    DEBUG-gated bypass: if no keys are configured at all (`NEXUS_API_KEYS`
    unset) *and* `settings.debug` is true, requests are let through as an
    unauthenticated "default" tenant — so a fresh local checkout without any
    key configured still works — with a clear warning logged so this never
    silently ships. The moment `NEXUS_API_KEYS` has even one entry, this
    bypass is off and every request needs a valid key, DEBUG or not.
    """
    settings = get_settings()

    if not settings.api_keys:
        if settings.debug:
            logger.warning(
                "NEXUS_API_KEYS is not configured; allowing this request through as "
                "an unauthenticated %r tenant because DEBUG=true. Set NEXUS_API_KEYS "
                "to turn on real per-key authentication before this API is reachable "
                "by anyone other than a single local developer.",
                _DEBUG_BYPASS_TENANT_ID,
            )
            return Principal(
                tenant_id=_DEBUG_BYPASS_TENANT_ID, user_id=_DEBUG_BYPASS_USER_ID, api_key=None
            )
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "No API keys are configured and DEBUG is disabled; this API rejects "
            "every request until NEXUS_API_KEYS is set.",
        )

    if not api_key or api_key not in settings.api_keys:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            f"Missing or invalid {API_KEY_HEADER_NAME!r} header.",
        )

    resolved = settings.api_keys[api_key]
    return Principal(
        tenant_id=resolved["tenant_id"], user_id=resolved["user_id"], api_key=api_key
    )
