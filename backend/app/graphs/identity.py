from langchain_core.runnables import RunnableConfig


def identity_from_config(config: RunnableConfig) -> tuple[str, str]:
    """Tenant and user come from the run config, never from graph state: state is
    client-writable, so reading identity from it would let a caller read another
    tenant's documents or another user's long-term memory.

    Shared by every graph that needs tenant/user scoping (rag_chat.py since Phase 1,
    the supervisor graph and specialists since Phase 6) so it's defined public and
    once, rather than as a private per-module helper re-imported across boundaries.
    """
    configurable = config.get("configurable") or {}
    tenant_id = configurable.get("tenant_id") or "default"
    user_id = configurable.get("user_id") or "default"
    return tenant_id, user_id
