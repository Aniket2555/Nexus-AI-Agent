from backend.app.graphs.identity import identity_from_config


def test_missing_configurable_defaults_both():
    assert identity_from_config({}) == ("default", "default")


def test_missing_keys_default_individually():
    tenant_id, user_id = identity_from_config({"configurable": {}})
    assert tenant_id == "default"
    assert user_id == "default"


def test_empty_string_user_id_falls_back_to_default():
    """The real regression: LangGraph Platform's default (no-auth) identity
    injection sends `user_id: ""` — present but empty, not missing. A plain
    `.get("user_id", "default")` lets that empty string through unchanged, which
    then fails QdrantStore's namespace validation the first time memory is
    touched. `or "default"` must catch this too, not just a missing key.
    """
    tenant_id, user_id = identity_from_config(
        {"configurable": {"tenant_id": "acme", "user_id": ""}}
    )
    assert tenant_id == "acme"
    assert user_id == "default"


def test_empty_string_tenant_id_falls_back_to_default():
    tenant_id, user_id = identity_from_config(
        {"configurable": {"tenant_id": "", "user_id": "u1"}}
    )
    assert tenant_id == "default"
    assert user_id == "u1"


def test_real_values_pass_through_unchanged():
    tenant_id, user_id = identity_from_config(
        {"configurable": {"tenant_id": "acme", "user_id": "u1"}}
    )
    assert (tenant_id, user_id) == ("acme", "u1")
