from types import SimpleNamespace

from backend.app.sandbox import network_policy


def _fake_settings(**overrides):
    defaults = {
        "egress_allowed_domains": (),
        "sandbox_egress_network": "nexus-sandbox-egress",
        "sandbox_proxy_host": "nexus-sandbox-proxy",
        "sandbox_proxy_port": 3128,
    }
    return SimpleNamespace(**{**defaults, **overrides})


def test_resolve_network_denies_all_by_default(monkeypatch):
    monkeypatch.setattr(network_policy, "get_settings", lambda: _fake_settings())

    config = network_policy.resolve_network()

    assert config.network_mode == "none"
    assert config.environment == {}
    assert config.proxied is False


def test_resolve_network_uses_settings_domains_when_none_passed(monkeypatch):
    monkeypatch.setattr(
        network_policy, "get_settings", lambda: _fake_settings(egress_allowed_domains=("example.com",))
    )

    config = network_policy.resolve_network()

    assert config.proxied is True
    assert config.network_mode == "nexus-sandbox-egress"


def test_resolve_network_explicit_domains_override_settings(monkeypatch):
    monkeypatch.setattr(network_policy, "get_settings", lambda: _fake_settings())

    config = network_policy.resolve_network(("api.example.com",))

    assert config.proxied is True
    assert config.environment["HTTP_PROXY"] == "http://nexus-sandbox-proxy:3128"
    assert config.environment["NO_PROXY"] == ""


def test_render_squid_config_uses_bare_domains_only():
    text = network_policy.render_squid_config(("example.com",))
    acl_line = next(line for line in text.splitlines() if line.startswith("acl allowed_domains"))

    assert acl_line == "acl allowed_domains dstdomain example.com"


def test_render_squid_config_falls_back_to_unmatchable_acl_when_empty():
    text = network_policy.render_squid_config(())
    assert "invalid.example" in text
