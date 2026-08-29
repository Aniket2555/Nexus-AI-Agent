from backend.app.llm import provider


def test_cache_configuration_degrades_gracefully_when_redis_is_unreachable(monkeypatch):
    """RedisCache.__init__ connects eagerly (verified directly against a real
    server) — an unreachable Redis must not prevent get_chat_model() from working,
    only skip the cache."""
    monkeypatch.setattr(provider, "_cache_configured", False)
    monkeypatch.setenv("REDIS_URL", "redis://localhost:1")
    provider.get_settings.cache_clear()

    provider._ensure_llm_cache_configured()

    assert provider._cache_configured is True

    provider.get_settings.cache_clear()


def test_cache_configuration_only_runs_once_per_process(monkeypatch):
    calls = []
    monkeypatch.setattr(provider, "_cache_configured", False)

    def _fake_redis_cache(*args, **kwargs):
        calls.append(1)
        raise ConnectionError("simulated")

    monkeypatch.setattr(provider, "MetricsRedisCache", _fake_redis_cache)

    provider._ensure_llm_cache_configured()
    provider._ensure_llm_cache_configured()
    provider._ensure_llm_cache_configured()

    assert len(calls) == 1
