from backend.app.loop_engine.budget import estimate_cost_usd
from backend.app.observability import metrics


def test_estimate_cost_usd_records_llm_cost_and_token_metrics(monkeypatch):
    monkeypatch.setattr(metrics.get_settings(), "cost_per_1k_input_tokens_usd", 0.001)
    monkeypatch.setattr(metrics.get_settings(), "cost_per_1k_output_tokens_usd", 0.002)

    cost_before = metrics.llm_cost_usd_total._value.get()
    input_before = metrics.llm_tokens_total.labels(direction="input")._value.get()
    output_before = metrics.llm_tokens_total.labels(direction="output")._value.get()

    cost = estimate_cost_usd(1000, 500)

    assert metrics.llm_cost_usd_total._value.get() == cost_before + cost
    assert metrics.llm_tokens_total.labels(direction="input")._value.get() == input_before + 1000
    assert metrics.llm_tokens_total.labels(direction="output")._value.get() == output_before + 500


def test_push_metrics_is_best_effort_and_never_raises(monkeypatch):
    monkeypatch.setattr(
        metrics,
        "get_settings",
        lambda: type("S", (), {"pushgateway_url": "http://localhost:1"})(),
    )

    metrics.push_metrics("test_job")


def test_cache_hit_and_miss_counters_increment_independently():
    hits_before = metrics.llm_cache_hits_total._value.get()
    misses_before = metrics.llm_cache_misses_total._value.get()

    metrics.llm_cache_hits_total.inc()
    metrics.llm_cache_misses_total.inc()
    metrics.llm_cache_misses_total.inc()

    assert metrics.llm_cache_hits_total._value.get() == hits_before + 1
    assert metrics.llm_cache_misses_total._value.get() == misses_before + 2


def test_metrics_redis_cache_lookup_records_hit_and_miss(monkeypatch):
    """llm/provider.py's MetricsRedisCache wraps RedisCache.lookup() to add the
    hit/miss counter langchain_redis doesn't expose — verified here without a
    live Redis by stubbing the base class's lookup() and the Pushgateway call,
    the same "mock the boundary, not the logic" approach push_metrics' own
    tests above use.
    """
    from backend.app.llm import provider

    monkeypatch.setattr(provider, "push_metrics", lambda job: None)

    cache = provider.MetricsRedisCache.__new__(provider.MetricsRedisCache)

    hits_before = metrics.llm_cache_hits_total._value.get()
    misses_before = metrics.llm_cache_misses_total._value.get()

    monkeypatch.setattr(provider.RedisCache, "lookup", lambda self, prompt, model: None)
    assert cache.lookup("prompt", "model") is None
    assert metrics.llm_cache_misses_total._value.get() == misses_before + 1
    assert metrics.llm_cache_hits_total._value.get() == hits_before

    monkeypatch.setattr(provider.RedisCache, "lookup", lambda self, prompt, model: ["cached"])
    assert cache.lookup("prompt", "model") == ["cached"]
    assert metrics.llm_cache_hits_total._value.get() == hits_before + 1


def test_rag_retrieval_latency_histogram_observes_durations():
    count_before = metrics.rag_retrieval_latency_seconds._sum.get()

    metrics.rag_retrieval_latency_seconds.observe(0.25)

    assert metrics.rag_retrieval_latency_seconds._sum.get() == count_before + 0.25


def test_rate_limit_tokens_remaining_gauge_is_set_per_label():
    metrics.rate_limit_tokens_remaining.labels(agent_name="research", tool_name="search").set(2.5)

    assert (
        metrics.rate_limit_tokens_remaining.labels(agent_name="research", tool_name="search")
        ._value.get()
        == 2.5
    )


async def test_redis_rate_limiter_allow_sets_gauge_and_pushes(monkeypatch):
    """RedisRateLimiter.allow() (policy.py) parses its rate_key back into
    (agent_name, tool_name) labels and pushes the gauge — verified here with a
    fake Redis client (async eval() returning the [allowed, tokens] shape the
    Lua script now returns) rather than a live Redis, mirroring
    test_policy.py's existing fakes for this module.

    Patches `apush_metrics`, not `push_metrics`: `allow()` is `async def` and
    genuinely runs on the event loop (unlike a plain sync LangGraph node,
    which the platform runtime offloads to a worker thread on its own), so a
    synchronous `push_to_gateway()` call here blocks the loop directly.
    Verified live against a real `langgraph dev` server, which raises
    `BlockingError` and kills the run outright when this is called
    unawaited — not a hypothetical concern.
    """
    from backend.app.mcp import policy

    pushed_jobs = []

    async def fake_apush_metrics(job):
        pushed_jobs.append(job)

    monkeypatch.setattr(policy, "apush_metrics", fake_apush_metrics)

    class _FakeRedis:
        async def eval(self, *args, **kwargs):
            return [1, "2.5"]

    limiter = policy.RedisRateLimiter(_FakeRedis(), capacity=5, refill_per_second=1.0)

    allowed = await limiter.allow("tool_rate:research:search")

    assert allowed is True
    assert pushed_jobs == ["nexus_rate_limiter"]
    assert (
        metrics.rate_limit_tokens_remaining.labels(agent_name="research", tool_name="search")
        ._value.get()
        == 2.5
    )


def test_parse_rate_key_falls_back_for_non_tool_rate_keys():
    from backend.app.mcp.policy import _parse_rate_key

    assert _parse_rate_key("tool_rate:research:search") == ("research", "search")
    assert _parse_rate_key("some_other_key") == ("some_other_key", "unknown")
