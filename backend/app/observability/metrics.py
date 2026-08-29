"""Custom application metrics (§8.6's "Agent Performance"/"RAG Quality"/"LLM
Usage" Grafana dashboards), pushed to a Prometheus Pushgateway rather than
scraped directly.

Why pushed, not scraped: `main.py`'s `GET /metrics`
(prometheus-fastapi-instrumentator) only sees traffic that goes through
FastAPI — the non-agent API (uploads, workflows, health). The graphs
themselves (supervisor.py, rag_chat.py) run inside `langgraph dev`'s own
process (DECISIONS.md D2), which has no HTTP route for Prometheus to scrape.
Pushing to a Pushgateway (`prometheus_client.push_to_gateway`, the library's
own documented pattern for exactly this "batch job, not a server" case) is
what makes agent/LLM metrics observable at all without adding an HTTP server
to a process that doesn't otherwise need one.
"""

import asyncio

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, push_to_gateway

from backend.app.config import get_settings

registry = CollectorRegistry()

agent_tasks_total = Counter(
    "nexus_agent_tasks_total",
    "Supervisor graph runs completed, by whether the reviewer passed them.",
    ["status"],
    registry=registry,
)

agent_loop_iterations = Histogram(
    "nexus_agent_loop_iterations",
    "Reflection-loop iterations per supervisor run before pass/escalate.",
    buckets=(1, 2, 3, 4, 5, 8, 13),
    registry=registry,
)

llm_cost_usd_total = Counter(
    "nexus_llm_cost_usd_total",
    "Estimated LLM cost in USD (loop_engine.budget.estimate_cost_usd).",
    registry=registry,
)

llm_tokens_total = Counter(
    "nexus_llm_tokens_total",
    "LLM tokens consumed, by direction.",
    ["direction"],
    registry=registry,
)

rag_context_precision = Histogram(
    "nexus_rag_context_precision",
    "Golden-set context precision (retrieval_metrics.context_precision), per eval run.",
    buckets=(0.0, 0.25, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0),
    registry=registry,
)

rag_context_recall = Histogram(
    "nexus_rag_context_recall",
    "Golden-set context recall (retrieval_metrics.context_recall), per eval run.",
    buckets=(0.0, 0.25, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0),
    registry=registry,
)

rag_faithfulness = Histogram(
    "nexus_rag_faithfulness",
    "LLM-judged faithfulness score (faithfulness.py), per scored question.",
    buckets=(0.0, 0.25, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0),
    registry=registry,
)

# --- Update 2026-08-27 (§8.6 gap closures) -----------------------------------

llm_cache_hits_total = Counter(
    "nexus_llm_cache_hits_total",
    "Redis LLM response cache hits (langchain_redis.RedisCache has no counter "
    "of its own — see llm/provider.py's MetricsRedisCache).",
    registry=registry,
)

llm_cache_misses_total = Counter(
    "nexus_llm_cache_misses_total",
    "Redis LLM response cache misses (langchain_redis.RedisCache has no counter "
    "of its own — see llm/provider.py's MetricsRedisCache).",
    registry=registry,
)

rag_retrieval_latency_seconds = Histogram(
    "nexus_rag_retrieval_latency_seconds",
    "Hybrid (dense+sparse) retrieval wall time in rag_chat.py's retrieve_node, "
    "across every rewrite-variant query issued for one turn.",
    buckets=(0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0),
    registry=registry,
)

rate_limit_tokens_remaining = Gauge(
    "nexus_rate_limit_tokens_remaining",
    "Token-bucket tokens remaining after the most recent RedisRateLimiter.allow() "
    "call, by the same (agent_name, tool_name) key policy.py's rate_key uses.",
    ["agent_name", "tool_name"],
    registry=registry,
)


def push_metrics(job: str) -> None:
    """Push everything recorded in `registry` since the last push.

    Best-effort: a Pushgateway that's down or unreachable must never fail the
    graph run or eval run it's instrumenting — observability is a side
    channel, not a dependency of the thing being observed.
    """
    settings = get_settings()
    try:
        push_to_gateway(settings.pushgateway_url, job=job, registry=registry)
    except Exception:
        pass


async def apush_metrics(job: str) -> None:
    """Async wrapper for callers that are themselves `async def` (rag_chat.py's
    retrieve_node, RedisRateLimiter.allow(), the evaluation harness) —
    `push_to_gateway` is a synchronous, blocking HTTP call, and calling it
    directly from a coroutine runs it on the event loop rather than a worker
    thread. Under `langgraph dev`'s instrumented runtime this trips the
    platform's blocking-call detector and raises `BlockingError`, killing the
    run outright (verified directly: reproduced against a real `langgraph dev`
    server, not just inferred). Sync callers (`finalize_node`, the LLM cache's
    `lookup()` override) don't need this — LangGraph/LangChain already run
    plain sync functions in a worker thread on their own.
    """
    await asyncio.to_thread(push_metrics, job)
