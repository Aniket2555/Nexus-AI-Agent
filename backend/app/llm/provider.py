import logging
from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain_core.caches import RETURN_VAL_TYPE
from langchain_core.globals import set_llm_cache
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_redis import RedisCache

from backend.app.config import get_settings
from backend.app.observability.metrics import (
    llm_cache_hits_total,
    llm_cache_misses_total,
    push_metrics,
)

logger = logging.getLogger(__name__)

_cache_configured = False


class MetricsRedisCache(RedisCache):
    """`RedisCache` plus the hit/miss counter it doesn't expose on its own (§8.6
    gap: "Cache Hit Ratio ... Not exposed").

    Only wraps `lookup()` — a cache *read* is the only place "hit" vs. "miss" is
    a meaningful distinction; `update()` (a cache write, always unconditional)
    isn't. Pushed the same way `supervisor.py`'s `finalize_node` pushes agent
    metrics: this cache is configured for (and only ever exercised from) the
    `langgraph dev` process (see `_ensure_llm_cache_configured`'s docstring), so
    there's no FastAPI `/metrics` route for these counters to sit behind.

    Raises exactly what the base class would on a lookup failure — wrapping
    adds a counter increment and a best-effort push, not a new success/failure
    path. `push_metrics()` itself never raises (Phase 8.6's documented
    contract), so an unreachable Pushgateway degrades to "no metric recorded,"
    never "cache lookup failed" — the same degrade-gracefully posture as
    `_ensure_llm_cache_configured` already applies to the cache being
    unreachable in the first place.
    """

    def lookup(self, prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None:
        result = super().lookup(prompt, llm_string)
        if result is not None:
            llm_cache_hits_total.inc()
        else:
            llm_cache_misses_total.inc()
        push_metrics("nexus_llm_cache")
        return result


def _ensure_llm_cache_configured() -> None:
    """Best-effort Redis LLM response cache (§4.1), configured once per process.

    `set_llm_cache()` is process-global — `rag_chat.py` runs under `langgraph dev`'s
    process, not the FastAPI app's, so this has to be wired in here (the one
    chokepoint every chat model call already passes through) rather than in
    `api/main.py`'s lifespan, which never runs in the `langgraph dev` process at all.

    `RedisCache(...)` connects *eagerly* — verified directly: pointing it at an
    unreachable host raises `redis.exceptions.ConnectionError` from `__init__`,
    not from the first cache lookup. The same eager-construction class of bug as
    Phase 1's `OpenAIEmbeddings` and Phase 3's `ChatGroq`. Since a down cache
    should degrade to "no caching," not "every chat call now fails," this is
    caught and logged rather than propagated.
    """
    global _cache_configured
    if _cache_configured:
        return
    settings = get_settings()
    try:
        cache = MetricsRedisCache(
            redis_url=settings.redis_url, ttl=settings.redis_cache_ttl_seconds
        )
        set_llm_cache(cache)
    except Exception:
        logger.warning(
            "Redis LLM cache unavailable at %s; continuing uncached.", settings.redis_url
        )
    _cache_configured = True


@lru_cache(maxsize=8)
def get_chat_model(model: str | None = None, temperature: float = 0.1) -> BaseChatModel:
    """Provider-agnostic chat model, cached per (model, temperature).

    `init_chat_model` resolves the provider from the "provider:model" prefix and picks
    up the matching key from the environment, which replaces a hand-rolled if/elif
    provider factory entirely (DECISIONS.md D4).

    streaming=True matters: LangGraph's `messages` stream mode — what Agent Chat UI
    consumes — emits tokens from the model's streaming callbacks. Without it `ainvoke`
    issues a single non-streaming request and the UI renders one block at the end.
    """
    _ensure_llm_cache_configured()
    settings = get_settings()
    return init_chat_model(
        model or settings.chat_model, temperature=temperature, streaming=True
    )
