from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Embedding model -> vector dimensionality. The Qdrant collection is sized from this
# rather than a hardcoded constant: switching models against a collection sized for a
# different width fails at upsert time with a confusing error.
EMBEDDING_DIMENSIONS: dict[str, int] = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "BAAI/bge-small-en-v1.5": 384,
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
    "text-embedding-ada-002": 1536,
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- LLM -------------------------------------------------------------------
    # Provider-qualified, as consumed by init_chat_model(): "groq:openai/gpt-oss-120b"
    # (DECISIONS.md D7 — the original "groq:llama-3.3-70b-versatile" pick was removed
    # from Groq's catalog after this project was built; Groq's lineup rotates, so
    # check console.groq.com/docs/models before assuming this string still resolves),
    # "openai:gpt-4o", "anthropic:claude-sonnet-4-5". Changing provider is a config
    # edit, not a code edit.
    chat_model: str = "groq:openai/gpt-oss-120b"
    groq_api_key: str = ""
    # HuggingFace repo id, loaded locally by langchain-huggingface — not a hosted API,
    # so there is no matching *_api_key field for it.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Optional. Unused by the Phase 1 default but kept so init_chat_model() can point
    # at OpenAI/Anthropic later without a code change (DECISIONS.md D4, D7).
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # --- Data stores -----------------------------------------------------------
    qdrant_url: str = "http://localhost:6333"
    # Not read by Phase 1 code (D6: the graph owns no checkpointer). Present because
    # `langgraph up` needs it, and the document registry moves here in Phase 2.
    postgres_url: str = "postgresql://nexus:nexus@localhost:5432/nexus"
    redis_url: str = "redis://localhost:6379"  # Phase 4
    elasticsearch_url: str = "http://localhost:9200"
    neo4j_url: str = "bolt://localhost:7687"  # Phase 4
    # Matches infra/docker/docker-compose.yml's NEO4J_AUTH defaults.
    neo4j_user: str = "neo4j"
    neo4j_password: str = "nexus_dev"

    # --- Retrieval -------------------------------------------------------------
    # Versioned collection/index names, kept in lockstep. §2.7 notes that a *chunking
    # strategy* change forces a _v2 bump + scripts/reindex.py; unlike the original
    # OpenAI-based plan, Phase 1 already committed to the final embedding model
    # (DECISIONS.md D7), so nothing here forces a bump purely from Phase 2 landing.
    qdrant_collection: str = "nexus_documents_v1"
    elasticsearch_index: str = "nexus_documents_v1"
    # Candidates fetched from each of dense/sparse before RRF fusion (§2.1).
    retrieval_top_k: int = 10
    # Kept after cross-encoder reranking (§2.2) — this is what generate_node sees.
    rerank_top_k: int = 5
    # Lowered from the original 0.3 (2026-08-27) after live testing found 0.3 too
    # strict even for the Phase 2.7 golden set: rerunning that eval at 0.1 improved
    # both metrics (precision 0.95->0.975, recall 0.95->1.0) rather than hurting
    # them, and it's what let a real "list the content in course 1 section" query
    # survive reranking against a real uploaded PDF (BGE cross-encoders score
    # meta/structural questions like this well below a QA-style question against
    # the same passage — 0.03-0.11 here vs. 0.96 for the "is this passage relevant"
    # case reranker.py's own docstring cites). Re-validate against the golden set
    # (`python -m backend.app.evaluation.run_eval`) before changing this again.
    rerank_score_threshold: float = 0.1
    # Cosine similarity above which two chunks are considered near-duplicates (§2.3).
    dedup_similarity_threshold: float = 0.95
    # Token budget the compressor (§2.3) targets for the assembled context block.
    compression_token_budget: int = 1200

    # --- Context & loop engineering (Phase 3) -----------------------------------
    # Matches groq:llama-3.3-70b-versatile's context window. Recheck this when D7's
    # chat model changes — TokenBudget has no way to discover it automatically.
    context_window_tokens: int = 128_000
    # Messages kept verbatim before the rolling-window summarizer kicks in.
    summarizer_keep_last_n: int = 6
    summarizer_trigger_message_count: int = 12
    # Reflection loop (§3.2): retry threshold and hard iteration cap.
    reflection_quality_threshold: float = 0.8
    reflection_max_iterations: int = 3
    # Illustrative Groq per-token pricing for the reflection loop's cost ceiling
    # (§3.2 "Budget Tracking"). Update against console.groq.com/docs/pricing for the
    # configured CHAT_MODEL — this is not fetched automatically.
    cost_per_1k_input_tokens_usd: float = 0.00059
    cost_per_1k_output_tokens_usd: float = 0.00079
    max_reflection_cost_usd: float = 0.05
    # Self-consistency (§3.2): number of parallel samples for the voting pattern.
    self_consistency_samples: int = 3
    # Query rewriting (§3.1): total queries searched with per turn, original + rewrites.
    multi_query_variants: int = 3

    # --- Memory & knowledge (Phase 4) -------------------------------------------
    redis_cache_ttl_seconds: int = 1800
    qdrant_memory_collection: str = "nexus_memory_v1"
    # Namespace tuples deeper than this raise at write time — bounds the number of
    # indexed ns_N payload fields longterm.py's Qdrant-backed BaseStore needs to
    # support prefix search (see §4.1's note on why Qdrant has no native prefix op).
    memory_max_namespace_depth: int = 6
    # Neo4j entity/relationship extraction (§4.2): node/edge types the extraction
    # prompt is constrained to, matching the plan's table exactly rather than
    # letting the LLM invent arbitrary labels a Cypher query pattern won't expect.
    kg_node_types: tuple[str, ...] = ("Person", "Project", "Concept", "Tool", "Document", "Code")
    kg_edge_types: tuple[str, ...] = (
        "RELATED_TO", "DEPENDS_ON", "CREATED_BY", "MENTIONS", "USES",
    )
    graph_rag_max_hops: int = 2
    graph_rag_top_k: int = 5

    # --- MCP tool ecosystem (Phase 5) -------------------------------------------
    # Token-bucket rate limiting (§5.1), per (agent, tool) pair.
    tool_rate_limit_capacity: int = 10
    tool_rate_limit_refill_per_second: float = 1.0
    tool_call_timeout_seconds: float = 30.0
    # SQL tool (§5.2): a hard ceiling independent of whatever LIMIT (if any) the
    # generated query specifies — belt and suspenders against a runaway result set.
    sql_tool_max_rows: int = 200
    # Egress allowlist (§5.5) for the browser/fetch tools. Empty by default so a
    # fresh deploy fails closed (denies everything) rather than failing open.
    egress_allowed_domains: tuple[str, ...] = ()
    # vision.py — verified against console.groq.com/docs/vision (2026-08-07): the
    # only vision-capable model Groq currently lists. Re-check that page if this
    # ever 404s — Groq's vision lineup has changed before and will again.
    vision_model: str = "groq:qwen/qwen3.6-27b"
    # image_gen.py has no free-tier provider (Groq: chat/vision/audio only, no
    # image generation endpoint — DECISIONS.md D7). Set both to enable it via
    # OpenAI's Images API; left blank by default, same as openai_api_key elsewhere.
    image_gen_model: str = "gpt-image-1"

    # --- Sandboxed execution (Phase 7) ------------------------------------------
    # Images built from infra/docker/sandbox/Dockerfile.*. Tagged locally rather
    # than pulled from a registry — §7.1 has no distribution story yet, only
    # local build (see sandbox/manager.py's build-on-first-use).
    sandbox_image_python: str = "nexus-sandbox-python:latest"
    sandbox_image_node: str = "nexus-sandbox-node:latest"
    sandbox_image_shell: str = "nexus-sandbox-shell:latest"
    sandbox_image_proxy: str = "nexus-sandbox-proxy:latest"
    sandbox_cpu_cores: float = 2.0
    sandbox_memory_mb: int = 512
    sandbox_tmp_mb: int = 1024
    # Fork-bomb guard beyond the literal §7.1 table (CPU/memory/disk/timeout) — a
    # runaway process spawning children is cheap to cap and costly not to.
    sandbox_pids_limit: int = 64
    sandbox_default_timeout_seconds: float = 60.0
    sandbox_max_output_bytes: int = 65_536
    sandbox_egress_network: str = "nexus-sandbox-egress"
    sandbox_proxy_host: str = "nexus-sandbox-proxy"
    sandbox_proxy_port: int = 3128

    # --- Observability (Phase 8) -------------------------------------------------
    pushgateway_url: str = "http://localhost:9091"

    # --- Non-agent API ---------------------------------------------------------
    # Explicit origins. "*" with allow_credentials=True is rejected by browsers.
    cors_origins: list[str] = ["http://localhost:3000"]
    max_upload_bytes: int = 25 * 1024 * 1024
    debug: bool = True

    # --- Auth (Phase 8) ---------------------------------------------------------
    # Simple API-key header (X-API-Key), not JWT/OAuth. JSON object mapping each key
    # to the tenant/user it resolves to — see backend/app/api/middleware/auth.py.
    # Empty (the default) plus DEBUG=true falls back to an unauthenticated "default"
    # tenant for local dev; set this to turn on real enforcement.
    api_keys: dict[str, dict[str, str]] = Field(default_factory=dict, validation_alias="NEXUS_API_KEYS")

    @property
    def embedding_dimensions(self) -> int:
        try:
            return EMBEDDING_DIMENSIONS[self.embedding_model]
        except KeyError:
            raise ValueError(
                f"Unknown embedding model {self.embedding_model!r}. Add it to "
                f"EMBEDDING_DIMENSIONS with its vector width."
            ) from None


@lru_cache
def get_settings() -> Settings:
    return Settings()
