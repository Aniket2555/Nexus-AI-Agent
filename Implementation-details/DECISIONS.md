# NEXUS — Decision Log

Single home for the questions that were previously repeated, unanswered, at the bottom of
three different planning documents. Record decisions here; do not re-open them in phase docs.

| # | Decision | Status | Date |
|---|---|---|---|
| D1 | Canonical planning document | **Settled** | 2026-08-07 |
| D2 | Agent runtime | **Settled** | 2026-08-07 |
| D3 | Web UI | **Settled** | 2026-08-07 |
| D4 | Build-vs-reuse posture | **Settled** | 2026-08-07 |
| D5 | Evaluation timing | **Settled** | 2026-08-07 |
| D6 | Checkpointer ownership | **Settled** | 2026-08-07 |
| D7 | Primary LLM provider | **Settled** | 2026-08-07 |
| D8 | Cloud provider | **Settled** | 2026-08-08 |
| D9 | LangSmith account | **Open** (low impact) | — |
| D10 | UI customisation depth | **Open** (needed by Phase 6) | — |
| D11 | Databricks / Spark integration | **Open** (needed by Phase 5) | — |
| D12 | Project name | **Open** (cosmetic) | — |

---

## Settled

### D1 — `Phases.md` is the only implementable plan
8 phases, 18–26 weeks (revised — see D4). Everything under `archive/` is history and carries a
SUPERSEDED banner. If a phase doc and an archived doc disagree, the phase doc wins; if two
phase docs disagree, that's a bug — fix it rather than picking one.

### D2 — LangGraph Platform is the agent runtime
All agent logic is `StateGraph` (state + nodes + edges), served by `langgraph dev` locally and
LangGraph Server in production. FastAPI is retained **only** for non-agent endpoints: document
upload, admin, evaluation, health. No custom agent loop, no custom orchestrator.

### D3 — Agent Chat UI is the web UI
`langchain-ai/agent-chat-ui` (Next.js), cloned and extended, rather than a frontend built from
scratch. Custom NEXUS panels are added as components on top of it. Depth of customisation is
still open — see D10.

### D4 — Reuse framework built-ins where the learning value is low
Adopted 2026-08-07. Replace hand-written code with library equivalents for the plumbing;
keep from-scratch implementations where writing them is the point of the project.

| Reuse (do not hand-write) | Build from scratch (learning value) |
|---|---|
| `init_chat_model()` over a custom provider factory | `context_engine/` — the whole assembly pipeline |
| `RecursiveCharacterTextSplitter.from_tiktoken_encoder()` | Loop *patterns*: reflection, self-consistency, debate |
| LangGraph node `RetryPolicy` over `loop_engine/retry.py` | Supervisor graph, planner, coordinator, reviewer |
| Checkpointer + `langgraph.store.BaseStore` over custom memory persistence | Evaluation harness and metrics |
| `set_llm_cache(RedisCache(...))` over a custom cache layer | Sandbox isolation policy |
| `langchain-mcp-adapters` consuming real MCP servers over 14 hand-written tool modules | Tools with no existing MCP server |

Consequence: the timeline shortens. See the revised table at the end of `Phases.md`.

### D5 — A thin evaluation harness lands at the end of Phase 2, not Phase 8
Phases 2–4 make claims ("reranking improves precision", "compression 2-4x without quality
loss", "query rewriting improves retrieval") that cannot be checked without one. Phase 2.7
adds a 20–50 question golden set plus faithfulness / context-precision scoring. The full eval
suite, benchmarks and dashboards stay in Phase 8.

### D6 — The graph does **not** own a checkpointer
Compile with `builder.compile()` and no checkpointer argument. LangGraph Platform injects its
own persistence and **ignores** a custom one — it emits a warning saying so
([langgraph#6559](https://github.com/langchain-ai/langgraph/issues/6559)). `langgraph dev`
persists to a local store, not to your Postgres
([langgraph#5790](https://github.com/langchain-ai/langgraph/issues/5790)).

`PostgresSaver` is correct **only** if a graph is embedded directly in a Python process
outside the Platform. NEXUS does not do that, so it appears nowhere in the build.

### D7 — Primary LLM provider: Groq (chat) + local HuggingFace (embeddings)
Chosen so the whole dev stack runs on free tiers with a single API key. `init_chat_model()`
(D4) means switching provider later is a config edit, not a code edit — so this costs nothing
to revisit if Groq's free-tier rate limits become a problem, or once a paid budget exists.

- **Chat:** Groq, via `CHAT_MODEL=groq:llama-3.3-70b-versatile`. OpenAI-compatible, generous
  free-tier rate limits, fast inference. Requires `GROQ_API_KEY`.
- **Embeddings:** Groq has **no embeddings endpoint or models** — confirmed directly against
  `console.groq.com/docs/models` and `console.groq.com/docs/api-reference` (an initial web
  search claiming otherwise was wrong and was rejected). Filled with a **local**
  `sentence-transformers` model via `langchain-huggingface`
  (`sentence-transformers/all-MiniLM-L6-v2`, 384-dim) instead of a second hosted provider:
  it runs on CPU, needs no API key, makes no network call per embedding, and is free with no
  rate limit. The tradeoff is a one-time ~90 MB model download and slightly lower retrieval
  quality than a large hosted embedding model — acceptable for Phase 1–2; §2.7's eval harness
  is what will tell us if it's worth upgrading to `BAAI/bge-small-en-v1.5` (also 384-dim, drop-in)
  or a hosted model later.
- `openai_api_key` / `anthropic_api_key` stay in `Settings` as optional fields — `init_chat_model`
  remains provider-agnostic (D4), so nothing stops pointing `CHAT_MODEL` at OpenAI or Anthropic
  later by setting the key and changing one env var.

### D8 — Cloud provider: Azure, with self-hosted Qdrant/Elasticsearch (not Azure AI Search)
Azure, as every archived plan already assumed and Phase 8's own component table already names
(AKS, ACR, Azure OpenAI, Azure Cache for Redis, Azure Database for PostgreSQL, Key Vault) —
nothing surfaced through Phases 1–7 that argued for a different cloud, so there was no reason
to re-open it rather than confirm it.

**Azure AI Search is *not* adopted as a Qdrant/Elasticsearch replacement**, despite Phase 8's
table listing it as an option. Retrieval (Phase 2) is built directly against Qdrant's and
Elasticsearch's own client APIs and query shapes (`query_points`, RRF fusion, `dstdomain`-style
payload filters) — swapping the retrieval backend at Phase 8 would mean rewriting `rag/retrieval/`
against a third API, for a managed-service convenience that isn't a stated requirement anywhere
in `Phases.md`. AKS runs Qdrant and Elasticsearch as stateful workloads instead, same containers
as `docker-compose.yml`, just orchestrated. Revisit only if self-hosting these under load
becomes an actual operational problem — not preemptively.

Terraform, Helm, and CI/CD (§8.2–§8.4) are written for real and validated with the tooling that
doesn't require an Azure subscription (`terraform validate`, `helm lint`/`template`, YAML
parsing) — this project has neither Azure credentials nor a live AKS cluster available, so
`terraform apply` / `helm install` against real Azure resources is not exercised. Documented
per-file in Phase 8's verification table, same honesty standard as every other phase's
infra-it-couldn't-reach (Phase 5's missing Node.js, Phase 6's unbuilt frontend, Phase 6's
un-exercised LangSmith Studio).

---

## Open — needed before the phase in brackets

### D9 — LangSmith account *(low impact)*
Largely moot. `langgraph dev` runs as a local in-memory server **without** a LangSmith key,
and `LANGSMITH_TRACING=false` keeps all data local. A key is needed only for the hosted Studio
UI and for cloud-stored traces ([Studio docs](https://docs.langchain.com/oss/python/langgraph/studio)).

> **Recommendation:** take the free tier. Studio's state inspection and time-travel debugging
> are the single biggest productivity win in Phases 3–6, and D5's eval harness targets
> LangSmith datasets. If tracing must stay on-prem, self-host Langfuse instead and drop the
> Studio-based debugging steps from the phase docs.

### D10 — UI customisation depth *(needed by Phase 6)*
Determines whether Phase 6 extends Agent Chat UI's existing components or rebuilds the surface
with `assistant-ui`. Currently the plan lists 8 custom components without saying which.

### D11 — Databricks / Spark integration *(needed by Phase 5)*
Whether Spark / Delta Lake / Databricks appear as MCP tools in the Phase 5 ecosystem or are
out of scope. Under D4 the question becomes narrower: is there an MCP server to consume, or
does this require hand-written tools?

### D12 — Project name *(cosmetic)*
"NEXUS" is a working name and is hardcoded in prompts, the Qdrant collection name
(`nexus_documents`), the Postgres database and role, and container names. Cheap to change
before Phase 1, progressively less so after.
