# NEXUS

An enterprise agentic AI platform built on [LangGraph](https://langchain-ai.github.io/langgraph/),
with a hybrid dense/sparse RAG pipeline, a multi-agent supervisor, long-term
memory backed by a knowledge graph, an MCP tool ecosystem, sandboxed code
execution, and full observability.

The frontend is a fork of
[`langchain-ai/agent-chat-ui`](https://github.com/langchain-ai/agent-chat-ui)
extended with NEXUS-specific panels (document upload, RAG source viewer, agent
status, memory inspector).

See `Implementation-details/Phases.md` for the full phase-by-phase build
history and `Implementation-details/DECISIONS.md` for the architectural
decisions and why each one was made.

## Architecture at a glance

- **`rag_chat` graph** — single-turn RAG: rewrite → retrieve (hybrid
  dense+sparse, RRF fusion) → memory recall → rerank (cross-encoder) →
  compress → assemble → generate → reflect → finalize → remember.
- **`nexus_supervisor` graph** — multi-agent: plan → dispatch specialists
  (research/code/data/browser/vision) as a dependency DAG → coordinate →
  review → reflect → escalate-to-human-on-repeated-failure → finalize.
- **Memory** — long-term memory in Qdrant (via a custom `BaseStore`
  implementation), a Neo4j knowledge graph, and a Redis LLM response cache.
- **MCP** — real MCP servers (filesystem, playwright, github) plus
  hand-written tools (SQL, vision, image generation, sandboxed Python/shell).
- **Sandbox** — Docker-isolated code execution with CPU/memory/disk/pid
  limits and an egress-allowlisted network via a squid proxy.
- **Observability** — Prometheus + Pushgateway + Grafana + Alertmanager.

## Local setup

```bash
./scripts/setup.sh          # brings up infra containers, installs the Python env
cp .env.example .env        # then fill in GROQ_API_KEY at minimum
python -m scripts.seed_data # loads a small sample corpus so there's something to query
```

Then, in separate terminals:

```bash
langgraph dev                                                # the agent graphs
uvicorn backend.app.api.main:app --reload --port 8000        # the REST API (uploads, workflows)
cd frontend && npm install && npm run dev                    # the chat UI
```

## Tests and evaluation

```bash
pytest                                              # unit + integration tests
python -m backend.app.evaluation.run_eval           # retrieval quality against the golden set
python -m backend.app.evaluation.benchmarks         # retrieval + agent benchmark, combined report
```

## Reindexing / reseeding

```bash
python -m scripts.reindex --from-registry --tenant-id acme --version v2
python -m scripts.seed_data
```
