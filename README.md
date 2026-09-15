# NEXUS

> An enterprise-grade agentic AI platform — RAG, multi-agent orchestration, long-term memory, sandboxed code execution, and full observability, all in one stack.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Platform-blueviolet)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Next.js 15](https://img.shields.io/badge/Next.js-15-black?logo=next.js)](https://nextjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What is NEXUS?

NEXUS is a full-stack AI agent platform built on [LangGraph](https://langchain-ai.github.io/langgraph/). It combines:

- **Hybrid RAG** — dense + sparse retrieval with Reciprocal Rank Fusion (RRF), cross-encoder reranking, and context compression
- **Multi-agent supervisor** — a planner/dispatcher graph that fans out to specialist agents (research, code, data analysis, browser automation, vision)
- **Long-term memory** — Qdrant vector store (`BaseStore`) + Neo4j knowledge graph, with Redis for LLM response caching
- **MCP tool ecosystem** — real MCP servers (filesystem, Playwright, GitHub) plus hand-written tools (SQL, vision, image generation, sandboxed Python/shell)
- **Docker-isolated sandbox** — CPU / memory / disk / PID limits + egress-allowlisted network via Squid proxy
- **Full observability** — Prometheus + Pushgateway + Grafana + Alertmanager

The frontend is a fork of [`langchain-ai/agent-chat-ui`](https://github.com/langchain-ai/agent-chat-ui) (Next.js 15 / React 19), extended with NEXUS-specific panels.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent runtime | LangGraph Platform (`langgraph dev` / LangGraph Server) |
| REST API | FastAPI + Uvicorn |
| Frontend | Next.js 15, React 19, TypeScript |
| Vector store | Qdrant 1.19 |
| Keyword search | Elasticsearch 8.15 |
| Relational DB | PostgreSQL 16 |
| Knowledge graph | Neo4j 5 |
| Cache | Redis 7 |
| LLM (chat) | Groq — `llama-3.3-70b-versatile` (configurable) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, no API key) |
| Observability | Prometheus · Grafana · Alertmanager |
| CI/CD | GitHub Actions |
| Cloud (prod) | Azure (AKS, ACR, Azure OpenAI-compatible endpoint) |

---

## Architecture Overview

### `rag_chat` graph *(single-turn RAG)*

```
query rewrite → hybrid retrieve (dense + sparse, RRF)
             → memory recall
             → cross-encoder rerank
             → context compression
             → assemble prompt
             → generate
             → reflect
             → finalize → remember
```

### `nexus_supervisor` graph *(multi-agent)*

```
plan → dispatch specialists as a dependency DAG
     (research / code / data / browser / vision)
     → coordinate → review → reflect
     → escalate-to-human on repeated failure
     → finalize
```

---

## Project Structure

```
nexus/
├── backend/
│   └── app/
│       ├── api/              # FastAPI routes (documents, workflows, health, metrics)
│       ├── graphs/           # LangGraph state graphs (rag_chat, supervisor)
│       ├── rag/              # Ingestion, processing, retrieval pipeline
│       ├── memory/           # Long-term memory manager, Qdrant store, Neo4j KG
│       ├── mcp/              # MCP client, policy, custom tools
│       ├── sandbox/          # Docker-isolated code execution
│       ├── context_engine/   # Prompt assembly pipeline
│       ├── loop_engine/      # Reflection, self-consistency, debate patterns
│       ├── security/         # Egress policy, content redaction, boundaries
│       ├── evaluation/       # Retrieval quality harness + benchmark suite
│       ├── observability/    # Prometheus metrics
│       └── prompts/          # Jinja2 agent prompt templates
├── frontend/                 # Next.js 15 chat UI (fork of agent-chat-ui)
│   └── src/components/nexus/ # NEXUS-specific panels (RAG sources, memory, supervisor…)
├── infra/                    # Docker Compose, Kubernetes / Helm, Terraform, Nginx
├── scripts/                  # setup.sh, seed_data, reindex
├── tests/                    # Unit + integration tests (pytest)
├── langgraph.json            # Graph registry for LangGraph Platform
└── pyproject.toml
```

---

## Prerequisites

- **Python 3.11+** and [`uv`](https://docs.astral.sh/uv/) (or `pip`)
- **Node.js 20+** and `npm`
- **Docker** + **Docker Compose** (for infra containers)
- A free **[Groq API key](https://console.groq.com/)** (minimum requirement)

---

## Quick Start

### 1. Clone and configure

```bash
git clone https://github.com/<your-org>/nexus.git
cd nexus
cp .env.example .env
# Open .env and set GROQ_API_KEY (required) and any other keys you want
```

### 2. Start infrastructure

```bash
./scripts/setup.sh        # brings up Qdrant, Postgres, Elasticsearch, Redis, Neo4j
```

### 3. Seed sample data

```bash
python -m scripts.seed_data   # loads a small sample corpus so there's something to query
```

### 4. Run the three services (separate terminals)

```bash
# Terminal 1 — LangGraph agent runtime
langgraph dev

# Terminal 2 — FastAPI REST server (uploads, workflows, health)
uvicorn backend.app.api.main:app --reload --port 8000

# Terminal 3 — Next.js chat UI
cd frontend && npm install && npm run dev
```

Open **http://localhost:3000** in your browser.

---

## Environment Variables

The full list is in [`.env.example`](.env.example). The only **required** key for a local run is:

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Chat LLM (Groq Llama 3.3 70B) |

Optional (unlock extra features):

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Use OpenAI models via `CHAT_MODEL=openai:gpt-4o` |
| `ANTHROPIC_API_KEY` | Use Claude via `CHAT_MODEL=anthropic:claude-3-5-sonnet` |
| `LANGSMITH_API_KEY` | LangSmith tracing + Studio UI |
| `GITHUB_TOKEN` | GitHub MCP server |
| `CHAT_MODEL` | Override the default model (e.g., `groq:llama-3.3-70b-versatile`) |

> **Switching LLM providers** is a single env-var change — `CHAT_MODEL=openai:gpt-4o` with the matching API key. No code changes needed.

---

## Testing & Evaluation

```bash
# Unit + integration tests
pytest

# Retrieval quality against the golden set
python -m backend.app.evaluation.run_eval

# Full retrieval + agent benchmark with combined report
python -m backend.app.evaluation.benchmarks
```

---

## Re-indexing / Re-seeding

```bash
python -m scripts.reindex --from-registry --tenant-id acme --version v2
python -m scripts.seed_data
```

---

## Frontend Panels

The NEXUS sidebar (`frontend/src/components/nexus/`) adds five panels on top of the base chat UI:

| Panel | Purpose |
|---|---|
| **Agent Status** | Live node-by-node execution trace |
| **RAG Source Viewer** | Retrieved chunks with relevance scores |
| **Memory Inspector** | Long-term memory entries and knowledge graph nodes |
| **Supervisor** | Multi-agent plan, dispatch status, sub-agent results |
| **Document Upload** | Ingest documents into the RAG pipeline |

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Agent runtime | LangGraph Platform | `StateGraph` + built-in persistence, checkpointing, retries |
| LLM provider | Groq (default) | Free tier, fast inference, OpenAI-compatible — swap via one env var |
| Embeddings | Local `all-MiniLM-L6-v2` | No API key, no rate limits, no network call per embedding |
| Retrieval | Qdrant + Elasticsearch + RRF | Dense + sparse fusion outperforms either alone |
| Checkpointer | Platform-injected | Compiling with a custom checkpointer conflicts with LangGraph Platform ([#6559](https://github.com/langchain-ai/langgraph/issues/6559)) |
| Cloud (prod) | Azure (AKS) | AKS + ACR + Azure OpenAI — self-hosted Qdrant/ES, not Azure AI Search |

---

## Contributing

1. Fork the repo and create a feature branch.
2. Run `pytest` and ensure all tests pass before opening a PR.
3. Follow the existing code style — `ruff` for linting, `mypy` for type checking.

```bash
ruff check .
mypy backend/
```

---

## License

[MIT](LICENSE)
