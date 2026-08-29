# NEXUS frontend

A fork of [`langchain-ai/agent-chat-ui`](https://github.com/langchain-ai/agent-chat-ui),
extended with NEXUS-specific panels (document upload, RAG source viewer, agent
status, memory inspector, supervisor view). See `Implementation-details/DECISIONS.md`
(D3, D10) for why this is a fork rather than a from-scratch build.

## Local setup

```bash
npm install
npm run dev
```

The app expects two backend services running locally (see the repo root
`README.md`):

- `langgraph dev` on `http://localhost:2024` — the `rag_chat` and
  `nexus_supervisor` graphs, streamed via `@langchain/langgraph-sdk`.
- `uvicorn backend.app.api.main:app --port 8000` — the non-agent REST API
  (document upload, workflows).

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | `http://localhost:2024` | LangGraph server the chat stream connects to. |
| `NEXT_PUBLIC_ASSISTANT_ID` | `rag_chat` | Graph to invoke — `rag_chat` for single-turn RAG, `nexus_supervisor` for multi-agent orchestration. |
| `NEXT_PUBLIC_NEXUS_API_URL` | `http://localhost:8000` | FastAPI base URL used by the Document Uploader panel. |
| `NEXT_PUBLIC_NEXUS_API_KEY` | unset | Sent as `X-API-Key` to the REST API. Not required locally — `require_api_key` bypasses auth when `DEBUG=true` and no `NEXUS_API_KEYS` are configured. |
| `NEXT_PUBLIC_NEXUS_TENANT_ID` | `default` | Forwarded as `config.configurable.tenant_id` on every chat run, so retrieval sees documents uploaded under the same tenant. Must match what `NEXT_PUBLIC_NEXUS_API_KEY` resolves to server-side (`NEXUS_API_KEYS` in the backend's `.env`). |
| `NEXT_PUBLIC_NEXUS_USER_ID` | `default` | Forwarded as `config.configurable.user_id` on every chat run (memory/workflow attribution). |

The deployment URL, assistant ID, and LangSmith API key (for a hosted
deployment) can also be set at runtime through the in-app form; those are
stored in `localStorage`, not this `.env`.

## NEXUS-specific additions

Everything under `src/components/nexus/` and `src/components/generative-ui/`
is NEXUS-specific; everything else is the vendored upstream `agent-chat-ui`
base, kept as close to it as practical so upstream fixes stay easy to pull in.

- `NexusSidebar.tsx` — the single entry point wiring the five panels below
  into a slide-over next to `Thread`, without touching `Thread`'s own layout.
- `AgentStatusPanel.tsx` — live iteration/quality-score readout, shared by
  both graphs since both reuse `should_retry()`.
- `RAGSourceViewer.tsx` — `rag_chat`'s `citations`.
- `MemoryInspector.tsx` — `rag_chat`'s `recalled_memory` (long-term facts +
  knowledge graph).
- `SupervisorPanel.tsx` — `nexus_supervisor`'s `sub_tasks` / `agent_results`
  DAG, expandable per task into `generative-ui/`'s renderers.
- `DocumentUploader.tsx` — talks to the FastAPI `/api/v1/documents` endpoints
  directly, not the graph stream.
- `generative-ui/{CodeBlock,DataChart,SearchResults}.tsx` — render a
  `SpecialistResult`'s `details` for the code/data/research specialists.
