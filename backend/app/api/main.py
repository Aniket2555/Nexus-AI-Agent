from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from backend.app.api.middleware.rate_limit import enforce_api_rate_limit
from backend.app.api.v1.documents import router as documents_router
from backend.app.api.v1.workflows import router as workflows_router
from backend.app.config import get_settings
from backend.app.memory.manager import get_memory_manager
from backend.app.models.database import ensure_schema
from backend.app.rag.retrieval.dense import get_retriever
from backend.app.rag.retrieval.sparse import get_sparse_retriever

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await get_retriever().ensure_collection()
    await get_sparse_retriever().ensure_index()
    await get_memory_manager().store.ensure_collection()
    await ensure_schema()

    yield

    await get_retriever().close()
    await get_sparse_retriever().close()
    await get_memory_manager().store.close()


app = FastAPI(title="NEXUS API", version="0.1.0", debug=settings.debug, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(
    documents_router, prefix="/api/v1", dependencies=[Depends(enforce_api_rate_limit)]
)

app.include_router(
    workflows_router, prefix="/api/v1", dependencies=[Depends(enforce_api_rate_limit)]
)

Instrumentator().instrument(app).expose(app)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy", "service": "nexus-api", "version": "0.1.0"}
