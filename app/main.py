"""FastAPI application entry point."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import chat, ingestion
from app.core.deps import get_vector_store
from app.db import models  # noqa: F401  (import registers the tables on Base.metadata)
from app.db.session import Base, engine


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Startup: create SQL tables and the Qdrant collection if they don't exist."""
    Base.metadata.create_all(engine)
    get_vector_store().ensure_collection()
    yield


app = FastAPI(title="Palm Mind RAG Backend", version="1.0.0", lifespan=lifespan)
app.include_router(ingestion.router)
app.include_router(chat.router)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}
