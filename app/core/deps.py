"""Dependency wiring.

Each factory builds a service once (lru_cache) and reuses it, so expensive clients
(LLM/embedding models, Qdrant, Redis) are created a single time. FastAPI routes receive
these through `Depends(...)`, which also lets tests swap in fakes via dependency_overrides.
"""

from functools import lru_cache

from app.core.config import get_settings
from app.services.booking import BookingService
from app.services.chat import ChatService
from app.services.ingestion import IngestionService
from app.services.llm import LLMClient
from app.services.memory import ChatMemory
from app.services.rag import RagService
from app.services.vector_store import VectorStore


@lru_cache
def get_llm() -> LLMClient:
    """Shared LLM client: local embeddings plus chat completions."""
    return LLMClient(get_settings())


@lru_cache
def get_vector_store() -> VectorStore:
    """Shared Qdrant client wrapper."""
    return VectorStore(get_settings())


@lru_cache
def get_ingestion_service() -> IngestionService:
    """Service used by the document upload endpoint."""
    return IngestionService(get_llm(), get_vector_store())


@lru_cache
def get_chat_service() -> ChatService:
    """Service used by the chat endpoint: memory + RAG + booking combined."""
    settings = get_settings()
    rag = RagService(get_llm(), get_vector_store(), settings)
    return ChatService(ChatMemory(settings), rag, BookingService(get_llm()))