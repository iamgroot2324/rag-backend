"""Application configuration.

All tunables (API keys, service URLs, model names, limits) live here so the rest of
the code never reads environment variables directly.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables / a local .env file."""

    # Read values from .env and ignore any unrelated variables in the environment.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Chat LLM: any OpenAI-compatible server (Ollama locally, or OpenAI itself) ---
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"  # Ollama ignores it, but the client requires a value
    llm_chat_model: str = "llama3.2:3b"

    # --- Local embeddings ---
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384  # all-MiniLM-L6-v2 outputs 384 dimensions

    # --- Qdrant (vector database) ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "documents"

    # --- Redis (chat memory) ---
    redis_url: str = "redis://localhost:6379/0"
    chat_history_limit: int = 20  # max messages kept per session
    chat_ttl_seconds: int = 60 * 60 * 24  # sessions expire after 24h of inactivity

    # --- SQL database (document metadata + bookings) ---
    database_url: str = "sqlite:///./app.db"

    # --- RAG / upload limits ---
    retrieval_top_k: int = 4  # number of chunks retrieved per question
    max_upload_bytes: int = 10 * 1024 * 1024  # 10 MB


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (built once per process)."""
    return Settings()