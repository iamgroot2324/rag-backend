"""Request/response schemas for the document ingestion API."""

from enum import Enum

from pydantic import BaseModel


class ChunkStrategy(str, Enum):
    """The two selectable chunking strategies."""

    FIXED = "fixed"  # fixed-size character windows with overlap
    SENTENCE = "sentence"  # sentence-aware packing


class IngestionResponse(BaseModel):
    """Returned after a document has been chunked, embedded and stored."""

    document_id: str
    filename: str
    chunking_strategy: ChunkStrategy
    chunk_count: int
    char_count: int
