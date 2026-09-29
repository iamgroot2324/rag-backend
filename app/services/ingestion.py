"""Document ingestion pipeline: extract -> chunk -> embed -> store."""

from sqlalchemy.orm import Session

from app.db.models import Document
from app.schemas.ingestion import ChunkStrategy, IngestionResponse
from app.services.chunking import chunk_text
from app.services.extraction import extract_text
from app.services.llm import LLMClient
from app.services.vector_store import VectorStore


class EmptyDocumentError(ValueError):
    """Raised when a file contains no extractable text (e.g. a scanned PDF)."""


class IngestionService:
    """Turns an uploaded file into searchable vectors plus a metadata record."""

    def __init__(self, llm: LLMClient, store: VectorStore) -> None:
        self._llm = llm
        self._store = store

    def ingest(
        self,
        db: Session,
        filename: str,
        content_type: str,
        data: bytes,
        strategy: ChunkStrategy,
        chunk_size: int,
        overlap: int,
    ) -> IngestionResponse:
        """Run the full ingestion pipeline for one file."""
        text = extract_text(filename, data)
        chunks = chunk_text(text, strategy, chunk_size, overlap)
        if not chunks:
            raise EmptyDocumentError("No extractable text found in the document")

        vectors = self._llm.embed(chunks)

        # Create the metadata row first so we have a document id for the vector payloads.
        doc = Document(
            filename=filename,
            content_type=content_type,
            chunking_strategy=strategy.value,
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            chunk_count=len(chunks),
            char_count=len(text),
        )
        db.add(doc)
        db.flush()  # populate doc.id without committing yet
        self._store.upsert(doc.id, filename, chunks, vectors)
        db.commit()  # commit only after vectors are stored, so failures leave no orphan row

        return IngestionResponse(
            document_id=doc.id,
            filename=filename,
            chunking_strategy=strategy,
            chunk_count=len(chunks),
            char_count=len(text),
        )
