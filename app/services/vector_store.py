"""Qdrant-backed vector store for chunk embeddings."""

import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.core.config import Settings


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by a similarity search, with its source info and score."""

    document_id: str
    filename: str
    chunk_index: int
    text: str
    score: float


class VectorStore:
    """Stores chunk vectors and runs similarity search."""

    def __init__(self, settings: Settings) -> None:
        self._client = QdrantClient(url=settings.qdrant_url)
        self._collection = settings.qdrant_collection
        self._dim = settings.embedding_dim

    def ensure_collection(self) -> None:
        """Create the collection on first run (cosine distance suits text embeddings)."""
        if not self._client.collection_exists(self._collection):
            self._client.create_collection(
                self._collection,
                vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE),
            )

    def upsert(self, document_id: str, filename: str, chunks: list[str], vectors: list[list[float]]) -> None:
        """Store each chunk's vector with its text and origin as payload."""
        points = [
            PointStruct(
                id=str(uuid.uuid4()),  # Qdrant point ids must be unsigned ints or UUIDs
                vector=vec,
                payload={"document_id": document_id, "filename": filename, "chunk_index": i, "text": chunk},
            )
            for i, (chunk, vec) in enumerate(zip(chunks, vectors))
        ]
        self._client.upsert(self._collection, points=points)

    def search(self, vector: list[float], top_k: int) -> list[RetrievedChunk]:
        """Return the top_k chunks most similar to the query vector."""
        hits = self._client.query_points(self._collection, query=vector, limit=top_k, with_payload=True).points
        return [
            RetrievedChunk(
                document_id=str(h.payload["document_id"]),
                filename=str(h.payload["filename"]),
                chunk_index=int(h.payload["chunk_index"]),
                text=str(h.payload["text"]),
                score=float(h.score),
            )
            for h in hits
            if h.payload  # skip any point without a payload
        ]
