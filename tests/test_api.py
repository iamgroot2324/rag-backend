"""HTTP-level tests using FastAPI's TestClient with all external services faked."""

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api import chat, ingestion
from app.core.deps import get_chat_service, get_ingestion_service
from app.db.models import Document
from app.db.session import get_db
from app.services.booking import BookingService
from app.services.chat import ChatService
from app.services.ingestion import IngestionService
from tests.fakes import FakeLLM, FakeMemory, FakeRag, extraction_json


class FakeStore:
    """Vector store double that just records what would have been stored."""

    def __init__(self) -> None:
        self.upserts: list[tuple[str, str, list[str]]] = []

    def upsert(self, document_id: str, filename: str, chunks: list[str], vectors: list[list[float]]) -> None:
        assert len(chunks) == len(vectors)  # every chunk must have a vector
        self.upserts.append((document_id, filename, chunks))


@pytest.fixture
def store() -> FakeStore:
    return FakeStore()


@pytest.fixture
def client(db: Session, store: FakeStore) -> Iterator[TestClient]:
    """A test app with routers mounted and dependencies overridden by fakes."""
    app = FastAPI()
    app.include_router(ingestion.router)
    app.include_router(chat.router)

    ingestion_service = IngestionService(FakeLLM(), store)  # type: ignore[arg-type]
    chat_service = ChatService(
        FakeMemory(), FakeRag(), BookingService(FakeLLM([extraction_json()]))  # type: ignore[arg-type]
    )
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_ingestion_service] = lambda: ingestion_service
    app.dependency_overrides[get_chat_service] = lambda: chat_service
    yield TestClient(app)


def test_upload_txt_stores_vectors_and_metadata(client: TestClient, store: FakeStore, db: Session) -> None:
    body = "Sentence one. " * 100
    r = client.post(
        "/api/v1/documents",
        files={"file": ("doc.txt", body.encode(), "text/plain")},
        data={"strategy": "sentence", "chunk_size": "300", "chunk_overlap": "50"},
    )
    assert r.status_code == 201
    payload = r.json()
    assert payload["chunking_strategy"] == "sentence"
    # The reported chunk count matches what was sent to the vector store.
    assert payload["chunk_count"] == len(store.upserts[0][2]) > 1
    # Metadata row exists in the SQL database.
    assert db.get(Document, payload["document_id"]) is not None


def test_upload_rejects_unsupported_type(client: TestClient) -> None:
    r = client.post("/api/v1/documents", files={"file": ("x.png", b"data", "image/png")})
    assert r.status_code == 415


def test_upload_rejects_empty_file(client: TestClient) -> None:
    r = client.post("/api/v1/documents", files={"file": ("e.txt", b"   ", "text/plain")})
    assert r.status_code == 422


def test_upload_rejects_bad_strategy(client: TestClient) -> None:
    r = client.post(
        "/api/v1/documents",
        files={"file": ("a.txt", b"hello there.", "text/plain")},
        data={"strategy": "magic"},  # not in the ChunkStrategy enum
    )
    assert r.status_code == 422


def test_upload_rejects_overlap_ge_size(client: TestClient) -> None:
    r = client.post(
        "/api/v1/documents",
        files={"file": ("a.txt", b"hello there.", "text/plain")},
        data={"chunk_size": "200", "chunk_overlap": "300"},
    )
    assert r.status_code == 422


def test_chat_returns_answer(client: TestClient) -> None:
    r = client.post("/api/v1/chat", json={"session_id": "abc", "message": "hi"})
    assert r.status_code == 200
    assert r.json()["answer"] == "rag answer"


def test_chat_validates_input(client: TestClient) -> None:
    # An empty message violates the min_length constraint.
    assert client.post("/api/v1/chat", json={"session_id": "abc", "message": ""}).status_code == 422
