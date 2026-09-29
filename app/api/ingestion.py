"""HTTP routes for document ingestion."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.deps import get_ingestion_service
from app.db.session import get_db
from app.schemas.ingestion import ChunkStrategy, IngestionResponse
from app.services.extraction import SUPPORTED_EXTENSIONS, UnsupportedFileError
from app.services.ingestion import EmptyDocumentError, IngestionService

router = APIRouter(prefix="/api/v1", tags=["ingestion"])


@router.post("/documents", response_model=IngestionResponse, status_code=201)
def upload_document(
    file: Annotated[UploadFile, File(description=".pdf or .txt file")],
    service: Annotated[IngestionService, Depends(get_ingestion_service)],
    db: Annotated[Session, Depends(get_db)],
    strategy: Annotated[ChunkStrategy, Form()] = ChunkStrategy.FIXED,
    chunk_size: Annotated[int, Form(ge=100, le=5000)] = 1000,
    chunk_overlap: Annotated[int, Form(ge=0, le=1000)] = 150,
) -> IngestionResponse:
    """Upload a document, chunk it with the chosen strategy, embed it and store it."""
    filename = file.filename or "upload"

    # Reject unsupported types early, before reading the body.
    if not filename.lower().endswith(SUPPORTED_EXTENSIONS):
        raise HTTPException(415, "Only .pdf and .txt files are supported")

    data = file.file.read()
    if len(data) > get_settings().max_upload_bytes:
        raise HTTPException(413, "File too large")

    try:
        return service.ingest(
            db, filename, file.content_type or "application/octet-stream", data, strategy, chunk_size, chunk_overlap
        )
    except (UnsupportedFileError, EmptyDocumentError, ValueError) as exc:
        # Bad input (empty file, overlap >= size, ...) is a client error, not a server error.
        raise HTTPException(422, str(exc)) from exc
