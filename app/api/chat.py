"""HTTP routes for the conversational RAG API."""

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.deps import get_chat_service
from app.db.session import get_db
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat import ChatService

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(
    body: ChatRequest,
    service: Annotated[ChatService, Depends(get_chat_service)],
    db: Annotated[Session, Depends(get_db)],
) -> ChatResponse:
    """Send a message in a session; answers questions or drives interview booking."""
    return service.handle(body, db)
