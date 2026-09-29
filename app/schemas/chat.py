"""Request/response schemas for the conversational RAG API."""

from datetime import date, time

from pydantic import BaseModel, EmailStr, Field


class ChatRequest(BaseModel):
    """Incoming chat message. The session_id ties turns of one conversation together."""

    session_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=4000)


class SourceChunk(BaseModel):
    """A retrieved chunk that supported the answer (for transparency/citations)."""

    document_id: str
    filename: str
    chunk_index: int
    score: float  # cosine similarity from the vector search


class BookingConfirmation(BaseModel):
    """Details of a booking that was just saved."""

    booking_id: str
    name: str
    email: EmailStr
    interview_date: date
    interview_time: time


class ChatResponse(BaseModel):
    """Reply to a chat message; `booking` is set only on the turn that completes a booking."""

    session_id: str
    answer: str
    sources: list[SourceChunk] = []
    booking: BookingConfirmation | None = None


class BookingSlots(BaseModel):
    """Partially collected booking details (kept in Redis between turns)."""

    name: str | None = None
    email: str | None = None
    date: str | None = None  # YYYY-MM-DD
    time: str | None = None  # HH:MM (24h)

    def missing(self) -> list[str]:
        """Names of the fields that still need to be collected, in a fixed order."""
        return [f for f in ("name", "email", "date", "time") if not getattr(self, f)]
