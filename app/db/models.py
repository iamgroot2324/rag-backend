"""ORM models for document metadata and interview bookings."""

import uuid
from datetime import date, datetime, time, timezone

from sqlalchemy import Date, DateTime, Integer, String, Time
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


def _uuid() -> str:
    """Generate a string UUID used as a primary key."""
    return str(uuid.uuid4())


def _now() -> datetime:
    """Current UTC time (timezone-aware) used for created_at columns."""
    return datetime.now(timezone.utc)


class Document(Base):
    """Metadata about an uploaded document (the text itself lives in the vector store)."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    filename: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str] = mapped_column(String(100))
    chunking_strategy: Mapped[str] = mapped_column(String(20))  # "fixed" or "sentence"
    chunk_size: Mapped[int] = mapped_column(Integer)
    chunk_overlap: Mapped[int] = mapped_column(Integer)
    chunk_count: Mapped[int] = mapped_column(Integer)
    char_count: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Booking(Base):
    """A confirmed interview booking collected through the chat."""

    __tablename__ = "bookings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(String(100), index=True)  # chat session that made it
    name: Mapped[str] = mapped_column(String(200))
    email: Mapped[str] = mapped_column(String(320))  # 320 = max valid email length
    interview_date: Mapped[date] = mapped_column(Date)
    interview_time: Mapped[time] = mapped_column(Time)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
