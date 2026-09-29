"""Tests for booking slot extraction, validation and persistence."""

from datetime import date, time, timedelta

from sqlalchemy.orm import Session

from app.db.models import Booking
from app.schemas.chat import BookingSlots
from app.services.booking import BookingService, Extraction
from tests.fakes import FakeLLM, extraction_json


def test_merge_accumulates_slots(future_date: str) -> None:
    # Turn 1: name and email arrive; name is trimmed.
    slots, invalid = BookingService.merge(None, Extraction(name=" Sugam ", email="s@example.com"))
    assert slots.name == "Sugam" and slots.email == "s@example.com"
    assert invalid == [] and slots.missing() == ["date", "time"]

    # Turn 2: date and time complete the set.
    slots, _ = BookingService.merge(slots, Extraction(date=future_date, time="15:00"))
    assert slots.missing() == []


def test_merge_rejects_invalid_values() -> None:
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    slots, invalid = BookingService.merge(
        None, Extraction(email="not-an-email", date=yesterday, time="25:99")
    )
    # Invalid values must not be stored, and each one is reported.
    assert slots.email is None and slots.date is None and slots.time is None
    assert len(invalid) == 3


def test_merge_bad_date_format() -> None:
    _, invalid = BookingService.merge(None, Extraction(date="next friday"))
    assert invalid == ["date"]


def test_merge_keeps_existing_when_new_value_missing(future_date: str) -> None:
    current = BookingSlots(name="A", email="a@example.com", date=future_date)
    slots, _ = BookingService.merge(current, Extraction(time="09:30"))
    assert slots.name == "A" and slots.time == "09:30"  # old values kept, new one added


def test_save_persists_booking(db: Session, future_date: str) -> None:
    slots = BookingSlots(name="Sugam", email="s@example.com", date=future_date, time="14:30")
    confirmation = BookingService.save(db, "sess-1", slots)

    # The row must exist in the database with the right values.
    row = db.get(Booking, confirmation.booking_id)
    assert row is not None
    assert row.session_id == "sess-1"
    assert row.interview_time == time(14, 30)
    assert confirmation.email == "s@example.com"


def test_extract_parses_llm_json() -> None:
    svc = BookingService(FakeLLM([extraction_json(wants_booking=True, name="Sugam")]))  # type: ignore[arg-type]
    ex = svc.extract([], "book me an interview", None)
    assert ex.wants_booking and ex.name == "Sugam"


def test_extract_falls_back_on_garbage() -> None:
    # Non-JSON model output must not crash the chat; it yields an empty extraction.
    svc = BookingService(FakeLLM(["not json at all"]))  # type: ignore[arg-type]
    ex = svc.extract([], "hello", None)
    assert ex == Extraction()
