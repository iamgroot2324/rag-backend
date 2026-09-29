"""Tests for chat routing between RAG and the multi-turn booking flow."""

from sqlalchemy.orm import Session

from app.db.models import Booking
from app.schemas.chat import ChatRequest
from app.services.booking import BookingService
from app.services.chat import ChatService
from tests.fakes import FakeLLM, FakeMemory, FakeRag, extraction_json


def make_service(replies: list[str]) -> tuple[ChatService, FakeMemory, FakeRag]:
    """Build a ChatService whose LLM replies with the given scripted extraction JSONs."""
    memory, rag = FakeMemory(), FakeRag()
    booking = BookingService(FakeLLM(replies))  # type: ignore[arg-type]
    return ChatService(memory, rag, booking), memory, rag  # type: ignore[arg-type]


def req(msg: str, sid: str = "s1") -> ChatRequest:
    """Shortcut for building a chat request."""
    return ChatRequest(session_id=sid, message=msg)


def test_question_goes_to_rag_and_is_remembered(db: Session) -> None:
    svc, memory, rag = make_service([extraction_json()])  # no booking intent
    resp = svc.handle(req("What is the leave policy?"), db)

    assert resp.answer == "rag answer"
    assert rag.questions == ["What is the leave policy?"]
    # Both the user message and the reply are stored in chat memory.
    assert [m["role"] for m in memory.get_history("s1")] == ["user", "assistant"]


def test_multi_turn_booking_flow(db: Session, future_date: str) -> None:
    # One scripted extraction per turn: name -> email -> date/time.
    svc, memory, rag = make_service(
        [
            extraction_json(wants_booking=True, name="Sugam"),
            extraction_json(email="sugam@example.com"),
            extraction_json(date=future_date, time="15:00"),
        ]
    )

    # Turn 1: booking starts; the bot asks for the email next.
    r1 = svc.handle(req("I'd like to book an interview, I'm Sugam"), db)
    assert "email" in r1.answer and r1.booking is None
    assert memory.get_booking_state("s1") is not None

    # Turn 2: email supplied; date is still missing.
    r2 = svc.handle(req("sugam@example.com"), db)
    assert r2.booking is None and "date" in r2.answer

    # Turn 3: date and time supplied; booking is saved and state cleared.
    r3 = svc.handle(req("next week at 3pm"), db)
    assert r3.booking is not None
    assert r3.booking.name == "Sugam"
    assert memory.get_booking_state("s1") is None
    assert db.query(Booking).count() == 1
    assert rag.questions == []  # RAG was never used for booking turns


def test_invalid_email_is_reported(db: Session) -> None:
    svc, memory, _ = make_service(
        [extraction_json(wants_booking=True, name="A", email="nope")]
    )
    resp = svc.handle(req("book interview, I'm A, email nope"), db)
    assert "email" in resp.answer and "doesn't look valid" in resp.answer
    assert memory.get_booking_state("s1").email is None  # type: ignore[union-attr]


def test_cancel_clears_booking_state(db: Session) -> None:
    svc, memory, _ = make_service(
        [extraction_json(wants_booking=True, name="A"), extraction_json(cancel=True)]
    )
    svc.handle(req("book an interview, I'm A"), db)
    assert memory.get_booking_state("s1") is not None

    resp = svc.handle(req("never mind"), db)
    assert "cancelled" in resp.answer
    assert memory.get_booking_state("s1") is None
    assert db.query(Booking).count() == 0  # nothing was saved


def test_sessions_are_isolated(db: Session) -> None:
    svc, memory, _ = make_service(
        [extraction_json(wants_booking=True, name="A"), extraction_json()]
    )
    svc.handle(req("book interview, I'm A", sid="one"), db)
    svc.handle(req("hello", sid="two"), db)
    # A booking in session "one" must not leak into session "two".
    assert memory.get_booking_state("one") is not None
    assert memory.get_booking_state("two") is None
