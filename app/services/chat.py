
"""Chat orchestration: decides whether each turn is a booking step or a RAG question."""

from sqlalchemy.orm import Session

from app.schemas.chat import BookingSlots, ChatRequest, ChatResponse
from app.services.booking import BookingService, Extraction
from app.services.memory import ChatMemory
from app.services.rag import RagService


# Human-friendly wording used when asking the user for a missing booking field.
_LABELS = {
    "name": "your full name",
    "email": "your email address",
    "date": "the interview date",
    "time": "the preferred time",
}


class ChatService:
    """Routes each turn to either the booking flow or the RAG pipeline."""

    def __init__(
        self,
        memory: ChatMemory,
        rag: RagService,
        booking: BookingService,
    ) -> None:
        self._memory = memory
        self._rag = rag
        self._booking = booking

    def handle(self, req: ChatRequest, db: Session) -> ChatResponse:
        """Process one user message and return the assistant's reply."""
        sid = req.session_id
        history = self._memory.get_history(sid)
        state = self._memory.get_booking_state(sid)

        # Ask the LLM whether the message contains booking intent/details.
        extraction = self._booking.extract(history, req.message, state)

        # Make obvious booking requests reliable even if the small local
        # model fails to set wants_booking=true.
        if self._looks_like_booking_request(req.message):
            extraction.wants_booking = True

        # The user changed their mind mid-booking.
        if state is not None and extraction.cancel:
            self._memory.clear_booking_state(sid)
            return self._finish(
                sid,
                req.message,
                "No problem, I've cancelled the booking request.",
            )

        # Continue an active booking, or start one if the user just asked for it.
        if state is not None or extraction.wants_booking:
            return self._booking_turn(req, db, state, extraction)

        # Otherwise it's a normal question: answer with RAG.
        answer, sources = self._rag.answer(history, req.message)
        self._memory.append(sid, "user", req.message)
        self._memory.append(sid, "assistant", answer)

        return ChatResponse(
            session_id=sid,
            answer=answer,
            sources=sources,
        )

    @staticmethod
    def _looks_like_booking_request(message: str) -> bool:
        """Detect very clear booking requests without relying only on the LLM."""
        text = message.lower().strip()

        booking_phrases = (
            "book an interview",
            "book interview",
            "book my interview",
            "schedule an interview",
            "schedule interview",
            "schedule my interview",
            "want to book an interview",
            "want to schedule an interview",
            "i want to book",
            "i want to schedule",
        )

        return any(phrase in text for phrase in booking_phrases)

    def _booking_turn(
        self,
        req: ChatRequest,
        db: Session,
        state: BookingSlots | None,
        extraction: Extraction,
    ) -> ChatResponse:
        """Handle one step of the booking conversation."""
        sid = req.session_id

        slots, invalid = self._booking.merge(
            state or BookingSlots(),
            extraction,
        )

        missing = slots.missing()

        # All four fields collected and valid: save and confirm.
        if not missing:
            confirmation = self._booking.save(db, sid, slots)
            self._memory.clear_booking_state(sid)

            text = (
                f"Your interview is booked, {confirmation.name}! "
                f"{confirmation.interview_date} at "
                f"{confirmation.interview_time.strftime('%H:%M')}. "
                f"A confirmation will go to {confirmation.email}."
            )

            resp = self._finish(sid, req.message, text)
            resp.booking = confirmation
            return resp

        # Still missing something: persist progress and ask for the remaining fields.
        self._memory.set_booking_state(sid, slots)

        prefix = (
            f"That {' and '.join(invalid)} doesn't look valid. "
            if invalid
            else ""
        )

        ask = ", ".join(_LABELS[m] for m in missing)

        return self._finish(
            sid,
            req.message,
            f"{prefix}To book your interview, please share {ask}.",
        )

    def _finish(
        self,
        sid: str,
        user_msg: str,
        answer: str,
    ) -> ChatResponse:
        """Record the exchange in chat memory and build a plain response."""
        self._memory.append(sid, "user", user_msg)
        self._memory.append(sid, "assistant", answer)

        return ChatResponse(
            session_id=sid,
            answer=answer,
        )

