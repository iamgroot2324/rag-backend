"""Interview booking: LLM-based slot extraction, validation and persistence."""

import json
from datetime import date, datetime, time

from pydantic import BaseModel, EmailStr, TypeAdapter, ValidationError
from sqlalchemy.orm import Session

from app.db.models import Booking
from app.schemas.chat import BookingConfirmation, BookingSlots
from app.services.llm import LLMClient

# Reusable validator for email addresses (pydantic's EmailStr).
_email_adapter: TypeAdapter[EmailStr] = TypeAdapter(EmailStr)

# Prompt asking the LLM to turn free-form chat into structured booking fields.
# Today's date is injected so relative dates like "tomorrow" can be resolved.
_EXTRACT_PROMPT = """You extract interview-booking details from a conversation.
Today's date is {today}. Return ONLY JSON with keys:
  "wants_booking": true if the user wants to book/schedule an interview (or is in the middle of doing so),
  "cancel": true if the user wants to abandon the booking,
  "name": full name or null,
  "email": email address or null,
  "date": date as YYYY-MM-DD (resolve words like "tomorrow") or null,
  "time": 24h time as HH:MM or null.
Only include values the user has actually stated. Already collected: {collected}"""


class Extraction(BaseModel):
    """Structured result of the LLM extraction step (all fields optional)."""

    wants_booking: bool = False
    cancel: bool = False
    name: str | None = None
    email: str | None = None
    date: str | None = None
    time: str | None = None


class BookingService:
    """Uses the LLM to fill booking slots across turns, validates them, and saves the result."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def extract(self, history: list[dict[str, str]], message: str, current: BookingSlots | None) -> Extraction:
        """Ask the LLM what booking info (if any) the latest message contains."""
        system = _EXTRACT_PROMPT.format(
            today=date.today().isoformat(),
            collected=current.model_dump_json() if current else "{}",
        )
        # Only the last few turns are needed for context.
        messages = [{"role": "system", "content": system}, *history[-6:], {"role": "user", "content": message}]
        try:
            return Extraction.model_validate(json.loads(self._llm.chat(messages, json_mode=True)))
        except (json.JSONDecodeError, ValidationError):
            # If the model returns malformed output, treat it as "nothing extracted".
            return Extraction()

    @staticmethod
    def merge(current: BookingSlots | None, ex: Extraction) -> tuple[BookingSlots, list[str]]:
        """Merge newly extracted values into the slots, validating each one.

        Returns the updated slots and a list of fields that failed validation.
        Invalid values are dropped so the user can be asked for them again.
        """
        slots = current.model_copy() if current else BookingSlots()
        invalid: list[str] = []

        if ex.name and ex.name.strip():
            slots.name = ex.name.strip()

        if ex.email:
            try:
                slots.email = str(_email_adapter.validate_python(ex.email))
            except ValidationError:
                invalid.append("email")

        if ex.date:
            try:
                parsed = date.fromisoformat(ex.date)
                if parsed < date.today():  # can't book in the past
                    invalid.append("date (must not be in the past)")
                else:
                    slots.date = parsed.isoformat()
            except ValueError:
                invalid.append("date")

        if ex.time:
            try:
                # Parse then re-format so the stored value is always normalised HH:MM.
                slots.time = datetime.strptime(ex.time, "%H:%M").strftime("%H:%M")
            except ValueError:
                invalid.append("time")

        return slots, invalid

    @staticmethod
    def save(db: Session, session_id: str, slots: BookingSlots) -> BookingConfirmation:
        """Persist a fully populated booking and return its confirmation."""
        # Callers only invoke this once no slots are missing.
        assert slots.name and slots.email and slots.date and slots.time
        booking = Booking(
            session_id=session_id,
            name=slots.name,
            email=slots.email,
            interview_date=date.fromisoformat(slots.date),
            interview_time=time.fromisoformat(slots.time),
        )
        db.add(booking)
        db.commit()
        return BookingConfirmation(
            booking_id=booking.id,
            name=booking.name,
            email=booking.email,
            interview_date=booking.interview_date,
            interview_time=booking.interview_time,
        )
