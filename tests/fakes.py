"""Test doubles that replace the LLM, Redis memory and RAG pipeline."""

import json

from app.schemas.chat import BookingSlots, SourceChunk


class FakeLLM:
    """Returns queued chat replies in order; embeds every text to a constant vector."""

    def __init__(self, replies: list[str] | None = None) -> None:
        self.replies = list(replies or [])
        self.calls: list[list[dict[str, str]]] = []  # records prompts for assertions

    def embed(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    def chat(self, messages: list[dict[str, str]], json_mode: bool = False) -> str:
        self.calls.append(messages)
        # Pop the next scripted reply; empty string if the script ran out.
        return self.replies.pop(0) if self.replies else ""


def extraction_json(**fields: object) -> str:
    """Build the JSON string the LLM would return for booking extraction."""
    base = {"wants_booking": False, "cancel": False, "name": None, "email": None, "date": None, "time": None}
    return json.dumps({**base, **fields})


class FakeMemory:
    """In-process stand-in for the Redis-backed ChatMemory."""

    def __init__(self) -> None:
        self.history: dict[str, list[dict[str, str]]] = {}
        self.booking: dict[str, BookingSlots] = {}

    def get_history(self, sid: str) -> list[dict[str, str]]:
        return list(self.history.get(sid, []))

    def append(self, sid: str, role: str, content: str) -> None:
        self.history.setdefault(sid, []).append({"role": role, "content": content})

    def get_booking_state(self, sid: str) -> BookingSlots | None:
        return self.booking.get(sid)

    def set_booking_state(self, sid: str, slots: BookingSlots) -> None:
        self.booking[sid] = slots

    def clear_booking_state(self, sid: str) -> None:
        self.booking.pop(sid, None)


class FakeRag:
    """Returns a fixed answer and records which questions reached the RAG pipeline."""

    def __init__(self, answer: str = "rag answer") -> None:
        self._answer = answer
        self.questions: list[str] = []

    def answer(self, history: list[dict[str, str]], message: str) -> tuple[str, list[SourceChunk]]:
        self.questions.append(message)
        return self._answer, []
