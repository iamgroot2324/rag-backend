"""Redis-backed chat memory and in-progress booking state."""

import json

import redis

from app.core.config import Settings
from app.schemas.chat import BookingSlots


class ChatMemory:
    """Stores per-session chat history and partially collected booking details."""

    def __init__(self, settings: Settings) -> None:
        # decode_responses=True returns str instead of bytes.
        self._r = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        self._limit = settings.chat_history_limit
        self._ttl = settings.chat_ttl_seconds

    def _hist_key(self, sid: str) -> str:
        return f"chat:history:{sid}"

    def _book_key(self, sid: str) -> str:
        return f"chat:booking:{sid}"

    def get_history(self, sid: str) -> list[dict[str, str]]:
        """Return the session's messages oldest-first, as {"role", "content"} dicts."""
        raw = self._r.lrange(self._hist_key(sid), 0, -1)
        return [json.loads(item) for item in raw]  # type: ignore[union-attr]

    def append(self, sid: str, role: str, content: str) -> None:
        """Append a message, keep only the last N, and refresh the expiry (atomically)."""
        key = self._hist_key(sid)
        pipe = self._r.pipeline()
        pipe.rpush(key, json.dumps({"role": role, "content": content}))
        pipe.ltrim(key, -self._limit, -1)  # trim to the most recent N messages
        pipe.expire(key, self._ttl)
        pipe.execute()

    def get_booking_state(self, sid: str) -> BookingSlots | None:
        """Return the in-progress booking slots, or None if no booking is active."""
        raw = self._r.get(self._book_key(sid))
        return BookingSlots.model_validate_json(raw) if raw else None  # type: ignore[arg-type]

    def set_booking_state(self, sid: str, slots: BookingSlots) -> None:
        """Save booking slots so they survive until the next turn."""
        self._r.set(self._book_key(sid), slots.model_dump_json(), ex=self._ttl)

    def clear_booking_state(self, sid: str) -> None:
        """Forget the in-progress booking (after completion or cancellation)."""
        self._r.delete(self._book_key(sid))
