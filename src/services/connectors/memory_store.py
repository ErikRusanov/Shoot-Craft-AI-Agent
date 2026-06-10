"""In-memory state store and event bus — the Redis-shaped surface, no Redis.

The no-Redis wiring: picked **at process start** when `Settings.redis_url` is
unset (single-process dev, tests). There is no runtime failover from Redis to
this — losing Redis mid-flight must surface as an error, not silently degrade
into state that vanishes with the process.

Both classes honor their ports' contracts (token-fenced locks, set-if-absent
idempotency, atomic-by-construction budget, live event tailing) but skip real
TTL expiry: ``ttl_seconds`` is accepted and ignored. They run in a single event
loop, which is exactly why atomicity holds for free — no method awaits between
read and write, so concurrent coroutines never interleave a check-and-increment.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from protocols.event_bus import StreamedEvent
from schemas import Event, EventAdapter, FaceProfile, SessionState


class InMemoryStateStore:
    """Faces, sessions, locks, idempotency keys and budget counters in dicts."""

    def __init__(self) -> None:
        self._faces: dict[str, FaceProfile] = {}
        self._sessions: dict[str, SessionState] = {}
        self._locks: dict[str, str] = {}  # key -> holder token
        self._idem: dict[str, bytes] = {}
        self._budget: dict[str, int] = {}  # session_key -> used count

    # --- face profile ---
    async def get_face(self, face_key: str) -> FaceProfile | None:
        return self._faces.get(face_key)

    async def put_face(self, face: FaceProfile, *, ttl_seconds: int) -> None:
        # Deep-copy via the model so a later caller mutation can't leak back in.
        self._faces[face.face_key] = face.model_copy(deep=True)

    # --- session state ---
    async def get_session(self, session_key: str) -> SessionState | None:
        return self._sessions.get(session_key)

    async def put_session(self, session: SessionState, *, ttl_seconds: int) -> None:
        self._sessions[session.session_key] = session.model_copy(deep=True)

    # --- token-fenced lock ---
    async def acquire_lock(self, key: str, *, token: str, ttl_seconds: int) -> bool:
        if key in self._locks:
            return False
        self._locks[key] = token
        return True

    async def release_lock(self, key: str, *, token: str) -> bool:
        if self._locks.get(key) != token:
            return False
        del self._locks[key]
        return True

    # --- idempotency ---
    async def idem_get(self, key: str) -> bytes | None:
        return self._idem.get(key)

    async def idem_put(self, key: str, value: bytes, *, ttl_seconds: int) -> bool:
        if key in self._idem:
            return False
        self._idem[key] = value
        return True

    # --- budget (atomic) ---
    async def check_and_incr_budget(
        self, session_key: str, *, limit: int, ttl_seconds: int
    ) -> bool:
        used = self._budget.get(session_key, 0)
        if used >= limit:
            return False
        self._budget[session_key] = used + 1
        return True


class InMemoryEventBus:
    """Per-session append-only streams with a live, blocking tail.

    Events are round-tripped through :data:`~schemas.EventAdapter` on publish so
    this bus exercises the same (de)serialization the Redis one does, and stream
    ids are simple 1-based counters — opaque to the consumer, resumable by value.
    """

    def __init__(self) -> None:
        self._streams: dict[str, list[StreamedEvent]] = {}
        self._cond = asyncio.Condition()

    async def publish(self, session_key: str, event: Event) -> str:
        reloaded: Event = EventAdapter.validate_json(EventAdapter.dump_json(event))
        async with self._cond:
            stream = self._streams.setdefault(session_key, [])
            eid = str(len(stream) + 1)
            stream.append(StreamedEvent(eid, reloaded))
            self._cond.notify_all()
            return eid

    async def tail(
        self, session_key: str, *, last_id: str | None = None
    ) -> AsyncIterator[StreamedEvent]:
        # Counter ids map to list positions: id "N" is at index N-1, so resuming
        # after it starts at index N. Unknown/None id → from the beginning.
        idx = int(last_id) if last_id is not None and last_id.isdigit() else 0
        while True:
            async with self._cond:
                stream = self._streams.setdefault(session_key, [])
                while idx >= len(stream):
                    await self._cond.wait()
                new = stream[idx:]
                idx = len(stream)
            for item in new:
                yield item
