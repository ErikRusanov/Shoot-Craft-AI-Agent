"""Port: session state store — the Redis-shaped surface, no relational DB.

Five responsibilities the loop leans on, all TTL-bound and keyed by
``face_key`` / ``session_key``:

- **face / session** aggregates — read/write of :class:`~schemas.state.FaceProfile`
  and :class:`~schemas.state.SessionState`. The key is taken from the model, so
  callers never thread it separately.
- **lock** — a token-fenced mutex around a session mutation. ``token`` is the
  fence: release only succeeds for the holder, so a lock that expired and was
  re-taken elsewhere can't be released out from under the new owner.
- **idempotency** — store-once/read of a mutation's result by ``idem_key`` so a
  retried request replays the original outcome instead of doing the work twice.
- **budget** — a single atomic check-and-increment of the paid-generation
  counter. Atomic because two concurrent attempts must not both slip past the
  last allowed slot (the real store does it in one Lua script).

Everything is ``async``; concrete stores keep the I/O path non-blocking. The
in-memory fallback honors the same contract minus real expiry.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from schemas import FaceProfile, SessionState


@runtime_checkable
class StateStore(Protocol):
    """State, locks, idempotency and budget for one session — behind one port."""

    # --- face profile, keyed by face.face_key ---
    async def get_face(self, face_key: str) -> FaceProfile | None:
        """The stored profile, or ``None`` if absent/expired."""
        ...

    async def put_face(self, face: FaceProfile, *, ttl_seconds: int) -> None:
        """Store ``face`` under its ``face_key`` with a fresh TTL."""
        ...

    # --- session state, keyed by session.session_key ---
    async def get_session(self, session_key: str) -> SessionState | None:
        """The stored session, or ``None`` if absent/expired."""
        ...

    async def put_session(self, session: SessionState, *, ttl_seconds: int) -> None:
        """Store ``session`` under its ``session_key`` with a fresh TTL."""
        ...

    # --- token-fenced lock ---
    async def acquire_lock(self, key: str, *, token: str, ttl_seconds: int) -> bool:
        """Take the lock for ``key`` if free; ``True`` on success, ``False`` if held."""
        ...

    async def release_lock(self, key: str, *, token: str) -> bool:
        """Release ``key`` only if ``token`` still holds it; ``True`` if released."""
        ...

    # --- idempotency ---
    async def idem_get(self, key: str) -> bytes | None:
        """The stored result for ``idem_key``, or ``None`` if this is the first time."""
        ...

    async def idem_put(self, key: str, value: bytes, *, ttl_seconds: int) -> bool:
        """Store ``value`` for ``key`` only if absent.

        ``True`` if stored, ``False`` if a value was already there (lost the race).
        """
        ...

    # --- budget (atomic) ---
    async def check_and_incr_budget(
        self, session_key: str, *, limit: int, ttl_seconds: int
    ) -> bool:
        """Atomically reserve one paid generation.

        If the counter is below ``limit``, increment it and return ``True`` (the
        generation may proceed and is charged); otherwise leave it and return
        ``False``. ``ttl_seconds`` bounds the counter to the session's lifetime.
        """
        ...
