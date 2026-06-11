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
- **budget** — a reserve/settle pair over a per-session micro-USD counter.
  ``budget_reserve`` atomically adds a conservative estimate if it fits under the
  limit (two concurrent reservations must not both slip past the last dollar —
  the real store does it in one Lua script); ``budget_adjust`` settles that
  reservation to the real cost (or refunds it) with a signed delta.

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

    # --- budget (reserve/settle, atomic) ---
    async def budget_reserve(
        self, session_key: str, *, estimate_micro: int, limit_micro: int, ttl_seconds: int
    ) -> bool:
        """Atomically reserve ``estimate_micro`` micro-USD against the limit.

        If the already-spent counter plus ``estimate_micro`` stays at or below
        ``limit_micro``, add the estimate and return ``True`` (the paid call may
        proceed); otherwise leave the counter and return ``False`` (budget
        exhausted). ``ttl_seconds`` bounds the counter to the session's lifetime.
        """
        ...

    async def budget_adjust(self, session_key: str, *, delta_micro: int, ttl_seconds: int) -> int:
        """Apply a signed micro-USD delta to the counter; return the new value.

        Settle a reservation to its real cost (``delta = actual - estimate``,
        usually negative) or refund it (``delta = -estimate``). The counter is
        floored at zero, so a refund can never drive it negative.
        """
        ...
