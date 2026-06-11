"""Redis-backed :class:`~protocols.state_store.StateStore` (redis-py asyncio).

Key layout, all TTL-bound — biometrics and session state are transient:

- ``face:{face_key}``   — FaceProfile JSON
- ``sess:{session_key}`` — SessionState JSON
- ``lock:{key}``        — holder token (``SET NX PX``; release via Lua fence)
- ``idemp:{key}``       — stored mutation result (``SET NX``)
- ``budget:{session_key}`` — micro-USD spent counter (Lua reserve/adjust)

The client must be created with ``decode_responses=False`` (the default):
idempotency values are opaque bytes. Scripts run via ``register_script``, so a
Redis restart that flushes the script cache transparently falls back to EVAL.
"""

from __future__ import annotations

from redis.asyncio import Redis

from schemas import FaceProfile, SessionState
from utils.lua import BUDGET_ADJUST, BUDGET_RESERVE, RELEASE_LOCK_IF_HELD


class RedisStateStore:
    """State, locks, idempotency and budget on one Redis client."""

    def __init__(self, client: Redis) -> None:
        self._r = client
        self._release_lock = client.register_script(RELEASE_LOCK_IF_HELD)
        self._budget_reserve = client.register_script(BUDGET_RESERVE)
        self._budget_adjust = client.register_script(BUDGET_ADJUST)

    # --- face profile ---
    async def get_face(self, face_key: str) -> FaceProfile | None:
        raw = await self._r.get(f"face:{face_key}")
        return None if raw is None else FaceProfile.model_validate_json(raw)

    async def put_face(self, face: FaceProfile, *, ttl_seconds: int) -> None:
        await self._r.set(f"face:{face.face_key}", face.model_dump_json(), ex=ttl_seconds)

    # --- session state ---
    async def get_session(self, session_key: str) -> SessionState | None:
        raw = await self._r.get(f"sess:{session_key}")
        return None if raw is None else SessionState.model_validate_json(raw)

    async def put_session(self, session: SessionState, *, ttl_seconds: int) -> None:
        await self._r.set(f"sess:{session.session_key}", session.model_dump_json(), ex=ttl_seconds)

    # --- token-fenced lock ---
    async def acquire_lock(self, key: str, *, token: str, ttl_seconds: int) -> bool:
        # PX over EX: sub-second TTLs matter for lock handover in tests and
        # tight retry loops.
        res = await self._r.set(f"lock:{key}", token, nx=True, px=ttl_seconds * 1000)
        return bool(res)

    async def release_lock(self, key: str, *, token: str) -> bool:
        res = await self._release_lock(keys=[f"lock:{key}"], args=[token])
        return bool(res)

    # --- idempotency ---
    async def idem_get(self, key: str) -> bytes | None:
        raw = await self._r.get(f"idemp:{key}")
        return None if raw is None else bytes(raw)

    async def idem_put(self, key: str, value: bytes, *, ttl_seconds: int) -> bool:
        res = await self._r.set(f"idemp:{key}", value, nx=True, ex=ttl_seconds)
        return bool(res)

    # --- budget (reserve/settle, atomic via Lua) ---
    async def budget_reserve(
        self, session_key: str, *, estimate_micro: int, limit_micro: int, ttl_seconds: int
    ) -> bool:
        res = await self._budget_reserve(
            keys=[f"budget:{session_key}"], args=[estimate_micro, limit_micro, ttl_seconds]
        )
        return bool(res)

    async def budget_adjust(self, session_key: str, *, delta_micro: int, ttl_seconds: int) -> int:
        res = await self._budget_adjust(
            keys=[f"budget:{session_key}"], args=[delta_micro, ttl_seconds]
        )
        return int(res)
