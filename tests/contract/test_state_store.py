"""Contract: :class:`~protocols.state_store.StateStore`.

Pins the five responsibilities — face/session round-trips, token-fenced locking,
set-if-absent idempotency, and atomic budget reservation — independently of the
backing store. The ``store`` fixture (tests/conftest.py) parametrizes every test
over all implementations: in-memory and Redis.
"""

from __future__ import annotations

import asyncio

from protocols import StateStore
from schemas import (
    FaceProfile,
    FrameMetrics,
    GateReason,
    SessionState,
    Verdict,
)

TTL = 60


def _face(face_key: str = "face-1") -> FaceProfile:
    return FaceProfile(
        face_key=face_key,
        embedding=[0.1, 0.2, 0.3],
        gate_verdict=Verdict.PASSED,
        gate_reason=GateReason.OK,
        metrics=FrameMetrics(
            face_count=1,
            face_area_ratio=0.3,
            blur_var=120.0,
            yaw=2.0,
            pitch=-1.0,
            roll=0.5,
            brightness=128.0,
            width=1024,
            height=1024,
        ),
        photo_ref="photos/face-1.jpg",
    )


def _session(session_key: str = "sess-1", face_key: str = "face-1") -> SessionState:
    return SessionState(session_key=session_key, face_key=face_key)


def test_is_a_state_store(store: StateStore) -> None:
    assert isinstance(store, StateStore)


async def test_face_roundtrip(store: StateStore) -> None:
    assert await store.get_face("face-1") is None
    face = _face()
    await store.put_face(face, ttl_seconds=TTL)
    got = await store.get_face("face-1")
    assert got is not None and got == face


async def test_session_roundtrip(store: StateStore) -> None:
    assert await store.get_session("sess-1") is None
    session = _session()
    await store.put_session(session, ttl_seconds=TTL)
    got = await store.get_session("sess-1")
    assert got is not None and got == session


async def test_put_face_is_snapshot(store: StateStore) -> None:
    # Mutating the caller's copy after put must not change what was stored.
    face = _face()
    await store.put_face(face, ttl_seconds=TTL)
    face.embedding.append(9.9)
    got = await store.get_face("face-1")
    assert got is not None and got.embedding == [0.1, 0.2, 0.3]


async def test_lock_is_exclusive_and_token_fenced(store: StateStore) -> None:
    assert await store.acquire_lock("sess-1", token="a", ttl_seconds=TTL) is True
    # Held: a different holder cannot take it.
    assert await store.acquire_lock("sess-1", token="b", ttl_seconds=TTL) is False
    # Releasing with the wrong token is a no-op.
    assert await store.release_lock("sess-1", token="b") is False
    # The holder releases, then the lock is free again.
    assert await store.release_lock("sess-1", token="a") is True
    assert await store.acquire_lock("sess-1", token="b", ttl_seconds=TTL) is True


async def test_idempotency_store_once(store: StateStore) -> None:
    assert await store.idem_get("idem-1") is None
    assert await store.idem_put("idem-1", b"result-A", ttl_seconds=TTL) is True
    # Second put for the same key loses — the first result stands.
    assert await store.idem_put("idem-1", b"result-B", ttl_seconds=TTL) is False
    assert await store.idem_get("idem-1") == b"result-A"


async def test_budget_reserve_stops_at_limit(store: StateStore) -> None:
    # Reserve a fixed micro-USD estimate until the next one would exceed the limit.
    ok = [
        await store.budget_reserve("sess-1", estimate_micro=1, limit_micro=3, ttl_seconds=TTL)
        for _ in range(5)
    ]
    assert ok == [True, True, True, False, False]


async def test_budget_reserve_rejects_estimate_over_limit(store: StateStore) -> None:
    # A single estimate larger than the whole limit is refused outright.
    assert (
        await store.budget_reserve("sess-1", estimate_micro=5, limit_micro=3, ttl_seconds=TTL)
        is False
    )


async def test_budget_adjust_settles_and_floors(store: StateStore) -> None:
    assert await store.budget_reserve("sess-1", estimate_micro=80, limit_micro=100, ttl_seconds=TTL)
    # Settle the reservation down to the real cost (delta = actual - estimate).
    assert await store.budget_adjust("sess-1", delta_micro=-13, ttl_seconds=TTL) == 67
    # A refund can never drive the counter negative — it floors at zero.
    assert await store.budget_adjust("sess-1", delta_micro=-1000, ttl_seconds=TTL) == 0


async def test_budget_is_atomic_under_concurrency(store: StateStore) -> None:
    # Exactly `limit / estimate` concurrent reservations may win — never one more.
    results = await asyncio.gather(
        *(
            store.budget_reserve("sess-1", estimate_micro=1, limit_micro=4, ttl_seconds=TTL)
            for _ in range(20)
        )
    )
    assert sum(results) == 4


async def test_budget_is_per_session(store: StateStore) -> None:
    assert await store.budget_reserve("sess-1", estimate_micro=1, limit_micro=1, ttl_seconds=TTL)
    assert not await store.budget_reserve(
        "sess-1", estimate_micro=1, limit_micro=1, ttl_seconds=TTL
    )
    # A different session has its own counter.
    assert await store.budget_reserve("sess-2", estimate_micro=1, limit_micro=1, ttl_seconds=TTL)
