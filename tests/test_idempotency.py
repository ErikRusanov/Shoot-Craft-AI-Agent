"""IdempotencyService: run-once semantics over the store's set-if-absent.

The interesting case is the lost put race — two concurrent calls with one
``idem_key`` must converge on a single canonical result.
"""

from __future__ import annotations

import pytest

from services.connectors import InMemoryStateStore
from services.idempotency import IdempotencyService

TTL = 60


async def test_first_call_executes_and_stores() -> None:
    svc = IdempotencyService(InMemoryStateStore())
    calls = 0

    async def op() -> bytes:
        nonlocal calls
        calls += 1
        return b"result-A"

    result, replayed = await svc.run_once("idem-1", ttl_seconds=TTL, op=op)
    assert (result, replayed) == (b"result-A", False)
    assert calls == 1


async def test_second_call_replays_without_executing() -> None:
    svc = IdempotencyService(InMemoryStateStore())

    async def first() -> bytes:
        return b"result-A"

    async def second() -> bytes:  # pragma: no cover — must never run
        raise AssertionError("op re-executed for a replayed idem_key")

    await svc.run_once("idem-1", ttl_seconds=TTL, op=first)
    result, replayed = await svc.run_once("idem-1", ttl_seconds=TTL, op=second)
    assert (result, replayed) == (b"result-A", True)


async def test_lost_put_race_returns_the_winner() -> None:
    # A concurrent call sneaks its result in between our idem_get miss and our
    # idem_put — the loser must discard its own result and return the winner's.
    class RacingStore(InMemoryStateStore):
        async def idem_put(self, key: str, value: bytes, *, ttl_seconds: int) -> bool:
            await super().idem_put(key, b"winner", ttl_seconds=ttl_seconds)
            return await super().idem_put(key, value, ttl_seconds=ttl_seconds)

    svc = IdempotencyService(RacingStore())

    async def op() -> bytes:
        return b"loser"

    result, replayed = await svc.run_once("idem-1", ttl_seconds=TTL, op=op)
    assert (result, replayed) == (b"winner", True)


async def test_vanished_record_after_lost_race_is_loud() -> None:
    # Pathological: lost the put race *and* the winner's record expired before
    # we could read it back. Two divergent results may exist — refuse to guess.
    class VanishingStore(InMemoryStateStore):
        async def idem_put(self, key: str, value: bytes, *, ttl_seconds: int) -> bool:
            return False

    svc = IdempotencyService(VanishingStore())

    async def op() -> bytes:
        return b"orphan"

    with pytest.raises(RuntimeError, match="idempotency record vanished"):
        await svc.run_once("idem-1", ttl_seconds=TTL, op=op)
