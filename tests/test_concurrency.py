"""Reliability mechanics under concurrency and across a Redis restart.

The properties the loop's correctness rests on:

- budget never overdraws, no matter how many reservations race;
- the lock actually serializes a read-modify-write critical section;
- a tail can disconnect/reconnect by id forever and still see every event
  exactly once, even while a publisher keeps writing;
- locks expire (crash recovery: a dead holder cannot wedge the session);
- state, budget and streams survive a Redis restart (AOF) — including the
  Lua script cache being flushed.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Iterator
from typing import cast

import pytest
from redis.asyncio import Redis
from redis.exceptions import RedisError
from testcontainers.redis import RedisContainer

from protocols import EventBus, StateStore
from protocols.event_bus import StreamedEvent
from schemas import DoneEvent, FsmState, SessionState, StageEvent
from services.connectors import RedisEventBus, RedisStateStore
from tests.conftest import REDIS_IMAGE

TTL = 60


async def _drain(
    bus: EventBus, session_key: str, *, last_id: str | None, n: int
) -> list[StreamedEvent]:
    """Read exactly ``n`` events and close the tail, as one bounded operation."""
    agen = cast("AsyncGenerator[StreamedEvent]", bus.tail(session_key, last_id=last_id))
    out: list[StreamedEvent] = []

    async def _read() -> None:
        async for item in agen:
            out.append(item)
            if len(out) == n:
                return

    try:
        await asyncio.wait_for(_read(), timeout=10.0)
    finally:
        await agen.aclose()
    return out


# --- budget under contention ---


async def test_parallel_budget_never_overdraws(store: StateStore) -> None:
    limit = 10
    results = await asyncio.gather(
        *(store.check_and_incr_budget("sess-1", limit=limit, ttl_seconds=TTL) for _ in range(60))
    )
    assert sum(results) == limit
    # The counter is saturated for good — a late retry still gets nothing.
    assert await store.check_and_incr_budget("sess-1", limit=limit, ttl_seconds=TTL) is False


# --- lock serialization ---


async def test_lock_serializes_critical_section(store: StateStore) -> None:
    workers = 20
    counter = 0

    async def bump(i: int) -> None:
        nonlocal counter
        token = f"worker-{i}"
        # Polling is the point: a distributed lock has no local waiter to park
        # on — contenders retry, exactly like the real loop will.
        while not await store.acquire_lock("sess-1", token=token, ttl_seconds=TTL):  # noqa: ASYNC110
            await asyncio.sleep(0.002)
        try:
            # A lockless read-modify-write would interleave at this await and
            # lose updates; the lock must make it appear atomic.
            snapshot = counter
            await asyncio.sleep(0)
            counter = snapshot + 1
        finally:
            assert await store.release_lock("sess-1", token=token) is True

    await asyncio.gather(*(bump(i) for i in range(workers)))
    assert counter == workers


async def test_lock_expires_and_can_be_retaken(redis_url: str) -> None:
    # Redis-only: TTL expiry is the crash-recovery path (the in-memory store
    # ignores TTLs by contract). A holder that died must not wedge the session.
    client = Redis.from_url(redis_url)
    await client.flushdb()
    try:
        store = RedisStateStore(client)
        assert await store.acquire_lock("sess-1", token="crashed", ttl_seconds=1) is True
        assert await store.acquire_lock("sess-1", token="next", ttl_seconds=TTL) is False
        await asyncio.sleep(1.2)
        assert await store.acquire_lock("sess-1", token="next", ttl_seconds=TTL) is True
        # The expired holder's release must not touch the new owner's lock.
        assert await store.release_lock("sess-1", token="crashed") is False
    finally:
        await client.aclose()


# --- stream resume ---


async def test_reconnect_chain_sees_every_event_exactly_once(bus: EventBus) -> None:
    total = 30
    published: list[str] = []

    async def publisher() -> None:
        for _ in range(total):
            published.append(await bus.publish("sess-1", StageEvent(stage=FsmState.GENERATING)))
            await asyncio.sleep(0.001)

    pub = asyncio.create_task(publisher())
    try:
        received: list[str] = []
        last_id: str | None = None
        # Reconnect every few events, like an SSE client that keeps dropping.
        while len(received) < total:
            chunk = await _drain(bus, "sess-1", last_id=last_id, n=min(7, total - len(received)))
            received.extend(e.id for e in chunk)
            last_id = received[-1]
    finally:
        await pub

    assert received == published


# --- surviving a Redis restart ---


@pytest.fixture
def persistent_redis() -> Iterator[RedisContainer]:
    """A dedicated Redis with synchronous AOF, so a restart loses nothing."""
    container = RedisContainer(REDIS_IMAGE).with_command(
        "redis-server --appendonly yes --appendfsync always"
    )
    try:
        container.start()
    except Exception:
        pytest.skip("Docker is not available — skipping Redis restart test")
    try:
        yield container
    finally:
        container.stop()


async def _await_redis(client: Redis) -> None:
    async with asyncio.timeout(30.0):
        while True:
            try:
                # ping() is typed for both sync and async clients — narrow it.
                await cast("Awaitable[bool]", client.ping())
                return
            # redis-py's ConnectionError is a RedisError, not the builtin;
            # BusyLoadingError (AOF replay in progress) is one too.
            except RedisError, OSError:
                await asyncio.sleep(0.2)


def _container_url(container: RedisContainer) -> str:
    host = container.get_container_host_ip()
    port = container.get_exposed_port(container.port)
    return f"redis://{host}:{port}/0"


async def test_state_budget_and_stream_survive_redis_restart(
    persistent_redis: RedisContainer,
) -> None:
    client = Redis.from_url(_container_url(persistent_redis))
    try:
        store = RedisStateStore(client)
        bus = RedisEventBus(client)

        session = SessionState(session_key="sess-1", face_key="face-1")
        await store.put_session(session, ttl_seconds=3600)
        assert await store.check_and_incr_budget("sess-1", limit=3, ttl_seconds=3600) is True
        ids = [await bus.publish("sess-1", StageEvent(stage=FsmState.PLANNING)) for _ in range(3)]
    finally:
        await client.aclose()

    persistent_redis.get_wrapped_container().restart()
    # Docker re-publishes the container port on a random host port after the
    # restart, so the test must re-resolve the URL and reconnect — in prod the
    # address is stable and redis-py reconnects through the same client.
    client = Redis.from_url(_container_url(persistent_redis))
    try:
        await _await_redis(client)
        store = RedisStateStore(client)
        bus = RedisEventBus(client)

        # State is still there.
        assert await store.get_session("sess-1") == session
        # Budget resumes from the persisted count (1 used of 3) — and the Lua
        # script registers against the fresh, empty script cache.
        assert await store.check_and_incr_budget("sess-1", limit=3, ttl_seconds=3600) is True
        assert await store.check_and_incr_budget("sess-1", limit=3, ttl_seconds=3600) is True
        assert await store.check_and_incr_budget("sess-1", limit=3, ttl_seconds=3600) is False
        # The stream resumes by pre-restart id with nothing lost...
        got = await _drain(bus, "sess-1", last_id=ids[1], n=1)
        assert got[0].id == ids[2]
        # ...and accepts new events again.
        new_id = await bus.publish("sess-1", DoneEvent())
        got = await _drain(bus, "sess-1", last_id=ids[2], n=1)
        assert got[0].id == new_id
    finally:
        await client.aclose()
