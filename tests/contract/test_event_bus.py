"""Contract: :class:`~protocols.event_bus.EventBus`.

Pins ordering, ids, ``Last-Event-ID`` reconnect semantics and live following.
The bus must round-trip events through the wire form, so an event read back is a
fully-validated :class:`~schemas.events.Event`, not whatever object was published.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable

import pytest

from protocols import EventBus, StreamedEvent
from schemas import DoneEvent, FsmState, StageEvent
from tests.fakes import InMemoryEventBus

BUS_FACTORIES = [
    pytest.param(InMemoryEventBus, id="memory"),
]


@pytest.fixture(params=BUS_FACTORIES)
def bus(request: pytest.FixtureRequest) -> EventBus:
    factory: Callable[[], EventBus] = request.param
    return factory()


async def _take(it: AsyncIterator[StreamedEvent], n: int) -> list[StreamedEvent]:
    """Read exactly ``n`` events, bounded so a stalled tail fails loudly."""
    out: list[StreamedEvent] = []
    for _ in range(n):
        out.append(await asyncio.wait_for(it.__anext__(), timeout=2.0))
    return out


async def test_is_an_event_bus(bus: EventBus) -> None:
    assert isinstance(bus, EventBus)


async def test_publish_then_tail_preserves_order(bus: EventBus) -> None:
    await bus.publish("sess-1", StageEvent(stage=FsmState.PLANNING))
    await bus.publish("sess-1", StageEvent(stage=FsmState.GENERATING))
    await bus.publish("sess-1", DoneEvent())

    got = await _take(bus.tail("sess-1"), 3)

    assert [e.id for e in got] == ["1", "2", "3"]
    first = got[0].event
    assert isinstance(first, StageEvent) and first.stage is FsmState.PLANNING
    assert isinstance(got[2].event, DoneEvent)


async def test_streams_are_isolated_per_session(bus: EventBus) -> None:
    await bus.publish("sess-1", DoneEvent())
    await bus.publish("sess-2", StageEvent(stage=FsmState.FAILED))

    got = await _take(bus.tail("sess-2"), 1)
    assert isinstance(got[0].event, StageEvent)


async def test_reconnect_resumes_after_last_id(bus: EventBus) -> None:
    await bus.publish("sess-1", StageEvent(stage=FsmState.PLANNING))
    mid = await bus.publish("sess-1", StageEvent(stage=FsmState.GENERATING))
    await bus.publish("sess-1", DoneEvent())

    # Reconnect with the id we last saw: only what came after it should arrive.
    got = await _take(bus.tail("sess-1", last_id=mid), 1)
    assert got[0].id == "3"
    assert isinstance(got[0].event, DoneEvent)


async def test_tail_follows_live_publishes(bus: EventBus) -> None:
    # Tail starts before anything exists, then receives events as they land.
    it = bus.tail("sess-1")
    reader = asyncio.create_task(_take(it, 2))
    await asyncio.sleep(0)  # let the reader block on the empty stream
    await bus.publish("sess-1", StageEvent(stage=FsmState.GENERATING))
    await bus.publish("sess-1", DoneEvent())

    got = await asyncio.wait_for(reader, timeout=2.0)
    assert [type(e.event) for e in got] == [StageEvent, DoneEvent]
