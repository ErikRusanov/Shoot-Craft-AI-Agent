"""SSE relay — tails the EventBus and frames events for EventSource clients.

Each frame carries the bus stream id as the SSE ``id:``, so a dropped client
reconnects with ``Last-Event-ID`` and resumes strictly after what it saw.
The stream closes itself on a terminal event — `tail` is infinite by contract,
the lifecycle is not.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from sse_starlette import ServerSentEvent

from protocols.event_bus import EventBus
from schemas import EventAdapter

TERMINAL_EVENT_TYPES = frozenset({"done", "failed"})


async def session_event_stream(
    bus: EventBus, session_key: str, *, last_id: str | None = None
) -> AsyncIterator[ServerSentEvent]:
    async for item in bus.tail(session_key, last_id=last_id):
        yield ServerSentEvent(
            id=item.id,
            event=item.event.type,
            data=EventAdapter.dump_json(item.event).decode(),
        )
        if item.event.type in TERMINAL_EVENT_TYPES:
            return
