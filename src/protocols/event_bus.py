"""Port: event bus — the per-session SSE stream, append + live tail.

Each session has one ordered stream ``events:{session_key}``; the API tails it
and relays :class:`~schemas.events.Event` over SSE. Every appended event gets an
opaque id, surfaced so a dropped SSE connection can reconnect via
``Last-Event-ID`` and resume **after** the last id it saw — no gaps, no replays.

``tail`` is a *live* iterator: it yields backlog after ``last_id`` and then keeps
yielding new events as they are published (the real backing is Redis ``XREAD``
with blocking). A consumer stops by closing the iterator.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import NamedTuple, Protocol, runtime_checkable

from schemas import Event


class StreamedEvent(NamedTuple):
    """An event with its stream id, so SSE can emit ``id:`` for reconnects."""

    id: str
    event: Event


@runtime_checkable
class EventBus(Protocol):
    """Append events to a session stream and tail it live."""

    async def publish(self, session_key: str, event: Event) -> str:
        """Append ``event`` to the session stream; return its assigned id."""
        ...

    def tail(self, session_key: str, *, last_id: str | None = None) -> AsyncIterator[StreamedEvent]:
        """Yield events strictly after ``last_id`` (or from the start if ``None``),
        then follow the stream live until the consumer stops."""
        ...
