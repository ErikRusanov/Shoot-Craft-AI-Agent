"""Redis Streams-backed :class:`~protocols.event_bus.EventBus`.

One stream per session — ``events:{session_key}`` — appended with ``XADD``
(capped with approximate ``MAXLEN`` so a chatty session can't grow unbounded)
and tailed with blocking ``XREAD``. Stream entry ids are the resume tokens the
port promises: ``XREAD`` returns entries strictly *after* the given id, which is
exactly the ``Last-Event-ID`` reconnect semantic.

Events travel as one ``data`` field holding the
:data:`~schemas.EventAdapter` wire JSON, so a consumer always gets back a
fully-validated :class:`~schemas.events.Event`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from redis.asyncio import Redis

from protocols.event_bus import StreamedEvent
from schemas import Event, EventAdapter

# How long one XREAD blocks before looping. The loop exists only so a closed
# consumer (SSE disconnect) is noticed promptly instead of parking forever.
_BLOCK_MS = 5_000


class RedisEventBus:
    """Append to and live-tail per-session Redis Streams."""

    def __init__(self, client: Redis, *, maxlen: int = 1000) -> None:
        self._r = client
        self._maxlen = maxlen

    @staticmethod
    def _stream(session_key: str) -> str:
        return f"events:{session_key}"

    async def publish(self, session_key: str, event: Event) -> str:
        eid = await self._r.xadd(
            self._stream(session_key),
            {"data": EventAdapter.dump_json(event)},
            maxlen=self._maxlen,
            approximate=True,
        )
        return eid.decode() if isinstance(eid, bytes) else str(eid)

    async def tail(
        self, session_key: str, *, last_id: str | None = None
    ) -> AsyncIterator[StreamedEvent]:
        # "0" asks XREAD for everything after the stream's zero id — the full
        # backlog — then each batch advances the cursor to its last entry.
        cursor = last_id or "0"
        stream = self._stream(session_key)
        while True:
            batches = await self._r.xread({stream: cursor}, block=_BLOCK_MS)
            if not batches:
                continue  # block timed out with nothing new — re-arm
            for _key, entries in batches:
                for raw_id, fields in entries:
                    cursor = raw_id.decode() if isinstance(raw_id, bytes) else str(raw_id)
                    yield StreamedEvent(cursor, EventAdapter.validate_json(fields[b"data"]))
