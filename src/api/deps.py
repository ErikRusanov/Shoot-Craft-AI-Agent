"""DI — the only place the graph meets concrete connectors.

`build_container` picks the wiring from `Settings` — `redis_url` set → Redis
store/bus, unset → in-memory; `object_storage` picks S3 or a local directory —
and assembles the compiled graph plus the `SessionRunner` that drives it. The
choice happens once, at process start; there is no runtime failover between
backends. Routes pull the container off `app.state`; nothing below the API
layer ever imports a concrete connector.

The runner lives here (not in graph/) because it is process-level glue: it owns
background tasks and translates the graph's interrupt surface into bus events —
both squarely an application concern.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import structlog
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from redis.asyncio import Redis

from config import Settings
from graph.builder import build_graph
from graph.state import GraphState
from protocols import EventBus, ObjectStorage, StateStore
from schemas import FailedEvent, NeedInputEvent
from services.connectors import (
    InMemoryEventBus,
    InMemoryStateStore,
    LocalObjectStorage,
    RedisEventBus,
    RedisStateStore,
    S3ObjectStorage,
)

log = structlog.get_logger(__name__)


class SessionRunner:
    """Drives graph runs in background tasks, one per session.

    Both `start` and `resume` funnel into the same `_run`: invoke the graph on
    `thread_id == session_key` and, if it paused on an interrupt, publish the
    `need_input` event from the surfaced payload. Publishing happens here —
    never inside the interrupted node — because LangGraph re-executes the node
    body on resume and an in-node publish would duplicate.
    """

    def __init__(self, graph: CompiledStateGraph[GraphState], bus: EventBus) -> None:
        self._graph = graph
        self._bus = bus
        self._tasks: dict[str, asyncio.Task[None]] = {}

    def start(self, session_key: str, *, face_key: str) -> None:
        initial: GraphState = {"session_key": session_key, "face_key": face_key, "slots": {}}
        self._spawn(session_key, initial)

    def resume(self, session_key: str, value: str) -> None:
        self._spawn(session_key, Command(resume=value))

    def _spawn(self, session_key: str, payload: GraphState | Command[Any]) -> None:
        # Keep a strong reference: asyncio only weakly holds running tasks.
        task = asyncio.create_task(self._run(session_key, payload))
        self._tasks[session_key] = task

        def _reap(done: asyncio.Task[None]) -> None:
            # A resume may have replaced the entry — only reap our own task.
            if self._tasks.get(session_key) is done:
                del self._tasks[session_key]

        task.add_done_callback(_reap)

    async def _run(self, session_key: str, payload: GraphState | Command[Any]) -> None:
        config: RunnableConfig = {"configurable": {"thread_id": session_key}}
        try:
            result: dict[str, Any] = await self._graph.ainvoke(payload, config=config)
        except Exception:
            log.exception("graph_run_failed", session_key=session_key)
            await self._bus.publish(session_key, FailedEvent(reason="internal error"))
            return
        interrupts = result.get("__interrupt__") or []
        if interrupts:
            await self._bus.publish(session_key, NeedInputEvent(**interrupts[0].value))

    async def aclose(self) -> None:
        for task in list(self._tasks.values()):
            task.cancel()
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        self._tasks.clear()


@dataclass(slots=True)
class Container:
    """Everything the API layer needs, assembled once per app."""

    settings: Settings
    store: StateStore
    bus: EventBus
    storage: ObjectStorage
    redis: Redis | None  # owned client behind store+bus; None on in-memory wiring
    runner: SessionRunner = field(init=False)

    async def aclose(self) -> None:
        await self.runner.aclose()
        if isinstance(self.storage, S3ObjectStorage):
            await self.storage.aclose()
        if self.redis is not None:
            await self.redis.aclose()


def build_container(settings: Settings) -> Container:
    redis: Redis | None = None
    if settings.redis_url:
        redis = Redis.from_url(settings.redis_url)
        store: StateStore = RedisStateStore(redis)
        bus: EventBus = RedisEventBus(redis, maxlen=settings.event_stream_maxlen)
    else:
        store = InMemoryStateStore()
        bus = InMemoryEventBus()

    storage: ObjectStorage
    if settings.object_storage == "s3":
        assert settings.s3_bucket is not None  # enforced by the Settings validator
        storage = S3ObjectStorage(
            bucket=settings.s3_bucket,
            endpoint_url=settings.s3_endpoint_url,
            region=settings.s3_region,
            access_key=settings.s3_access_key,
            secret_key=settings.s3_secret_key,
        )
    else:
        storage = LocalObjectStorage(settings.local_storage_path)

    # In-memory checkpointer for now; the Redis saver replaces it behind the
    # same BaseCheckpointSaver seam when the graph grows real persistence.
    graph = build_graph(bus, checkpointer=InMemorySaver())

    container = Container(settings=settings, store=store, bus=bus, storage=storage, redis=redis)
    container.runner = SessionRunner(graph, bus)
    return container
