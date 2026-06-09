"""DI — the only place the graph meets concrete connectors.

`build_container` picks the wiring by `Settings.connectors` and assembles the
compiled graph plus the `SessionRunner` that drives it. Routes pull the
container off `app.state`; nothing below the API layer ever imports a concrete
connector.

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

from config import Settings
from graph.builder import build_graph
from graph.state import GraphState
from protocols.event_bus import EventBus
from schemas import FailedEvent, NeedInputEvent

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
    bus: EventBus
    runner: SessionRunner = field(init=False)


def build_container(settings: Settings) -> Container:
    if settings.connectors == "fake":
        # Dev/test wiring reuses the contract-tested fakes; the app runs from
        # the repo root, so `tests` is importable outside the installed tree.
        from tests.fakes.store import InMemoryEventBus

        bus: EventBus = InMemoryEventBus()
    else:
        raise NotImplementedError(
            "connectors='real' needs the Redis connectors — not part of the skeleton"
        )

    # In-memory checkpointer for the skeleton; the Redis saver replaces it
    # behind the same BaseCheckpointSaver seam.
    graph = build_graph(bus, checkpointer=InMemorySaver())

    container = Container(settings=settings, bus=bus)
    container.runner = SessionRunner(graph, bus)
    return container
