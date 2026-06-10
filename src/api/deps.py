"""DI — the only place the graph meets concrete connectors.

`build_container` picks the wiring from `Settings`:

- ``redis_url`` set → Redis store, bus **and** graph checkpointer; unset →
  in-memory versions of all three.
- ``object_storage`` → S3 or a local directory.
- ``fake_connectors`` → the model-shaped ports (generator, face engine, slot
  filler) become deterministic in-process fakes; otherwise OpenRouter and
  InsightFace. The two axes compose: fake models over real Redis is the
  FSM-persistence test wiring.

The choice happens once, at process start; there is no runtime failover
between backends. Routes pull the container off `app.state`; nothing below
the API layer ever imports a concrete connector. Tests substitute at the port
level by assembling `GraphServices` themselves.

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
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from redis.asyncio import Redis

from config import Settings
from graph.builder import build_graph
from graph.nodes import GraphServices
from graph.state import GraphState, initial_state
from protocols import (
    Embedder,
    EventBus,
    FaceAnalyzer,
    ImageGenerator,
    ObjectStorage,
    SlotFiller,
    StateStore,
)
from schemas import FailedEvent, NeedInputEvent
from services.budget import BudgetService
from services.connectors import (
    FakeFaceEngine,
    FakeImageGenerator,
    InMemoryEventBus,
    InMemoryStateStore,
    InsightFaceEmbedder,
    LocalObjectStorage,
    OpenRouterClient,
    OpenRouterImageGenerator,
    OpenRouterSlotFiller,
    RedisEventBus,
    RedisStateStore,
    S3ObjectStorage,
)
from services.facecheck import FaceCheckService
from services.generation_loop import GenerationLoop
from services.idempotency import IdempotencyService
from services.preset_matcher import load_library
from services.quality_gate import GateThresholds, QualityGate
from services.slot_filler import DefaultSlotFiller
from services.vision import VisionService

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

    def start(
        self, session_key: str, *, face_key: str, use_case: str, gender: str, budget_limit: int
    ) -> None:
        self._spawn(
            session_key,
            initial_state(
                session_key=session_key,
                face_key=face_key,
                use_case=use_case,
                gender=gender,
                budget_limit=budget_limit,
            ),
        )

    def resume(self, session_key: str, value: Any) -> None:
        """Answer the pending interrupt: a string for the ask slot, a decision
        dict (``approved`` / ``composition_id``) for the approval gate."""
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
    graph: CompiledStateGraph[GraphState]
    checkpointer: BaseCheckpointSaver[str]
    redis: Redis | None  # owned client behind store/bus/checkpointer; None on in-memory
    openrouter: OpenRouterClient | None  # owned transport behind generator+filler
    runner: SessionRunner = field(init=False)

    async def astart(self) -> None:
        """Async one-time init the sync factory cannot do (checkpointer indices)."""
        if isinstance(self.checkpointer, AsyncRedisSaver):
            await self.checkpointer.asetup()

    async def aclose(self) -> None:
        await self.runner.aclose()
        if self.openrouter is not None:
            await self.openrouter.aclose()
        if isinstance(self.storage, S3ObjectStorage):
            await self.storage.aclose()
        if self.redis is not None:
            await self.redis.aclose()


def _gate_thresholds(s: Settings) -> GateThresholds:
    return GateThresholds(
        min_side=s.gate_min_side,
        max_secondary_face_ratio=s.gate_max_secondary_face_ratio,
        min_face_side=s.gate_min_face_side,
        floor_face_side=s.gate_floor_face_side,
        min_blur_var=s.gate_min_blur_var,
        floor_blur_var=s.gate_floor_blur_var,
        min_brightness=s.gate_min_brightness,
        max_brightness=s.gate_max_brightness,
        floor_min_brightness=s.gate_floor_min_brightness,
        floor_max_brightness=s.gate_floor_max_brightness,
        risk_max_abs_yaw=s.gate_risk_max_abs_yaw,
    )


def build_container(settings: Settings) -> Container:
    redis: Redis | None = None
    checkpointer: BaseCheckpointSaver[str]
    if settings.redis_url:
        redis = Redis.from_url(settings.redis_url)
        store: StateStore = RedisStateStore(redis)
        bus: EventBus = RedisEventBus(redis, maxlen=settings.event_stream_maxlen)
        # Same client, separate key space; indices are created in astart().
        checkpointer = AsyncRedisSaver(redis_client=redis)
    else:
        store = InMemoryStateStore()
        bus = InMemoryEventBus()
        checkpointer = InMemorySaver()

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

    openrouter: OpenRouterClient | None = None
    generator: ImageGenerator
    slot_filler: SlotFiller
    analyzer: FaceAnalyzer
    embedder: Embedder
    if settings.fake_connectors:
        generator = FakeImageGenerator()
        slot_filler = DefaultSlotFiller()
        face_engine = FakeFaceEngine()
        analyzer, embedder = face_engine, face_engine
    else:
        openrouter = OpenRouterClient(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            timeout_seconds=settings.openrouter_timeout_seconds,
            attempts=settings.openrouter_retry_attempts,
        )
        generator = OpenRouterImageGenerator(openrouter, model=settings.generation_model)
        slot_filler = OpenRouterSlotFiller(openrouter, model=settings.slot_filler_model)
        # One InsightFace pack serves both ports from a single inference pass.
        insight = InsightFaceEmbedder(
            model_pack=settings.insightface_model,
            root=settings.insightface_root,
            det_size=settings.insightface_det_size,
        )
        analyzer, embedder = insight, insight

    services = GraphServices(
        store=store,
        storage=storage,
        bus=bus,
        vision=VisionService(analyzer, QualityGate(_gate_thresholds(settings))),
        library=load_library(settings),
        slot_filler=slot_filler,
        generation_loop=GenerationLoop(
            store=store,
            storage=storage,
            bus=bus,
            generator=generator,
            facecheck=FaceCheckService(embedder),
            budget=BudgetService(store),
            idempotency=IdempotencyService(store),
            session_ttl_seconds=settings.session_ttl_seconds,
            face_ttl_seconds=settings.face_ttl_seconds,
            max_iterations=settings.max_iterations,
        ),
        unit_price=settings.generation_unit_price,
        default_expected_generations=settings.default_expected_generations,
        session_ttl_seconds=settings.session_ttl_seconds,
        face_ttl_seconds=settings.face_ttl_seconds,
    )
    graph = build_graph(services, checkpointer=checkpointer)

    container = Container(
        settings=settings,
        store=store,
        bus=bus,
        storage=storage,
        graph=graph,
        checkpointer=checkpointer,
        redis=redis,
        openrouter=openrouter,
    )
    container.runner = SessionRunner(graph, bus)
    return container
