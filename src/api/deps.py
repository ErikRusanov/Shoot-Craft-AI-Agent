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
import json
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any
from uuid import uuid4

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
    BriefParser,
    Embedder,
    EventBus,
    FaceAnalyzer,
    ImageGenerator,
    ObjectStorage,
    PromptWriter,
    SlotFiller,
    StateStore,
    StepPlanner,
)
from schemas import FailedEvent, FailureCode, FsmState, NeedInputEvent, SessionState
from services.brief_parser import DeterministicBriefParser
from services.budget import BudgetService
from services.connectors import (
    FakeFaceEngine,
    FakeImageGenerator,
    InMemoryEventBus,
    InMemoryStateStore,
    InsightFaceEmbedder,
    LocalObjectStorage,
    OpenRouterBriefParser,
    OpenRouterClient,
    OpenRouterImageGenerator,
    OpenRouterPromptWriter,
    OpenRouterSlotFiller,
    OpenRouterStepPlanner,
    RedisEventBus,
    RedisStateStore,
    S3ObjectStorage,
    ThrottledImageGenerator,
)
from services.facecheck import FaceCheckService
from services.generation_loop import GenerationLoop
from services.idempotency import IdempotencyService
from services.planner import DeterministicStepPlanner
from services.preset_matcher import PresetLibrary, load_library
from services.pricing import PricingTable
from services.prompt_writer import DeterministicPromptWriter
from services.quality_gate import GateThresholds, QualityGate
from services.slot_filler import DefaultSlotFiller
from services.telemetry import Telemetry
from services.vision import VisionService
from utils.money import to_micro

log = structlog.get_logger(__name__)

# Slack on the run lock past the wall-clock: the lock must outlive a run that
# dies *at* the deadline (its finally still has to release), but a crashed
# process must not wedge the session for long.
_RUN_LOCK_GRACE_SECONDS = 60

_WALL_CLOCK_REASON = "session run exceeded the wall-clock limit"
_CANCELLED_REASON = "cancelled by the caller"


def _accounting(session: SessionState | None) -> dict[str, Any]:
    """Spend accounting for a terminal event, from the session record (or zeros
    when there is no session to read)."""
    if session is None:
        return {"iterations_used": 0, "generations_spent": 0, "cost_spent": Decimal("0")}
    return {
        "iterations_used": len(session.iterations),
        "generations_spent": session.generations_spent(),
        "cost_spent": session.cost_spent(),
    }


def build_pricing(settings: Settings) -> PricingTable:
    """The pricing table for this process: built-in rates plus startup overrides.

    Fails fast (here, not mid-session) if any configured model has no rate after
    overrides are applied. A per-stage model outside the built-in table (the
    defaults plus :data:`~services.pricing.KNOWN_AUX_RATES`) aliases the lite
    rate — aux calls settle on the provider-reported cost, so the rate is
    forecasting only and a cheap-ish guess beats a startup crash on rename.
    """
    pricing = PricingTable.default(
        generation_model=settings.generation_model, lite_model=settings.slot_filler_model
    )
    if settings.pricing_overrides_json:
        overrides = json.loads(settings.pricing_overrides_json)
        pricing = PricingTable.model_validate({**pricing.model_dump(mode="json"), **overrides})
    stage_models = (
        settings.brief_parser_model,
        settings.planner_model,
        settings.inventory_model,
        settings.prompt_writer_model,
    )
    for model in stage_models:
        if model not in pricing.model_rates:
            pricing.model_rates[model] = pricing.rate_for(settings.slot_filler_model)
    pricing.rate_for(settings.generation_model)
    pricing.rate_for(settings.slot_filler_model)
    return pricing


class SessionRunner:
    """Drives graph runs in background tasks, one per session.

    `start` and `resume` funnel into the same `_run`: invoke the graph on
    `thread_id == session_key` and, if it paused on an interrupt, publish the
    `need_input` event from the surfaced payload. Publishing happens here —
    never inside the interrupted node — because LangGraph re-executes the node
    body on resume and an in-node publish would duplicate.

    Hardening lives here too, because the runner is the only writer:

    - **run lock** (``lock:run:{session_key}`` in the store) — one in-flight
      graph run per session, across replicas, so a duplicated mutation can't
      double-drive the FSM. Spawning fails (returns ``False``) when held.
    - **wall clock** — a run is cancelled at ``session_wall_clock_seconds``
      and the session fails cleanly; the lock TTL rides slightly above it,
      so even a SIGKILLed process frees the session.
    - **cancel** — the caller-initiated terminal: cancels the in-flight task
      (if any), marks the session ``cancelled`` and closes the stream.
    - **telemetry** — one de-identified outcome event per terminal run.
    """

    def __init__(
        self,
        graph: CompiledStateGraph[GraphState],
        bus: EventBus,
        store: StateStore,
        telemetry: Telemetry,
        *,
        wall_clock_seconds: int,
        session_ttl_seconds: int,
    ) -> None:
        self._graph = graph
        self._bus = bus
        self._store = store
        self._telemetry = telemetry
        self._wall_clock = wall_clock_seconds
        self._session_ttl = session_ttl_seconds
        self._tasks: dict[str, asyncio.Task[None]] = {}

    async def start(
        self,
        session_key: str,
        *,
        face_key: str,
        use_case: str,
        brief: str,
        budget_limit: Decimal,
    ) -> bool:
        """Begin a fresh run; ``False`` when a run is already in flight.

        ``budget_limit`` is USD; the checkpoint carries it as micro-USD (Decimal
        does not survive the checkpointer's serde).
        """
        return await self._spawn(
            session_key,
            initial_state(
                session_key=session_key,
                face_key=face_key,
                use_case=use_case,
                brief=brief,
                budget_limit=to_micro(budget_limit),
            ),
        )

    async def resume(self, session_key: str, value: Any) -> bool:
        """Answer the pending interrupt: a string for the ask slot, a decision
        dict (``approved`` / ``composition_id``) for the approval gate.
        ``False`` when a run is already in flight."""
        return await self._spawn(session_key, Command(resume=value))

    async def cancel(self, session_key: str) -> None:
        """Stop the session: kill the in-flight run (if any), mark the session
        ``cancelled``, close the stream with a terminal event."""
        task = self._tasks.get(session_key)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        session = await self._store.get_session(session_key)
        if session is not None:
            session.fsm_state = FsmState.CANCELLED
            await self._store.put_session(session, ttl_seconds=self._session_ttl)
            self._telemetry.session_terminal(
                session, failure_reason=_CANCELLED_REASON, failure_code=FailureCode.CANCELLED.value
            )
        await self._bus.publish(
            session_key,
            FailedEvent(
                code=FailureCode.CANCELLED,
                reason=_CANCELLED_REASON,
                **_accounting(session),
            ),
        )

    @staticmethod
    def _lock_key(session_key: str) -> str:
        return f"run:{session_key}"

    async def _spawn(self, session_key: str, payload: GraphState | Command[Any]) -> bool:
        token = uuid4().hex
        acquired = await self._store.acquire_lock(
            self._lock_key(session_key),
            token=token,
            ttl_seconds=self._wall_clock + _RUN_LOCK_GRACE_SECONDS,
        )
        if not acquired:
            log.warning("session_run_locked", session_key=session_key)
            return False
        # Keep a strong reference: asyncio only weakly holds running tasks.
        task = asyncio.create_task(self._run(session_key, payload, token))
        self._tasks[session_key] = task

        def _reap(done: asyncio.Task[None]) -> None:
            # A resume may have replaced the entry — only reap our own task.
            if self._tasks.get(session_key) is done:
                del self._tasks[session_key]

        task.add_done_callback(_reap)
        return True

    async def _run(self, session_key: str, payload: GraphState | Command[Any], token: str) -> None:
        config: RunnableConfig = {"configurable": {"thread_id": session_key}}
        try:
            async with asyncio.timeout(self._wall_clock):
                result: dict[str, Any] = await self._graph.ainvoke(payload, config=config)
        except TimeoutError:
            await self._release(session_key, token)
            log.error("graph_run_wall_clock", session_key=session_key, limit=self._wall_clock)
            await self._fail_session(session_key, _WALL_CLOCK_REASON)
            return
        except asyncio.CancelledError:
            # cancel() owns the narration; here only the lock must not leak.
            await asyncio.shield(self._release(session_key, token))
            raise
        except Exception:
            await self._release(session_key, token)
            log.exception("graph_run_failed", session_key=session_key)
            await self._bus.publish(
                session_key, FailedEvent(code=FailureCode.INTERNAL, reason="internal error")
            )
            return
        # Release before narrating: a client reacting to `need_input` must find
        # the lock already free, or its immediate answer would bounce.
        await self._release(session_key, token)
        interrupts = result.get("__interrupt__") or []
        if interrupts:
            await self._bus.publish(session_key, NeedInputEvent(**interrupts[0].value))
            return
        failure = result.get("failure") or {}
        await self._emit_telemetry(session_key, failure_reason=failure.get("reason"))

    async def _release(self, session_key: str, token: str) -> None:
        try:
            await self._store.release_lock(self._lock_key(session_key), token=token)
        except Exception:  # a dying store must not mask the run's own outcome
            log.warning("run_lock_release_failed", session_key=session_key)

    async def _fail_session(
        self, session_key: str, reason: str, code: FailureCode = FailureCode.WALL_CLOCK
    ) -> None:
        session = await self._store.get_session(session_key)
        if session is not None:
            session.fsm_state = FsmState.FAILED
            await self._store.put_session(session, ttl_seconds=self._session_ttl)
            self._telemetry.session_terminal(
                session, failure_reason=reason, failure_code=code.value
            )
        await self._bus.publish(
            session_key,
            FailedEvent(code=code, reason=reason, **_accounting(session)),
        )

    async def _emit_telemetry(self, session_key: str, *, failure_reason: str | None) -> None:
        session = await self._store.get_session(session_key)
        if session is None:
            return
        if failure_reason is None and session.fsm_state is FsmState.FAILED:
            # The generation loop publishes its own terminal event; the graph
            # state carries no reason for it, so name the phase at least.
            failure_reason = "generation loop ended without a deliverable"
        self._telemetry.session_terminal(session, failure_reason=failure_reason)

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
    vision: VisionService
    library: PresetLibrary
    idempotency: IdempotencyService  # HTTP-layer replay; same store as step 4
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
        bus: EventBus = RedisEventBus(
            redis, maxlen=settings.event_stream_maxlen, ttl_seconds=settings.session_ttl_seconds
        )
        # Same client, separate key space; indices are created in astart().
        # Checkpoints follow the session TTL — an abandoned interrupt must not
        # outlive the session state it would resume into.
        checkpointer = AsyncRedisSaver(
            redis_client=redis,
            ttl={"default_ttl": settings.session_ttl_seconds / 60, "refresh_on_read": True},
        )
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
    brief_parser: BriefParser
    planner: StepPlanner
    writer: PromptWriter
    analyzer: FaceAnalyzer
    embedder: Embedder
    if settings.fake_connectors:
        generator = FakeImageGenerator()
        slot_filler = DefaultSlotFiller()
        brief_parser = DeterministicBriefParser()
        planner = DeterministicStepPlanner()
        writer = DeterministicPromptWriter()
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
        brief_parser = OpenRouterBriefParser(openrouter, model=settings.brief_parser_model)
        planner = OpenRouterStepPlanner(openrouter, model=settings.planner_model)
        writer = OpenRouterPromptWriter(openrouter, model=settings.prompt_writer_model)
        # One InsightFace pack serves both ports from a single inference pass.
        insight = InsightFaceEmbedder(
            model_pack=settings.insightface_model,
            root=settings.insightface_root,
            det_size=settings.insightface_det_size,
        )
        analyzer, embedder = insight, insight
    # Backpressure applies to fakes too: load tests exercise the same wiring.
    generator = ThrottledImageGenerator(
        generator, max_concurrent=settings.max_concurrent_generations
    )

    vision = VisionService(analyzer, QualityGate(_gate_thresholds(settings)))
    library = load_library(settings)
    pricing = build_pricing(settings)
    budget = BudgetService(store, pricing)
    services = GraphServices(
        store=store,
        storage=storage,
        bus=bus,
        vision=vision,
        library=library,
        slot_filler=slot_filler,
        brief_parser=brief_parser,
        planner=planner,
        generation_loop=GenerationLoop(
            store=store,
            storage=storage,
            bus=bus,
            generator=generator,
            writer=writer,
            facecheck=FaceCheckService(embedder),
            budget=budget,
            pricing=pricing,
            idempotency=IdempotencyService(store),
            session_ttl_seconds=settings.session_ttl_seconds,
            face_ttl_seconds=settings.face_ttl_seconds,
            max_iterations=settings.max_iterations,
            generation_model=settings.generation_model,
        ),
        budget=budget,
        pricing=pricing,
        generation_model=settings.generation_model,
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
        vision=vision,
        library=library,
        idempotency=IdempotencyService(store),
        graph=graph,
        checkpointer=checkpointer,
        redis=redis,
        openrouter=openrouter,
    )
    container.runner = SessionRunner(
        graph,
        bus,
        store,
        Telemetry(),
        wall_clock_seconds=settings.session_wall_clock_seconds,
        session_ttl_seconds=settings.session_ttl_seconds,
    )
    return container
