"""Generation loop: the money- and identity-critical invariants, on fakes.

The per-attempt similarity is scripted through ``ScriptedSimilarityEmbedder``
(the loop runs the *real* ``FaceCheckService``), the generator is the recording
fake, and state/budget/idempotency sit on the in-memory store. What is pinned:

- keep-best — a worse or below-floor later attempt never displaces the best;
- below ``identity_floor`` nothing is ever delivered;
- budget exhaustion ends in a clean ``failed`` (or delivers the existing best);
- retries stop at K (and at the config runaway ceiling);
- a transport failure refunds its reservation (costs nothing);
- a delivered frame settles to the provider's billed cost;
- re-entry with the same ``idem_key`` replays the recorded outcome;
- the paid frame is checkpointed (provider id + result ref) before face-check.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

import numpy as np
import pytest
from numpy.typing import NDArray

from protocols import Embedder
from schemas import (
    AppliesTo,
    BriefAnalysis,
    EditStep,
    Event,
    FaceProfile,
    FailedEvent,
    FailureCode,
    FrameMetrics,
    FsmState,
    GateReason,
    Generation,
    Plan,
    Preset,
    ResultEvent,
    RiskLevel,
    SessionState,
    Slot,
    Thresholds,
    Verdict,
)
from services.budget import BudgetService
from services.facecheck import FaceCheckService
from services.generation_loop import GenerationLoop
from services.idempotency import IdempotencyService
from services.pricing import PricingTable
from services.prompt_writer import IDENTITY_EMPHASIS, DeterministicPromptWriter
from tests.fakes import (
    FixedImageGenerator,
    FlakyImageGenerator,
    InMemoryEventBus,
    InMemoryObjectStorage,
    InMemoryStateStore,
    ScriptedSimilarityEmbedder,
    axis_embedding,
)

TTL = 3600
GEN_MODEL = "gen-model"
PRICING = PricingTable.default(generation_model=GEN_MODEL, lite_model="lite-model")
# A known billed cost so the budget settles to a predictable spend.
GEN_COST = Decimal("0.05")
LIBRARY_VERSION = "0.4.0"
SESSION_KEY = "sess-1"
FACE_KEY = "face-1"
PHOTO_REF = "photos/face-1"
REFERENCE_PHOTO = b"reference-photo-bytes"

IDENTITY_INSTRUCTION = "Keep the exact same face as in the reference photo."
PROMPT_STRUCTURE = "Avatar of the person in {style} style."


class RecordingBus(InMemoryEventBus):
    """The in-memory bus plus a flat publish log for order/content assertions."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[Event] = []

    async def publish(self, session_key: str, event: Event) -> str:
        self.events.append(event)
        return await super().publish(session_key, event)

    @property
    def types(self) -> list[str]:
        return [event.type for event in self.events]


def make_preset(*, k: int = 3, temperature: float = 0.4) -> Preset:
    return Preset(
        id="test_avatar",
        version="1.0.0",
        applies_to=AppliesTo(use_case=["avatar"]),
        identity_instruction=IDENTITY_INSTRUCTION,
        prompt_structure=PROMPT_STRUCTURE,
        negative_prompt="cartoon, different person",
        slots={"style": Slot(required=True, default="studio", enum=["studio", "outdoor"])},
        generation=Generation(
            temperature=temperature, aspect_ratio="1:1", face_media_resolution="high"
        ),
        thresholds=Thresholds(similarity_threshold=0.8, identity_floor=0.5, K_max_retries=k),
    )


def make_face() -> FaceProfile:
    return FaceProfile(
        face_key=FACE_KEY,
        embedding=axis_embedding(),
        gate_verdict=Verdict.PASSED,
        gate_reason=GateReason.OK,
        metrics=FrameMetrics(
            face_count=1,
            face_area_ratio=0.25,
            face_side=320.0,
            blur_var=500.0,
            yaw=0.0,
            pitch=0.0,
            roll=0.0,
            brightness=120.0,
            width=1024,
            height=1024,
        ),
        photo_ref=PHOTO_REF,
    )


@dataclass
class Harness:
    loop: GenerationLoop
    store: InMemoryStateStore
    storage: InMemoryObjectStorage
    bus: RecordingBus
    generator: FixedImageGenerator
    preset: Preset

    async def run(
        self, *, face_crop: bytes | None = None, idem_key: str = "op-1"
    ) -> ResultEvent | FailedEvent:
        return await self.loop.run(
            session_key=SESSION_KEY, preset=self.preset, idem_key=idem_key, face_crop=face_crop
        )

    async def session(self) -> SessionState:
        session = await self.store.get_session(SESSION_KEY)
        assert session is not None
        return session


async def make_harness(
    *,
    similarities: list[float | None] | None = None,
    embedder: Embedder | None = None,
    generator: FixedImageGenerator | None = None,
    budget_limit: Decimal = Decimal("10"),
    gen_cost: Decimal | None = GEN_COST,
    k: int = 3,
    max_iterations: int = 8,
    temperature: float = 0.4,
) -> Harness:
    store = InMemoryStateStore()
    storage = InMemoryObjectStorage()
    bus = RecordingBus()
    gen = generator if generator is not None else FixedImageGenerator(cost=gen_cost)
    preset = make_preset(k=k, temperature=temperature)

    await storage.put(PHOTO_REF, REFERENCE_PHOTO)
    await store.put_face(make_face(), ttl_seconds=TTL)
    await store.put_session(
        SessionState(
            session_key=SESSION_KEY,
            face_key=FACE_KEY,
            slots={"style": "studio"},
            thresholds=preset.thresholds,
            budget_limit=budget_limit,
            preset_id=preset.id,
            preset_version=preset.version,
            library_version=LIBRARY_VERSION,
        ),
        ttl_seconds=TTL,
    )

    emb = embedder if embedder is not None else ScriptedSimilarityEmbedder(similarities or [])
    loop = GenerationLoop(
        store=store,
        storage=storage,
        bus=bus,
        generator=gen,
        writer=DeterministicPromptWriter(),
        facecheck=FaceCheckService(emb),
        budget=BudgetService(store, PRICING),
        pricing=PRICING,
        idempotency=IdempotencyService(store),
        session_ttl_seconds=TTL,
        face_ttl_seconds=TTL,
        max_iterations=max_iterations,
        generation_model=GEN_MODEL,
    )
    return Harness(loop=loop, store=store, storage=storage, bus=bus, generator=gen, preset=preset)


# --- happy path ---


async def test_first_attempt_passes_and_delivers() -> None:
    h = await make_harness(similarities=[0.9])
    result = await h.run()

    assert isinstance(result, ResultEvent)
    assert result.best.iteration_n == 1
    assert result.best.similarity == pytest.approx(0.9, abs=1e-6)
    assert result.best.verdict is Verdict.PASSED
    assert result.best.risk_level is RiskLevel.LOW
    assert await h.storage.get(result.best.result_ref) == h.generator.image

    # The terminal is self-sufficient: spend and the preset triple travel with it.
    assert result.iterations_used == 1
    assert result.generations_spent == 1
    assert result.cost_spent == GEN_COST  # settled to the provider's billed cost
    assert result.preset_id == h.preset.id
    assert result.preset_version == h.preset.version
    assert result.library_version == LIBRARY_VERSION

    assert len(h.generator.calls) == 1
    assert h.generator.calls[0].reference_count == 1

    session = await h.session()
    assert session.fsm_state is FsmState.DONE
    assert [it.charged for it in session.iterations] == [True]
    assert session.iterations[0].cost == GEN_COST
    assert session.iterations[0].prompt_text is not None  # full prompt kept for debugging
    assert session.cost_spent() == GEN_COST
    assert h.bus.types == ["iteration_start", "iteration_result", "result"]


async def test_soft_then_pass_keeps_the_better_frame() -> None:
    h = await make_harness(similarities=[0.6, 0.9])
    result = await h.run()

    assert isinstance(result, ResultEvent)
    assert result.best.iteration_n == 2
    assert result.best.verdict is Verdict.PASSED
    assert len(h.generator.calls) == 2  # stopped on pass, not on K
    assert result.cost_spent == GEN_COST * 2
    assert h.bus.types == [
        "iteration_start",
        "iteration_result",
        "retry",
        "iteration_start",
        "iteration_result",
        "result",
    ]

    face = await h.store.get_face(FACE_KEY)
    assert face is not None
    assert face.convergence.attempts == 2
    assert face.convergence.best_similarity == pytest.approx(0.9, abs=1e-6)
    assert face.convergence.improved_last is True


# --- keep-best ---


async def test_failed_risk_attempt_does_not_clobber_passed_frame() -> None:
    # Attempt 1 lands soft (deliverable at risk), attempt 2 collapses below the
    # floor. The kept best must remain attempt 1, untouched.
    h = await make_harness(similarities=[0.7, 0.3], k=1)
    result = await h.run()

    assert isinstance(result, ResultEvent)
    assert result.best.iteration_n == 1
    assert result.best.similarity == pytest.approx(0.7, abs=1e-6)
    assert result.best.verdict is Verdict.SOFT
    assert result.best.risk_level is RiskLevel.MEDIUM

    session = await h.session()
    assert session.iterations[1].verdict is Verdict.BELOW_FLOOR
    assert session.best_result is not None
    assert session.best_result.result_ref == session.iterations[0].result_ref

    face = await h.store.get_face(FACE_KEY)
    assert face is not None
    assert face.convergence.improved_last is False
    assert face.convergence.best_similarity == pytest.approx(0.7, abs=1e-6)


async def test_equal_or_worse_soft_attempt_keeps_the_earlier_frame() -> None:
    h = await make_harness(similarities=[0.7, 0.7, 0.6], k=2)
    result = await h.run()

    assert isinstance(result, ResultEvent)
    assert result.best.iteration_n == 1  # strictly-better wins; ties keep the first


# --- identity floor ---


async def test_below_floor_is_never_delivered() -> None:
    h = await make_harness(similarities=[0.4, 0.3, 0.2], k=2)
    result = await h.run()

    assert isinstance(result, FailedEvent)
    assert "identity floor" in result.reason
    assert "result" not in h.bus.types

    session = await h.session()
    assert session.best_result is None
    assert session.fsm_state is FsmState.FAILED
    # Paid attempts stay paid even when undeliverable — no refunds for delivered frames.
    assert session.generations_spent() == 3
    assert result.cost_spent == GEN_COST * 3


async def test_frame_that_lost_the_face_is_below_floor() -> None:
    # The embedder's "no face" signal (a scripted None) must read as a failed
    # attempt and trigger a retry, not crash the loop.
    h = await make_harness(similarities=[None, 0.9], k=1)
    result = await h.run()

    assert isinstance(result, ResultEvent)
    assert result.best.iteration_n == 2
    session = await h.session()
    assert session.iterations[0].similarity == 0.0
    assert session.iterations[0].verdict is Verdict.BELOW_FLOOR


# --- budget ---


async def test_zero_budget_fails_cleanly_without_generating() -> None:
    h = await make_harness(similarities=[], budget_limit=Decimal("0"))
    result = await h.run()

    assert isinstance(result, FailedEvent)
    assert "budget" in result.reason
    assert h.generator.calls == []
    assert h.bus.types == ["failed"]  # no phantom iteration_start
    assert (await h.session()).fsm_state is FsmState.FAILED


async def test_budget_exhaustion_mid_loop_delivers_the_kept_best() -> None:
    # ~$0.078 reserved per generation: a $0.15 budget admits two, then the third
    # reservation is refused — the loop ships the best soft frame, not an error.
    h = await make_harness(similarities=[0.6, 0.7], budget_limit=Decimal("0.15"), k=5)
    result = await h.run()

    assert isinstance(result, ResultEvent)
    assert result.best.iteration_n == 2
    assert result.best.similarity == pytest.approx(0.7, abs=1e-6)
    assert result.best.verdict is Verdict.SOFT
    assert len(h.generator.calls) == 2
    assert result.cost_spent == GEN_COST * 2


async def test_network_failure_does_not_eat_budget() -> None:
    # A transport failure refunds its reservation, so the retry can reserve again
    # — even on a budget barely above one generation's padded estimate.
    h = await make_harness(
        similarities=[0.9],
        generator=FlakyImageGenerator(failures=1),
        budget_limit=Decimal("0.10"),
    )
    result = await h.run()

    assert isinstance(result, ResultEvent)
    assert len(h.generator.calls) == 2  # one failed, one delivered

    session = await h.session()
    assert [it.charged for it in session.iterations] == [False, True]
    assert session.iterations[0].provider_request_id is None
    assert session.iterations[0].result_ref is None
    assert session.iterations[0].cost == Decimal("0")  # refunded, costs nothing
    assert session.iterations[0].error is not None  # the dead attempt is explained
    assert h.bus.types == [
        "iteration_start",
        "iteration_result",
        "retry",
        "iteration_start",
        "iteration_result",
        "result",
    ]


async def test_persistent_generator_failure_fails_cleanly() -> None:
    h = await make_harness(
        similarities=[], generator=FlakyImageGenerator(failures=99), budget_limit=Decimal("10"), k=1
    )
    result = await h.run()

    assert isinstance(result, FailedEvent)
    assert "generation failed" in result.reason
    session = await h.session()
    assert [it.charged for it in session.iterations] == [False, False]
    assert (await h.session()).fsm_state is FsmState.FAILED
    # Every dead attempt refunded its reservation — nothing was billed.
    assert result.cost_spent == Decimal("0")


# --- retry policy ---


async def test_k_max_retries_is_respected() -> None:
    h = await make_harness(similarities=[0.6] * 4, k=3, budget_limit=Decimal("10"))
    result = await h.run()

    assert isinstance(result, ResultEvent)  # soft keep-best ships when K runs out
    assert len(h.generator.calls) == 4  # the first attempt + exactly K retries
    assert h.bus.types.count("retry") == 3


async def test_runaway_ceiling_caps_attempts_below_k() -> None:
    h = await make_harness(similarities=[0.6, 0.6], k=99, max_iterations=2)
    await h.run()
    assert len(h.generator.calls) == 2


async def test_retry_strengthens_reference_without_rewriting_the_prompt() -> None:
    h = await make_harness(similarities=[0.6, 0.9], temperature=0.4)
    await h.run(face_crop=b"tight-face-crop")

    first, second = h.generator.calls

    # Attempt 1: the plain built prompt, preset knobs untouched. The face crop is
    # now attached from the first attempt (identity boost), not only on retry.
    assert IDENTITY_EMPHASIS not in first.prompt
    assert first.params.temperature == pytest.approx(0.4)
    assert first.face_crop == b"tight-face-crop"

    # Retry: identity emphasis appended, temperature halved, crop still attached.
    assert IDENTITY_EMPHASIS in second.prompt
    assert second.params.temperature == pytest.approx(0.2)
    assert second.face_crop == b"tight-face-crop"

    # Frozen blocks stay verbatim on both attempts — the retry only *appends*.
    for call in (first, second):
        assert call.prompt.startswith(IDENTITY_INSTRUCTION)
        assert "Avatar of the person in studio style." in call.prompt
        assert call.prompt.endswith("Strictly avoid: " + h.preset.negative_prompt)
        assert call.params.aspect_ratio == h.preset.generation.aspect_ratio


# --- multi-step chain ---


async def _set_plan(
    h: Harness,
    steps: list[EditStep],
    *,
    mode: Literal["edit", "generate"] = "edit",
    preserve: list[str] | None = None,
) -> None:
    session = await h.session()
    session.brief_analysis = BriefAnalysis(mode=mode, preserve=preserve or [])
    session.plan = Plan(summary="", planned_generations=len(steps), steps=steps)
    await h.store.put_session(session, ttl_seconds=TTL)


async def test_two_step_chain_delivers_last_step_and_chains_images() -> None:
    h = await make_harness(similarities=[0.9, 0.88])
    await _set_plan(
        h,
        [
            EditStep(
                n=1, title="background", instruction="blue background", targets=["background"]
            ),
            EditStep(n=2, title="clothing", instruction="red shirt", targets=["clothing"]),
        ],
    )
    result = await h.run(face_crop=b"anchor-crop")

    assert isinstance(result, ResultEvent)
    assert result.best.iteration_n == 2  # the last completed step's best is delivered
    assert len(h.generator.calls) == 2
    # Step 1 edits the original photo; step 2 edits step 1's result (the chain).
    assert h.generator.calls[0].reference_image == REFERENCE_PHOTO
    assert h.generator.calls[1].reference_image == h.generator.image
    # The identity anchor crop is the original on every step.
    assert all(call.face_crop == b"anchor-crop" for call in h.generator.calls)

    session = await h.session()
    assert [it.step_n for it in session.iterations] == [1, 2]
    assert session.plan is not None
    assert [s.status for s in session.plan.steps] == ["completed", "completed"]
    assert all(s.result_ref is not None for s in session.plan.steps)

    # A real chain narrates step progress around the iteration events.
    assert h.bus.types == [
        "step_started",
        "iteration_start",
        "iteration_result",
        "step_result",
        "step_started",
        "iteration_start",
        "iteration_result",
        "step_result",
        "result",
    ]


async def test_partial_chain_delivers_completed_steps() -> None:
    # Step 1 passes; step 2 never reaches the floor → ship step 1's best, a valid
    # partial result, and leave step 2 uncompleted.
    h = await make_harness(similarities=[0.9, 0.3, 0.3], k=1)
    await _set_plan(
        h,
        [
            EditStep(n=1, title="background", instruction="blue", targets=["background"]),
            EditStep(n=2, title="clothing", instruction="red shirt", targets=["clothing"]),
        ],
    )
    result = await h.run()

    assert isinstance(result, ResultEvent)
    assert result.best.iteration_n == 1
    session = await h.session()
    assert session.plan is not None
    assert [s.status for s in session.plan.steps] == ["completed", "pending"]


async def test_skipped_step_is_not_generated() -> None:
    h = await make_harness(similarities=[0.9])
    await _set_plan(
        h,
        [
            EditStep(n=1, title="bg", instruction="blue", targets=["background"]),
            EditStep(n=2, title="x", instruction="y", targets=["clothing"], status="skipped"),
        ],
    )
    result = await h.run()

    assert isinstance(result, ResultEvent)
    assert len(h.generator.calls) == 1  # only the pending step ran
    # The skipped step is visible on the stream, never silently dropped.
    skipped = [e for e in h.bus.events if e.type == "step_result" and e.status == "skipped"]
    assert len(skipped) == 1


# --- idempotency ---


async def test_idempotent_reentry_replays_the_same_result() -> None:
    # The scripted embedder would raise on any extra embed call, so an equal
    # second result also proves face-check never re-ran.
    h = await make_harness(similarities=[0.9])
    first = await h.run(idem_key="op-1")
    second = await h.run(idem_key="op-1")

    assert second == first
    assert len(h.generator.calls) == 1
    assert h.bus.types.count("result") == 1  # replay does not re-emit


async def test_distinct_idem_keys_run_independently() -> None:
    h = await make_harness(similarities=[0.9, 0.95])
    first = await h.run(idem_key="op-1")
    second = await h.run(idem_key="op-2")

    assert isinstance(first, ResultEvent)
    assert isinstance(second, ResultEvent)
    assert len(h.generator.calls) == 2


# --- missing reference ---


async def test_missing_reference_photo_fails_cleanly() -> None:
    # The anchor photo expired between session start and the paid loop: the loop
    # must fail with a typed terminal, never crash into a generic internal error.
    h = await make_harness(similarities=[0.9])
    h.storage._data.pop(PHOTO_REF)

    result = await h.run()

    assert isinstance(result, FailedEvent)
    assert result.code is FailureCode.REFERENCE_MISSING
    assert h.generator.calls == []  # nothing paid for
    assert (await h.session()).fsm_state is FsmState.FAILED
    assert result.cost_spent == Decimal("0")


# --- write-ahead ---


async def test_paid_frame_is_checkpointed_before_facecheck() -> None:
    class ExplodingEmbedder:
        async def embed(self, image: bytes) -> NDArray[np.float32]:
            raise RuntimeError("face-check infrastructure died")

    # A face-check crash on a paid frame does not blow up the session: the frame
    # is already traceable, the attempt is recorded with its error, and the loop
    # fails cleanly instead of crashing into a generic internal error.
    h = await make_harness(embedder=ExplodingEmbedder())
    outcome = await h.run()

    assert isinstance(outcome, FailedEvent)
    assert outcome.code is FailureCode.GENERATION_FAILED

    session = await h.session()
    first = session.iterations[0]
    # The crash hit *after* the paid frame became traceable: the checkpoint
    # already carries the provider request id and the stored result ref, and the
    # reservation was settled (the frame was billed).
    assert first.charged is True
    assert first.cost == GEN_COST
    assert first.provider_request_id is not None
    assert first.result_ref == f"sessions/{SESSION_KEY}/iterations/1"
    assert await h.storage.get(first.result_ref) == h.generator.image
    assert first.similarity is None  # face-check never finished
    assert first.error is not None  # ...but the gap is explained

    # Every charged-but-unmeasured attempt still surfaces on the stream, so the
    # event count matches the recorded attempts (no silent under-count).
    assert h.bus.types.count("iteration_result") == len(session.iterations)
    assert all(it.charged and it.error is not None for it in session.iterations)
