"""Generation loop — drive attempts until the result looks like the person.

The deterministic core of the product: build the prompt once, then
generate → face-check → keep-best → retry, with every knob the retry turns
derived from the attempt number, never invented per run. The loop is the only
place a paid generation happens, so the money rules live here too:

- **Budget.** A slot is reserved (atomic ``check_and_incr``) immediately before
  the generator call it pays for. If that call fails without producing a frame
  (network, refusal), the reservation is *carried over* to the next attempt
  rather than burned — budget counts delivered frames, and the store has no
  refund primitive by design. Exhaustion before any deliverable frame is a
  clean ``failed``, not an exception.
- **Write-ahead.** ``provider_request_id`` and ``result_ref`` are checkpointed
  on the session *before* face-check runs: once the provider was paid, the
  frame must be traceable even if the process dies mid-measurement.
- **Keep-best.** The best frame at or above ``identity_floor`` survives; a
  worse later attempt can never displace it, and a below-floor frame is never
  delivered, period. Verdicts are bands of the same similarity score, so
  ordering by similarity already keeps ``passed`` ahead of ``soft``.
- **Retry ≤ K** (`thresholds.K_max_retries`, frozen from the preset), capped by
  the config-wide runaway ceiling. A retry *strengthens the reference* — adds
  the tight face crop, appends a fixed identity-emphasis addendum, lowers
  temperature — it never rewrites the prompt: the identity block and structure
  stay frozen (``build_prompt`` enforces it).
- **Idempotency.** The whole run is execute-once by ``idem_key``: re-entry
  replays the recorded terminal event without touching the generator or the
  budget.

The loop emits ``iteration_start`` / ``iteration_result`` / ``retry`` and ends
the stream's generation chapter with ``result`` or ``failed``; coarser stage
narration belongs to the graph, not here.
"""

from __future__ import annotations

import structlog

from protocols import EventBus, ImageGenerator, ObjectStorage, StateStore
from schemas import (
    BestResult,
    EventAdapter,
    FaceProfile,
    FailedEvent,
    FsmState,
    Iteration,
    IterationResultEvent,
    IterationStartEvent,
    Preset,
    ResultEvent,
    RetryEvent,
    SessionState,
    Verdict,
)
from services.budget import BudgetService
from services.facecheck import FaceCheckResult, FaceCheckService
from services.idempotency import IdempotencyService
from services.prompt_builder import build_prompt

logger = structlog.get_logger(__name__)

# The one sanctioned retry text: appended as an *addendum* after the frozen
# structure, never edited into it. Fixed wording keeps retries reproducible.
IDENTITY_EMPHASIS = (
    "Critical: render the exact same person as in the reference photo — identical "
    "facial geometry, eyes, nose, lips, jawline, skin tone and texture. "
    "A faithful, recognizable likeness matters more than any stylistic choice."
)

# Each retry halves the temperature: identity drift is the failure being
# corrected, and lower temperature trades variety for fidelity.
_TEMPERATURE_DECAY = 0.5

_BUDGET_EXHAUSTED = "budget exhausted before reaching the identity target"
_NO_DELIVERABLE = "no attempt reached the identity floor"


def _result_ref(session_key: str, n: int) -> str:
    return f"sessions/{session_key}/iterations/{n}"


class GenerationLoop:
    """Runs the generate → face-check → keep-best cycle for one session."""

    def __init__(
        self,
        *,
        store: StateStore,
        storage: ObjectStorage,
        bus: EventBus,
        generator: ImageGenerator,
        facecheck: FaceCheckService,
        budget: BudgetService,
        idempotency: IdempotencyService,
        session_ttl_seconds: int,
        face_ttl_seconds: int,
        max_iterations: int,
    ) -> None:
        self._store = store
        self._storage = storage
        self._bus = bus
        self._generator = generator
        self._facecheck = facecheck
        self._budget = budget
        self._idempotency = idempotency
        self._session_ttl = session_ttl_seconds
        self._face_ttl = face_ttl_seconds
        self._max_iterations = max_iterations

    async def run(
        self,
        *,
        session_key: str,
        preset: Preset,
        idem_key: str,
        face_crop: bytes | None = None,
        addendum: str = "",
    ) -> ResultEvent | FailedEvent:
        """Drive the loop to a terminal event, exactly once per ``idem_key``.

        ``face_crop`` is the tight crop of the anchor face (prepared by the
        caller, which still has the detection bbox); retries attach it to
        strengthen the reference. ``addendum`` is the session's sanctioned
        free-text extension, passed through to every prompt build.
        """

        async def op() -> bytes:
            outcome = await self._drive(
                session_key=session_key, preset=preset, face_crop=face_crop, addendum=addendum
            )
            return EventAdapter.dump_json(outcome)

        payload, replayed = await self._idempotency.run_once(
            idem_key, ttl_seconds=self._session_ttl, op=op
        )
        if replayed:
            logger.info("generation_loop.replayed", session_key=session_key, idem_key=idem_key)
        event = EventAdapter.validate_json(payload)
        if not isinstance(event, ResultEvent | FailedEvent):
            raise RuntimeError(f"idempotency record for {idem_key!r} is not a terminal event")
        return event

    async def _drive(
        self, *, session_key: str, preset: Preset, face_crop: bytes | None, addendum: str
    ) -> ResultEvent | FailedEvent:
        session, face = await self._load(session_key)
        thresholds = session.thresholds
        if thresholds is None:
            raise ValueError(f"session {session_key!r} has no frozen thresholds")

        base = build_prompt(preset, session.slots, addendum=addendum)
        emphasized = build_prompt(
            preset, session.slots, addendum=f"{addendum.strip()}\n\n{IDENTITY_EMPHASIS}".strip()
        )
        reference = await self._storage.get(face.photo_ref)

        session.fsm_state = FsmState.GENERATING
        await self._checkpoint(session)

        # K retries after the first attempt, under the config runaway ceiling.
        max_attempts = min(1 + thresholds.K_max_retries, self._max_iterations)
        failure_reason = _NO_DELIVERABLE
        reserved = False

        for attempt in range(1, max_attempts + 1):
            retrying = attempt > 1
            n = len(session.iterations) + 1
            built = emphasized if retrying else base
            params = built.params
            if retrying:
                params = params.model_copy(
                    update={
                        "temperature": built.params.temperature
                        * _TEMPERATURE_DECAY ** (attempt - 1)
                    }
                )

            # The reservation is per delivered frame: a transport-failed attempt
            # left `reserved` standing, and this attempt spends it instead.
            if not reserved:
                reserved = await self._budget.reserve_generation(
                    session_key, limit=session.budget_limit, ttl_seconds=self._session_ttl
                )
                if not reserved:
                    failure_reason = _BUDGET_EXHAUSTED
                    logger.info("generation_loop.budget_exhausted", session_key=session_key, n=n)
                    break

            await self._bus.publish(session_key, IterationStartEvent(n=n))

            try:
                image, provider_request_id = await self._generator.generate(
                    prompt=built.text,
                    params=params,
                    reference_images=[reference],
                    face_crop=face_crop if retrying else None,
                )
            except Exception as exc:  # the port defines no error contract
                # Nothing was produced: record the attempt uncharged, keep the
                # reservation for the next one. The connector already retried
                # transient failures; reaching here means the attempt is dead.
                failure_reason = f"generation failed: {exc}"
                session.iterations.append(
                    Iteration(n=n, prompt_hash=built.prompt_hash, charged=False)
                )
                await self._checkpoint(session)
                logger.warning(
                    "generation_loop.generate_failed", session_key=session_key, n=n, error=str(exc)
                )
                if attempt < max_attempts:
                    await self._bus.publish(session_key, RetryEvent(n=n + 1, reason=failure_reason))
                continue
            reserved = False  # the slot is now spent on a real frame

            result_ref = await self._storage.put(_result_ref(session_key, n), image)
            iteration = Iteration(
                n=n,
                prompt_hash=built.prompt_hash,
                provider_request_id=provider_request_id,
                result_ref=result_ref,
                charged=True,
            )
            session.iterations.append(iteration)
            # Write-ahead: the paid frame is traceable (provider id + storage
            # key) before face-check gets a chance to crash or stall.
            await self._checkpoint(session)

            check = await self._facecheck.check(
                reference_embedding=face.embedding, image=image, thresholds=thresholds
            )
            iteration.similarity = check.similarity
            iteration.verdict = check.verdict
            iteration.risk_level = check.risk_level
            self._track_convergence(face, check.similarity)
            self._keep_best(session, iteration, check)
            await self._checkpoint(session)
            await self._store.put_face(face, ttl_seconds=self._face_ttl)

            await self._bus.publish(
                session_key,
                IterationResultEvent(
                    n=n,
                    similarity=check.similarity,
                    verdict=check.verdict,
                    risk_level=check.risk_level,
                    charged=True,
                    result_ref=result_ref,
                ),
            )
            logger.info(
                "generation_loop.iteration",
                session_key=session_key,
                n=n,
                similarity=round(check.similarity, 4),
                verdict=check.verdict,
            )

            if check.verdict is Verdict.PASSED:
                break
            failure_reason = _NO_DELIVERABLE
            if attempt < max_attempts:
                await self._bus.publish(
                    session_key,
                    RetryEvent(
                        n=n + 1,
                        reason="identity similarity below target",
                        previous_verdict=check.verdict,
                    ),
                )

        return await self._finish(session, failure_reason)

    async def _load(self, session_key: str) -> tuple[SessionState, FaceProfile]:
        session = await self._store.get_session(session_key)
        if session is None:
            raise ValueError(f"session {session_key!r} not found")
        face = await self._store.get_face(session.face_key)
        if face is None:
            raise ValueError(f"face profile {session.face_key!r} not found or expired")
        if not face.embedding:
            raise ValueError(f"face profile {session.face_key!r} carries no embedding")
        return session, face

    async def _finish(
        self, session: SessionState, failure_reason: str
    ) -> ResultEvent | FailedEvent:
        """Deliver keep-best if anything above the floor exists, else fail cleanly."""
        if session.best_result is not None:
            session.fsm_state = FsmState.DONE
            await self._checkpoint(session)
            result = ResultEvent(best=session.best_result)
            await self._bus.publish(session.session_key, result)
            return result

        session.fsm_state = FsmState.FAILED
        await self._checkpoint(session)
        failed = FailedEvent(reason=failure_reason)
        await self._bus.publish(session.session_key, failed)
        return failed

    @staticmethod
    def _keep_best(session: SessionState, iteration: Iteration, check: FaceCheckResult) -> None:
        """Strictly-better-similarity wins; below-floor frames never enter."""
        if check.verdict is Verdict.BELOW_FLOOR:
            return
        if session.best_result is not None and check.similarity <= session.best_result.similarity:
            return
        assert iteration.result_ref is not None  # set by the write-ahead checkpoint
        session.best_result = BestResult(
            iteration_n=iteration.n,
            result_ref=iteration.result_ref,
            similarity=check.similarity,
            verdict=check.verdict,
            risk_level=check.risk_level,
        )

    @staticmethod
    def _track_convergence(face: FaceProfile, similarity: float) -> None:
        conv = face.convergence
        prior_best = conv.best_similarity
        conv.attempts += 1
        conv.last_similarity = similarity
        conv.improved_last = prior_best is None or similarity > prior_best
        conv.best_similarity = similarity if prior_best is None else max(prior_best, similarity)

    async def _checkpoint(self, session: SessionState) -> None:
        await self._store.put_session(session, ttl_seconds=self._session_ttl)
