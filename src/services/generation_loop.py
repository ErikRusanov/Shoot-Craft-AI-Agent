"""Generation loop — drive a step plan until each result looks like the person.

The deterministic core of the product. A brief decomposes into ordered steps
(:class:`~schemas.state.EditStep`); the loop runs them in sequence, and within
each step runs today's retry cycle — write the body → generate → face-check →
keep-best → retry. The money rules live here, scoped inside a step:

- **Chaining.** The working image a step edits is the previous step's kept-best;
  step 1 edits the original photo. The **identity anchor is always the crop of
  the original photo**, attached on every attempt of every step, and face-check
  is **always** against the original embedding — so face fidelity cannot drift
  along the chain.
- **Keep-best per step.** The best frame at/above ``identity_floor`` survives the
  step; the delivered result is the **last completed step's** best. A partially
  completed chain (budget ran out, or a later step found no deliverable) is a
  valid result, not a failure — only a chain that completes *no* step fails.
- **The writer composes the body.** ``compose`` once per step, ``revise`` per
  retry (feedback = the prior attempt's face-check); the builder assembles the
  frozen blocks and locks around it. The deterministic writer reproduces the old
  filled-template + identity-emphasis behavior exactly.
- **Budget.** Reserve a padded estimate before each generation, settle to the
  real ``usage.cost``, refund a transport/4xx failure; exhaustion ends the chain
  cleanly with whatever was already delivered. Writer LLM spend (when any) is
  recorded as a ``SLOT_FILL`` line.
- **Write-ahead & idempotency** are unchanged: the paid frame is checkpointed
  before face-check, and the whole run is execute-once by ``idem_key``.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import structlog

from protocols import (
    EventBus,
    GenerationRefusedError,
    ImageGenerator,
    ObjectStorage,
    PromptWriter,
    StateStore,
)
from protocols.budget import BudgetMeter
from protocols.prompt_writer import WriteRequest, WriteResult, WriterFeedback
from schemas import (
    BestResult,
    EditStep,
    EventAdapter,
    FaceProfile,
    FailedEvent,
    FailureCode,
    FsmState,
    Iteration,
    IterationResultEvent,
    IterationStartEvent,
    PaidCallKind,
    PaidCallRecord,
    PhotoInventory,
    Preset,
    ProviderUsage,
    ResultEvent,
    RetryEvent,
    SessionState,
    StepResultEvent,
    StepStartedEvent,
    Thresholds,
    Verdict,
)
from services.budget import BudgetService
from services.facecheck import FaceCheckResult, FaceCheckService
from services.idempotency import IdempotencyService
from services.pricing import PricingTable
from services.prompt_builder import (
    BuiltPrompt,
    FreeFormRejectedError,
    assemble_edit_prompt,
    assemble_prompt,
    edit_lock_items,
    fill_template,
)
from utils.images import decode_rgb, encode_jpeg, nearest_aspect_ratio, upscale

logger = structlog.get_logger(__name__)

# Each retry halves the temperature: identity drift is the failure being
# corrected, and lower temperature trades variety for fidelity.
_TEMPERATURE_DECAY = 0.5

# image_config.aspect_ratio vocabulary shared by the Gemini image models
# (OpenRouter image-generation guide, June 2026). Edit mode snaps to the source
# photo's nearest ratio: forcing the preset's generate-mode ratio would make the
# model recompose the frame — the opposite of "copy it pixel-for-pixel".
_SUPPORTED_RATIOS = ("1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9")

# A completed step whose planner left `applied` empty still locks its outcome,
# just generically.
_APPLIED_FALLBACK = "the result of the earlier edit: {instruction}"

_BUDGET_EXHAUSTED = "budget exhausted before reaching the identity target"
_NO_DELIVERABLE = "no attempt reached the identity floor"
_REFERENCE_MISSING = "the reference photo is missing from storage"

# Quality-enhancement prompt body — short and positive, no "do not change X" lists.
# Research and A/B tests show that long locked-attribute enumerations suppress the
# "enhance" signal so hard the model produces no visible change; a brief positive
# framing ("studio quality") achieves the same identity preservation via
# identity_instruction (frozen, first) plus the external face-check.
_ENHANCE_BODY = "Improve the quality of this photo. Make it professional studio quality."


def _result_ref(session_key: str, n: int) -> str:
    return f"sessions/{session_key}/iterations/{n}"


def _usage_cost(usage: ProviderUsage | None) -> Decimal | None:
    """The billed cost off a usage block, or ``None`` to settle on the estimate."""
    return usage.cost if usage is not None else None


@dataclass(slots=True)
class _StepOutcome:
    """What one step produced — its best frame (if any) and why it stopped."""

    best: BestResult | None
    best_bytes: bytes | None
    failure_reason: str
    failure_code: FailureCode
    budget_exhausted: bool


class GenerationLoop:
    """Runs the per-step generate → face-check → keep-best cycle for one session."""

    def __init__(
        self,
        *,
        store: StateStore,
        storage: ObjectStorage,
        bus: EventBus,
        generator: ImageGenerator,
        writer: PromptWriter,
        facecheck: FaceCheckService,
        budget: BudgetService,
        pricing: PricingTable,
        idempotency: IdempotencyService,
        session_ttl_seconds: int,
        face_ttl_seconds: int,
        max_iterations: int,
        generation_model: str,
        upscale_factor: int = 0,
    ) -> None:
        self._store = store
        self._storage = storage
        self._bus = bus
        self._generator = generator
        self._writer = writer
        self._facecheck = facecheck
        self._budget = budget
        self._pricing = pricing
        self._idempotency = idempotency
        self._session_ttl = session_ttl_seconds
        self._upscale_factor = upscale_factor
        self._face_ttl = face_ttl_seconds
        self._max_iterations = max_iterations
        self._generation_model = generation_model

    async def run(
        self,
        *,
        session_key: str,
        preset: Preset,
        idem_key: str,
        face_crop: bytes | None = None,
        addendum: str = "",
    ) -> ResultEvent | FailedEvent:
        """Drive the chain to a terminal event, exactly once per ``idem_key``."""

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
        preset_meta = {
            "preset_id": session.preset_id,
            "preset_version": session.preset_version,
            "library_version": session.library_version,
        }
        try:
            reference = await self._storage.get(face.photo_ref)
        except KeyError:
            logger.warning(
                "generation_loop.reference_missing",
                session_key=session_key,
                photo_ref=face.photo_ref,
            )
            return await self._finish(session, _REFERENCE_MISSING, FailureCode.REFERENCE_MISSING)

        session.fsm_state = FsmState.GENERATING
        await self._checkpoint(session)
        meter = self._budget.meter(
            session_key, limit=session.budget_limit, ttl_seconds=self._session_ttl
        )

        preserve = list(session.brief_analysis.preserve) if session.brief_analysis else []
        locked = {
            name: session.slots[name]
            for name, slot in preset.slots.items()
            if slot.policy == "locked" and name in session.slots
        }
        defaults = {name: v for name, v in session.slots.items() if name not in locked}

        # The working image the next step edits — the original photo to begin with,
        # then each completed step's best. The identity anchor crop stays original.
        working = reference
        failure_reason, failure_code = _NO_DELIVERABLE, FailureCode.NO_DELIVERABLE
        steps = self._plan_steps(session, preset)
        # Step events narrate a real chain; a single-step (generate / legacy) run
        # stays on the plain iteration events, so its stream is unchanged.
        emit_steps = len(steps) > 1

        # Keep the source photo's composition: the frame's nearest supported ratio
        # overrides the preset's default value on every step.
        aspect_ratio = nearest_aspect_ratio(
            face.metrics.width, face.metrics.height, _SUPPORTED_RATIOS
        )
        # The cumulative lock ledger, derived from the step record so a resumed
        # chain re-locks exactly what its completed steps already changed.
        applied: list[str] = []
        edited_targets: list[str] = []
        edited_text: list[str] = []

        def absorb(completed: EditStep) -> None:
            applied.append(
                completed.applied or _APPLIED_FALLBACK.format(instruction=completed.instruction)
            )
            edited_targets.extend(completed.targets)
            edited_text.append(completed.instruction)

        for step in steps:
            if step.status == "completed":
                absorb(step)

        for step in steps:
            if step.status == "skipped":
                if emit_steps:
                    await self._bus.publish(
                        session_key, StepResultEvent(n=step.n, status="skipped")
                    )
                continue
            if step.status == "completed":  # a resumed chain replays past steps
                continue
            if emit_steps:
                await self._bus.publish(
                    session_key,
                    StepStartedEvent(n=step.n, title=step.title, targets=list(step.targets)),
                )
            lock_items = edit_lock_items(
                face.inventory,
                preserve=preserve,
                applied=applied,
                edited_targets=[*edited_targets, *step.targets],
                edited_text=" ".join([*edited_text, step.instruction]),
            )
            outcome = await self._run_step(
                session=session,
                face=face,
                preset=preset,
                thresholds=thresholds,
                meter=meter,
                step=step,
                preserve=preserve,
                locked=locked,
                defaults=defaults,
                addendum=addendum,
                working_image=working,
                face_crop=face_crop,
                preset_meta=preset_meta,
                lock_items=lock_items,
                applied=tuple(applied),
                aspect_ratio=aspect_ratio,
            )
            failure_reason, failure_code = outcome.failure_reason, outcome.failure_code
            if outcome.best is None:
                # This step produced no deliverable — stop the chain and ship
                # whatever earlier steps delivered (a partial chain is valid).
                if emit_steps:
                    await self._bus.publish(
                        session_key, StepResultEvent(n=step.n, status="pending")
                    )
                break
            step.status = "completed"
            step.result_ref = outcome.best.result_ref
            session.best_result = outcome.best  # the latest completed step wins
            working = outcome.best_bytes if outcome.best_bytes is not None else working
            absorb(step)  # the next step locks this one's outcome
            await self._checkpoint(session)
            if emit_steps:
                await self._bus.publish(
                    session_key,
                    StepResultEvent(
                        n=step.n,
                        status="completed",
                        result_ref=outcome.best.result_ref,
                        similarity=outcome.best.similarity,
                    ),
                )
            if outcome.budget_exhausted:
                break

        return await self._finish(session, failure_reason, failure_code)

    def _plan_steps(self, session: SessionState, preset: Preset) -> list[EditStep]:
        """The plan's steps; a session without a plan is one synthetic step.

        With no plan (the legacy single-shot path), a single synthetic step
        carries the resolved slots — the deterministic body is then exactly today's.
        """
        if session.plan is not None and session.plan.steps:
            return session.plan.steps
        return [EditStep(n=1, title=preset.id, instruction="", targets=[])]

    async def _run_step(
        self,
        *,
        session: SessionState,
        face: FaceProfile,
        preset: Preset,
        thresholds: Thresholds,
        meter: BudgetMeter,
        step: EditStep,
        preserve: list[str],
        locked: dict[str, str],
        defaults: dict[str, str],
        addendum: str,
        working_image: bytes,
        face_crop: bytes | None,
        preset_meta: dict[str, str | None],
        lock_items: list[str],
        applied: tuple[str, ...],
        aspect_ratio: str | None,
    ) -> _StepOutcome:
        """Run the retry cycle for one step against ``working_image``."""
        session_key = session.session_key
        lighting = face.inventory.lighting.strip() if face.inventory is not None else None

        # Enhance steps skip the writer entirely: a short positive body
        # ("studio quality") is more effective than a long locked-attribute
        # list — the latter suppresses the change signal so hard the model
        # produces no visible improvement.
        request: WriteRequest | None
        if step.is_enhance:
            base_built = assemble_prompt(preset, _ENHANCE_BODY)
            prev_body = ""
            request = None
        else:
            request = self._write_request(
                preset,
                step,
                preserve,
                locked,
                defaults,
                addendum,
                session.slots,
                inventory=face.inventory,
                applied=applied,
            )
            base = await self._writer.compose(request, photo_metrics=face.metrics, meter=meter)
            await self._record_writer_cost(session, base)
            base_built = self._assemble(
                preset,
                base.body,
                step=step,
                locked=locked,
                lock_items=lock_items,
                lighting=lighting,
            )
            prev_body = base.body

        base_temperature = base_built.params.temperature

        max_attempts = min(1 + thresholds.K_max_retries, self._max_iterations)
        step_best: BestResult | None = None
        step_best_bytes: bytes | None = None
        last: WriterFeedback = WriterFeedback(None, None)
        failure_reason, failure_code = _NO_DELIVERABLE, FailureCode.NO_DELIVERABLE

        for attempt in range(1, max_attempts + 1):
            retrying = attempt > 1
            n = len(session.iterations) + 1
            if retrying:
                if step.is_enhance:
                    # Fixed body — no revision possible; temperature still decays.
                    built = base_built
                else:
                    assert request is not None
                    revised = await self._writer.revise(
                        prev_body, last, request=request, meter=meter
                    )
                    await self._record_writer_cost(session, revised)
                    prev_body = revised.body
                    built = self._assemble(
                        preset,
                        revised.body,
                        step=step,
                        locked=locked,
                        lock_items=lock_items,
                        lighting=lighting,
                    )
                params = base_built.params.model_copy(
                    update={"temperature": base_temperature * _TEMPERATURE_DECAY ** (attempt - 1)}
                )
            else:
                built = base_built
                params = base_built.params
            if aspect_ratio is not None:
                params = params.model_copy(update={"aspect_ratio": aspect_ratio})
            params = params.model_copy(update={"output_size": "4K" if step.is_enhance else "1K"})

            reservation = await meter.reserve(
                PaidCallKind.GENERATION,
                estimate=self._reserve_estimate(
                    built, send_crop=face_crop, output_size=params.output_size
                ),
            )
            if reservation is None:
                logger.info("generation_loop.budget_exhausted", session_key=session_key, n=n)
                return _StepOutcome(
                    step_best,
                    step_best_bytes,
                    _BUDGET_EXHAUSTED,
                    FailureCode.BUDGET_EXHAUSTED,
                    True,
                )

            await self._bus.publish(session_key, IterationStartEvent(n=n))

            try:
                generated = await self._generator.generate(
                    prompt=built.text,
                    params=params,
                    reference_images=[working_image],
                    face_crop=face_crop,
                )
            except GenerationRefusedError as exc:
                cost = await reservation.settle(_usage_cost(exc.usage))
                failure_reason = f"generation refused: {exc}"
                failure_code = FailureCode.GENERATION_FAILED
                charged = cost > Decimal("0")
                session.iterations.append(
                    Iteration(
                        n=n,
                        step_n=step.n,
                        prompt_hash=built.prompt_hash,
                        prompt_text=built.text,
                        charged=charged,
                        cost=cost,
                        usage=exc.usage,
                        error=failure_reason,
                    )
                )
                await self._checkpoint(session)
                logger.warning(
                    "generation_loop.generate_refused", session_key=session_key, n=n, error=str(exc)
                )
                await self._bus.publish(
                    session_key,
                    IterationResultEvent(
                        n=n, charged=charged, cost=cost, error=failure_reason, **preset_meta
                    ),
                )
                if attempt < max_attempts:
                    await self._bus.publish(session_key, RetryEvent(n=n + 1, reason=failure_reason))
                continue
            except Exception as exc:
                await reservation.cancel()
                failure_reason = f"generation failed: {exc}"
                failure_code = FailureCode.GENERATION_FAILED
                session.iterations.append(
                    Iteration(
                        n=n,
                        step_n=step.n,
                        prompt_hash=built.prompt_hash,
                        prompt_text=built.text,
                        charged=False,
                        error=failure_reason,
                    )
                )
                await self._checkpoint(session)
                logger.warning(
                    "generation_loop.generate_failed", session_key=session_key, n=n, error=str(exc)
                )
                await self._bus.publish(
                    session_key,
                    IterationResultEvent(n=n, charged=False, error=failure_reason, **preset_meta),
                )
                if attempt < max_attempts:
                    await self._bus.publish(session_key, RetryEvent(n=n + 1, reason=failure_reason))
                continue

            cost = await reservation.settle(_usage_cost(generated.usage))
            result_bytes = generated.image_bytes
            if self._upscale_factor > 1:
                result_bytes = encode_jpeg(upscale(decode_rgb(result_bytes), self._upscale_factor))
            result_ref = await self._storage.put(_result_ref(session_key, n), result_bytes)
            iteration = Iteration(
                n=n,
                step_n=step.n,
                prompt_hash=built.prompt_hash,
                prompt_text=built.text,
                provider_request_id=generated.provider_request_id,
                result_ref=result_ref,
                charged=cost > Decimal("0"),
                cost=cost,
                usage=generated.usage,
            )
            session.iterations.append(iteration)
            await self._checkpoint(session)  # write-ahead: traceable before face-check

            try:
                check = await self._facecheck.check(
                    reference_embedding=face.embedding,
                    image=generated.image_bytes,
                    thresholds=thresholds,
                )
            except Exception as exc:
                failure_reason = f"face-check failed: {exc}"
                failure_code = FailureCode.GENERATION_FAILED
                iteration.error = failure_reason
                await self._checkpoint(session)
                logger.warning(
                    "generation_loop.facecheck_failed", session_key=session_key, n=n, error=str(exc)
                )
                await self._bus.publish(
                    session_key,
                    IterationResultEvent(
                        n=n,
                        charged=iteration.charged,
                        cost=iteration.cost,
                        result_ref=result_ref,
                        error=failure_reason,
                        **preset_meta,
                    ),
                )
                if attempt < max_attempts:
                    await self._bus.publish(session_key, RetryEvent(n=n + 1, reason=failure_reason))
                continue

            iteration.similarity = check.similarity
            iteration.verdict = check.verdict
            iteration.risk_level = check.risk_level
            self._track_convergence(face, check.similarity)
            if self._keep_best_in_step(step_best, check):
                step_best = BestResult(
                    iteration_n=n,
                    result_ref=result_ref,
                    similarity=check.similarity,
                    verdict=check.verdict,
                    risk_level=check.risk_level,
                )
                step_best_bytes = generated.image_bytes
            await self._checkpoint(session)
            await self._store.put_face(face, ttl_seconds=self._face_ttl)

            await self._bus.publish(
                session_key,
                IterationResultEvent(
                    n=n,
                    similarity=check.similarity,
                    verdict=check.verdict,
                    risk_level=check.risk_level,
                    charged=iteration.charged,
                    cost=iteration.cost,
                    result_ref=result_ref,
                    **preset_meta,
                ),
            )
            logger.info(
                "generation_loop.iteration",
                session_key=session_key,
                n=n,
                step_n=step.n,
                similarity=round(check.similarity, 4),
                verdict=check.verdict,
            )

            if check.verdict is Verdict.PASSED:
                break
            last = WriterFeedback(check.similarity, check.verdict, attempt)
            failure_reason, failure_code = _NO_DELIVERABLE, FailureCode.NO_DELIVERABLE
            if attempt < max_attempts:
                await self._bus.publish(
                    session_key,
                    RetryEvent(
                        n=n + 1,
                        reason="identity similarity below target",
                        previous_verdict=check.verdict,
                    ),
                )

        return _StepOutcome(step_best, step_best_bytes, failure_reason, failure_code, False)

    def _assemble(
        self,
        preset: Preset,
        body: str,
        *,
        step: EditStep,
        locked: dict[str, str],
        lock_items: list[str],
        lighting: str | None = None,
    ) -> BuiltPrompt:
        """Assemble the edit prompt with the lock block.

        A step instruction the guards reject degrades to the legacy assembly
        (belt-and-braces — it already passed resolve-time sanitization); a
        poisoned *body* re-raises out of ``assemble_prompt`` exactly as before.
        """
        if step.instruction.strip():
            try:
                return assemble_edit_prompt(
                    preset,
                    body,
                    only_change=step.instruction,
                    lock_items=lock_items,
                    locks=locked,
                    lighting=lighting,
                )
            except FreeFormRejectedError:
                logger.warning("generation_loop.edit_assembly_rejected", step_n=step.n)
        return assemble_prompt(preset, body, locks=locked)

    def _write_request(
        self,
        preset: Preset,
        step: EditStep,
        preserve: list[str],
        locked: dict[str, str],
        defaults: dict[str, str],
        addendum: str,
        slots: dict[str, str],
        *,
        inventory: PhotoInventory | None = None,
        applied: tuple[str, ...] = (),
    ) -> WriteRequest:
        """Build the writer's request — and its deterministic fallback body."""
        template_body = fill_template(
            preset, self._slots_for_step(preset, step, slots), addendum=addendum
        )
        return WriteRequest(
            instruction=step.instruction,
            preserve=preserve,
            locked=locked,
            defaults=defaults,
            style_notes=preset.style_notes,
            template_body=template_body,
            inventory=inventory,
            applied=applied,
        )

    @staticmethod
    def _slots_for_step(preset: Preset, step: EditStep, slots: dict[str, str]) -> dict[str, str]:
        """The step's instruction fills the preset's free-form slot so the
        deterministic body carries that step's change."""
        merged = dict(slots)
        if not step.instruction:
            return merged
        free = next((name for name, s in preset.slots.items() if s.enum is None), None)
        if free is not None:
            merged[free] = step.instruction
        return merged

    @staticmethod
    def _keep_best_in_step(current: BestResult | None, check: FaceCheckResult) -> bool:
        """Strictly-better-similarity wins within a step; below-floor never enters."""
        if check.verdict is Verdict.BELOW_FLOOR:
            return False
        return current is None or check.similarity > current.similarity

    async def _record_writer_cost(self, session: SessionState, result: WriteResult) -> None:
        """Account writer LLM spend (when any) so cost_spent stays complete."""
        if result.cost > Decimal("0"):
            session.llm_calls.append(
                PaidCallRecord(kind=PaidCallKind.SLOT_FILL, cost=result.cost, usage=result.usage)
            )
            await self._checkpoint(session)

    def _reserve_estimate(
        self, built: BuiltPrompt, *, send_crop: bytes | None, output_size: str = "1K"
    ) -> Decimal:
        """Padded USD reservation for one generation with this exact prompt."""
        return self._pricing.generation_reserve(
            self._generation_model,
            prompt_chars=len(built.text),
            reference_count=1,
            output_size=output_size,
            face_detail=built.params.face_media_resolution if send_crop is not None else None,
        )

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
        self, session: SessionState, failure_reason: str, failure_code: FailureCode
    ) -> ResultEvent | FailedEvent:
        """Deliver the last completed step's best, else fail cleanly.

        Accounting travels with either terminal so the business service needs no
        second snapshot read.
        """
        iterations_used = len(session.iterations)
        generations_spent = session.generations_spent()
        cost_spent = session.cost_spent()

        if session.best_result is not None:
            session.fsm_state = FsmState.DONE
            await self._checkpoint(session)
            result = ResultEvent(
                best=session.best_result,
                iterations_used=iterations_used,
                generations_spent=generations_spent,
                cost_spent=cost_spent,
                preset_id=session.preset_id,
                preset_version=session.preset_version,
                library_version=session.library_version,
            )
            await self._bus.publish(session.session_key, result)
            return result

        session.fsm_state = FsmState.FAILED
        await self._checkpoint(session)
        failed = FailedEvent(
            code=failure_code,
            reason=failure_reason,
            iterations_used=iterations_used,
            generations_spent=generations_spent,
            cost_spent=cost_spent,
        )
        await self._bus.publish(session.session_key, failed)
        return failed

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
