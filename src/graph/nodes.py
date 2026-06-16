"""Graph nodes — thin orchestration over services.

Each node only calls services, narrates its stage to the EventBus, and returns
the state delta; every decision (gate bands, matching, prompt validation,
money) lives in the services. Nodes are built by a factory so dependencies
arrive via DI, never as globals.

Interrupt semantics that shape the design: on resume LangGraph re-executes the
interrupted node *from the top*, so `ask` and `approve` must not publish
anything before ``interrupt()`` — a pre-interrupt publish would duplicate on
every resume. (Store writes before the interrupt are fine: rewriting the same
session state is idempotent, re-appending an event is not.) The runner
(api/deps.py) emits the ``need_input`` event from the surfaced interrupt
payload exactly once instead.

Preset resolution is deliberately repeated in several nodes instead of being
threaded through the state: the library is immutable for the process lifetime
and ``resolve`` is a pure in-memory lookup, while a re-executed interrupt node
could not trust state it wrote before pausing anyway.
"""

from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol

from langgraph.types import interrupt

from graph.state import GraphState
from protocols import (
    BriefParser,
    EventBus,
    InventoryExtractor,
    ObjectStorage,
    SlotFiller,
    StateStore,
    StepPlanner,
)
from schemas import (
    BriefAnalysis,
    CompositionChoice,
    CostEvent,
    DoneEvent,
    EditStep,
    FailedEvent,
    FailureCode,
    FsmState,
    GateReason,
    PaidCallKind,
    PaidCallRecord,
    Plan,
    PlanEvent,
    Preset,
    ResultEvent,
    SessionState,
    Slot,
    StageEvent,
    Verdict,
)
from services.budget import BudgetService
from services.estimator import estimate_cost
from services.generation_loop import GenerationLoop
from services.planner import deterministic_steps
from services.preset_matcher import PresetLibrary
from services.pricing import PricingTable
from services.prompt_builder import FreeFormRejectedError, assemble_prompt, fill_template
from services.slot_filler import apply_composition
from services.vision import VisionService, face_crop_ref, photo_ref
from utils.money import from_micro


class NodeFn(Protocol):
    """A graph node. A Protocol (not a Callable alias) because LangGraph's
    `add_node` overloads require the ``state`` parameter to be passable by
    keyword, which a bare Callable type erases."""

    def __call__(self, state: GraphState) -> Awaitable[dict[str, Any]]: ...


@dataclass(frozen=True, slots=True)
class GraphServices:
    """Everything the nodes call — assembled once in api/deps.py."""

    store: StateStore
    storage: ObjectStorage
    bus: EventBus
    vision: VisionService
    library: PresetLibrary
    slot_filler: SlotFiller
    brief_parser: BriefParser
    inventory: InventoryExtractor
    planner: StepPlanner
    generation_loop: GenerationLoop
    budget: BudgetService
    pricing: PricingTable
    generation_model: str
    session_ttl_seconds: int
    face_ttl_seconds: int


# The plan-approval interrupt reuses the need_input mechanism; this reserved
# slot name tells the business service the question is the approval gate, not
# a preset slot.
APPROVAL_SLOT = "approve"
APPROVAL_QUESTION = "Approve the generation plan?"

# Free-form answers rejected as prompt injections are re-asked at most this
# many times before the session fails cleanly instead of ping-ponging forever.
MAX_REASKS = 2


# The generation loop is execute-once per session by construction: a crash
# between approval and delivery resumes into the *same* idempotency record, so
# already-paid attempts replay instead of being paid again.
def _generation_idem_key(session_key: str) -> str:
    return f"generation:{session_key}"


def _failure(
    reason: str, *, gate_reason: str | None = None, code: FailureCode = FailureCode.INTERNAL
) -> dict[str, Any]:
    return {"failure": {"reason": reason, "gate_reason": gate_reason, "code": code.value}}


def _locked_conflicts(preset: Preset, analysis: BriefAnalysis) -> list[str]:
    """Changes that target a locked attribute — surfaced; the lock still wins."""
    locked = {name for name, slot in preset.slots.items() if slot.policy == "locked"}
    return [
        f"requested change to '{change.target}' conflicts with a locked attribute "
        f"(the preset's fixed value is kept)"
        for change in analysis.changes
        if change.target in locked
    ]


def _plan_summary(
    preset: Preset,
    analysis: BriefAnalysis,
    steps: list[EditStep],
    slots: dict[str, str],
    budget_note: str | None,
) -> str:
    """Human-readable plan: the ordered steps (edit) or the styled target (generate)."""
    if analysis.mode == "edit" and steps:
        body = "; ".join(f"{s.n}. {s.title}: {s.instruction}" for s in steps)
    else:
        body = ", ".join(f"{name}={value}" for name, value in sorted(slots.items()))
    summary = f"{preset.id} v{preset.version}: {body}"
    if analysis.conflicts:
        summary += " | conflicts (locked attributes win): " + "; ".join(analysis.conflicts)
    if budget_note:
        summary += f" | {budget_note}"
    return summary


def _question_for(name: str, slot: Slot, *, reask_reason: str | None) -> str:
    base = f"What should the {name} of the photos be?"
    if reask_reason:
        return f"{reask_reason} {base}"
    return base


def _parse_decision(value: Any) -> tuple[bool, str | None]:
    """Normalize the approve-resume payload: a dict from the API, a bare string
    from a manual driver."""
    if isinstance(value, dict):
        composition = value.get("composition_id")
        return bool(value.get("approved")), str(composition) if composition else None
    return str(value).strip().lower() in {"approve", "approved", "yes", "true"}, None


def make_nodes(svc: GraphServices) -> dict[str, NodeFn]:
    """Bind the nodes to their services; returns them keyed by graph node name."""

    async def _session(session_key: str) -> SessionState:
        session = await svc.store.get_session(session_key)
        if session is None:
            raise ValueError(f"session {session_key!r} not found or expired")
        return session

    async def _checkpoint(session: SessionState) -> None:
        await svc.store.put_session(session, ttl_seconds=svc.session_ttl_seconds)

    def _resolve(state: GraphState) -> Preset:
        preset = svc.library.resolve(use_case=state["use_case"])
        if preset is None:  # the ask node already routed this case to fail
            raise ValueError(f"no preset admits use_case={state['use_case']!r}")
        return preset

    async def analyze(state: GraphState) -> dict[str, Any]:
        """Ensure the session record and the face profile exist; reuse a stored
        profile (an earlier ingest or session) instead of re-running detection."""
        session_key, face_key = state["session_key"], state["face_key"]
        session = await svc.store.get_session(session_key)
        if session is None:
            session = SessionState(
                session_key=session_key,
                face_key=face_key,
                budget_limit=from_micro(state["budget_limit"]),
            )
        session.fsm_state = FsmState.FACE_CHECK
        await _checkpoint(session)
        await svc.bus.publish(session_key, StageEvent(stage=FsmState.FACE_CHECK))

        face = await svc.store.get_face(face_key)
        if face is None:
            photo = await svc.storage.get(photo_ref(face_key))
            ingest = await svc.vision.ingest(
                photo, face_key=face_key, photo_ref=photo_ref(face_key)
            )
            face = ingest.profile
            await svc.store.put_face(face, ttl_seconds=svc.face_ttl_seconds)
            if ingest.face_crop is not None:
                await svc.storage.put(face_crop_ref(face_key), ingest.face_crop)

        return {
            "gate_verdict": face.gate_verdict.value,
            "gate_reason": face.gate_reason.value,
            "has_identity": bool(face.embedding),
        }

    async def quality_gate(state: GraphState) -> dict[str, Any]:
        """Route on the banded verdict. SOFT proceeds: by the contract the
        business service obtained the user's risk confirmation at ingest,
        before this session was started."""
        if state.get("gate_verdict") == Verdict.BELOW_FLOOR.value or not state.get("has_identity"):
            return _failure(
                "input photo cannot anchor the identity",
                gate_reason=state.get("gate_reason"),
                code=FailureCode.INPUT_REJECTED,
            )
        return {}

    async def parse_brief(state: GraphState) -> dict[str, Any]:
        """Read the brief into a :class:`BriefAnalysis` — mode, preserve-list,
        changes, conflicts — replacing the lone use-case token.

        A budgeted LLM call with a free deterministic fallback (it never fails):
        a caller-supplied ``use_case`` steers a target-driven generate; otherwise
        the parser decides edit vs generate and picks the ``use_case``. The
        analysis is stored on the session; resolution and planning read it there.
        """
        meter = svc.budget.meter(
            state["session_key"],
            limit=from_micro(state["budget_limit"]),
            ttl_seconds=svc.session_ttl_seconds,
        )
        result = await svc.brief_parser.parse(
            brief=state.get("brief", ""),
            use_case=state["use_case"] or None,
            use_cases=svc.library.use_case_tokens,
            meter=meter,
        )
        session = await _session(state["session_key"])
        session.brief_analysis = result.analysis
        if result.cost > 0:
            session.llm_calls.append(
                PaidCallRecord(kind=PaidCallKind.CLASSIFY, cost=result.cost, usage=result.usage)
            )
        await _checkpoint(session)
        # Empty use_case (an edit) resolves to the fallback edit preset.
        return {"use_case": result.analysis.use_case or ""}

    async def extract_inventory(state: GraphState) -> dict[str, Any]:
        """Catalogue the reference photo for edit-mode prompts — once per photo.

        Edit mode only: a generate-mode session never pays the VLM call. A
        profile that already carries an inventory (an earlier session on the
        same photo) is reused as-is. An empty extraction (the fallback) is not
        stored, so a transient failure retries on the next session instead of
        freezing emptiness for the profile's whole TTL.
        """
        session = await _session(state["session_key"])
        analysis = session.brief_analysis
        if analysis is None or analysis.mode != "edit":
            return {}
        face = await svc.store.get_face(state["face_key"])
        if face is None or face.inventory is not None:
            return {}
        try:
            photo = await svc.storage.get(photo_ref(state["face_key"]))
        except KeyError:
            return {}  # the generate node owns the missing-reference terminal
        meter = svc.budget.meter(
            state["session_key"],
            limit=from_micro(state["budget_limit"]),
            ttl_seconds=svc.session_ttl_seconds,
        )
        result = await svc.inventory.extract(photo, meter=meter)
        if result.cost > 0:
            session.llm_calls.append(
                PaidCallRecord(kind=PaidCallKind.INVENTORY, cost=result.cost, usage=result.usage)
            )
            await _checkpoint(session)
        if not result.inventory.is_empty():
            face.inventory = result.inventory
            await svc.store.put_face(face, ttl_seconds=svc.face_ttl_seconds)
        return {}

    async def ask(state: GraphState) -> dict[str, Any]:
        """The single clarifying question — the preset's ``ask:true`` slot."""
        preset = svc.library.resolve(use_case=state["use_case"])
        if preset is None:
            return _failure(
                f"no preset admits use_case={state['use_case']!r} and no fallback ships",
                code=FailureCode.NO_PRESET,
            )
        asked = next(((n, s) for n, s in preset.slots.items() if s.ask), None)
        if asked is None:
            return {"answer": None}
        name, slot = asked

        # A free-form ask slot (the fallback's scene) is answered by the user's
        # brief when there is one — no need to round-trip the same question. The
        # filler takes the brief verbatim; the prompt builder sanitizes it. A
        # re-ask (poisoned brief) falls through to the interrupt for a fresh answer.
        brief = state.get("brief", "")
        if slot.enum is None and brief.strip() and not state.get("reask_reason"):
            return {"answer": brief}

        session = await _session(state["session_key"])
        session.fsm_state = FsmState.NEED_INPUT
        await _checkpoint(session)

        answer = interrupt(
            {
                "slot": name,
                "question": _question_for(name, slot, reask_reason=state.get("reask_reason")),
                "options": [str(o) for o in slot.enum] if slot.enum else None,
                "default": str(slot.default) if slot.default is not None else None,
            }
        )
        return {"answer": str(answer)}

    async def resolve_constraints(state: GraphState) -> dict[str, Any]:
        """Resolve the preset by mode+use_case, fill its slots, validate by
        assembling the writer-path body, and surface locked-attribute conflicts.

        A free-form answer that reads as a prompt injection routes back to
        ``ask`` (bounded by MAX_REASKS) — the poisoned text never survives in
        state or in the session record. Locked attributes a change would fight are
        recorded as conflicts (the lock still wins in the loop), never dropped.
        """
        session_key = state["session_key"]
        await svc.bus.publish(session_key, StageEvent(stage=FsmState.PLANNING))

        preset = _resolve(state)
        face = await svc.store.get_face(state["face_key"])
        meter = svc.budget.meter(
            session_key,
            limit=from_micro(state["budget_limit"]),
            ttl_seconds=svc.session_ttl_seconds,
        )
        fill = await svc.slot_filler.fill(
            preset=preset,
            user_answer=state.get("answer"),
            photo_analysis=face.metrics if face else None,
            meter=meter,
        )
        try:
            # Validate via the writer-assembly path: the deterministic body is the
            # filled template, sanitized by assemble_prompt exactly as before.
            assemble_prompt(preset, fill_template(preset, fill.slots, addendum=fill.addendum))
        except FreeFormRejectedError:
            if state.get("reasks", 0) >= MAX_REASKS:
                return _failure(
                    "scene description repeatedly rejected as a prompt injection",
                    code=FailureCode.SCENE_REJECTED,
                )
            return {
                "reasks": state.get("reasks", 0) + 1,
                "reask_reason": "The previous answer was rejected — describe only the scene,"
                " without instructions about the face or the prompt.",
            }

        session = await _session(session_key)
        analysis = session.brief_analysis or BriefAnalysis(mode="generate", use_case=preset.id)
        analysis = analysis.model_copy(
            update={"conflicts": [*analysis.conflicts, *_locked_conflicts(preset, analysis)]}
        )
        session.brief_analysis = analysis
        session.fsm_state = FsmState.PLANNING
        session.preset_id = preset.id
        session.preset_version = preset.version
        session.library_version = svc.library.library_version
        session.slots = fill.slots
        # Account the slot-fill spend against the dollar budget (the meter already
        # settled it; this records the line so cost_spent stays complete).
        if fill.cost > 0:
            session.llm_calls.append(
                PaidCallRecord(kind=PaidCallKind.SLOT_FILL, cost=fill.cost, usage=fill.usage)
            )
        # Frozen at match time so a later library update cannot retroactively
        # move this session's identity bar.
        session.thresholds = preset.thresholds
        await _checkpoint(session)

        return {
            "preset_id": preset.id,
            "slots": fill.slots,
            "addendum": fill.addendum,
            "reask_reason": None,
        }

    async def plan_steps(state: GraphState) -> dict[str, Any]:
        """Cut the changes into ordered steps, forecast, publish.

        The plan the user approves is the full step plan — never trimmed to the
        budget. Spending is greedy: the runtime reserves before each generation and
        stops cleanly when the next one would overdraw, shipping whatever steps
        completed. The forecast reports the floor (one generation per step) and the
        budget ceiling, so an under-funded chain reads as "may finish partial"
        rather than a pre-emptively shortened plan.
        """
        session_key = state["session_key"]
        preset = _resolve(state)
        budget_limit = from_micro(state["budget_limit"])
        session = await _session(session_key)
        analysis = session.brief_analysis or BriefAnalysis(mode="generate", use_case=preset.id)

        meter = svc.budget.meter(
            session_key, limit=budget_limit, ttl_seconds=svc.session_ttl_seconds
        )
        face = await svc.store.get_face(state["face_key"])
        plan_result = await svc.planner.plan(
            analysis=analysis,
            inventory=face.inventory if face is not None else None,
            meter=meter,
        )
        steps = plan_result.steps or deterministic_steps(analysis)
        if plan_result.cost > 0:
            session.llm_calls.append(
                PaidCallRecord(
                    kind=PaidCallKind.SLOT_FILL, cost=plan_result.cost, usage=plan_result.usage
                )
            )

        cost = estimate_cost(
            preset,
            budget_limit=budget_limit,
            pricing=svc.pricing,
            generation_model=svc.generation_model,
            step_count=len(steps),
        )
        proposed = Plan(
            summary=_plan_summary(preset, analysis, steps, state.get("slots", {}), cost.note),
            compositions=[
                CompositionChoice(id=c.id, label=c.label, preview_asset=c.preview_asset)
                for c in preset.compositions
            ],
            planned_generations=cost.generations,
            steps=steps,
        )

        session.plan = proposed
        session.cost_estimate = cost
        session.fsm_state = FsmState.AWAITING_APPROVAL
        await _checkpoint(session)

        await svc.bus.publish(session_key, PlanEvent(plan=proposed))
        await svc.bus.publish(session_key, CostEvent(cost=cost))
        return {}

    async def approve(state: GraphState) -> dict[str, Any]:
        """The second interrupt: the user approves the plan (and may pick a
        composition) before any budget is spent."""
        decision = interrupt(
            {
                "slot": APPROVAL_SLOT,
                "question": APPROVAL_QUESTION,
                "options": ["approve", "reject"],
                "default": None,
            }
        )
        approved, composition_id = _parse_decision(decision)
        if not approved:
            return _failure("plan rejected by the user", code=FailureCode.PLAN_REJECTED)

        slots = dict(state.get("slots", {}))
        if composition_id is not None:
            slots = apply_composition(_resolve(state), slots, composition_id)

        session = await _session(state["session_key"])
        session.approved = True
        session.slots = slots
        if session.plan is not None:
            session.plan.selected_composition = composition_id
        await _checkpoint(session)
        return {"slots": slots}

    async def generate(state: GraphState) -> dict[str, Any]:
        """Run the paid loop. Terminal narration (result/failed) is the loop's;
        the node only signals which terminal the graph should route to."""
        session_key = state["session_key"]
        await svc.bus.publish(session_key, StageEvent(stage=FsmState.GENERATING))
        try:
            face_crop: bytes | None = await svc.storage.get(face_crop_ref(state["face_key"]))
        except KeyError:
            face_crop = None  # retries just won't strengthen the reference

        outcome = await svc.generation_loop.run(
            session_key=session_key,
            preset=_resolve(state),
            idem_key=_generation_idem_key(session_key),
            face_crop=face_crop,
            addendum=state.get("addendum", ""),
        )
        return {"delivered": isinstance(outcome, ResultEvent)}

    async def done(state: GraphState) -> dict[str, Any]:
        session = await svc.store.get_session(state["session_key"])
        steps = session.plan.steps if session and session.plan else []
        completed = sum(1 for s in steps if s.status == "completed")
        await svc.bus.publish(
            state["session_key"],
            DoneEvent(steps_completed=completed, steps_total=len(steps)),
        )
        return {}

    async def fail(state: GraphState) -> dict[str, Any]:
        """Terminal for every pre-loop failure (the loop fails its own runs)."""
        failure = state.get("failure") or {}
        session = await svc.store.get_session(state["session_key"])
        if session is not None:
            session.fsm_state = FsmState.FAILED
            await _checkpoint(session)
        # A pre-loop failure spent nothing in the common case, but a session may
        # already carry charged iterations or slot-fill spend on resume — account
        # from the record.
        generations_spent = session.generations_spent() if session else 0
        iterations_used = len(session.iterations) if session else 0
        cost_spent = session.cost_spent() if session else Decimal("0")
        raw_reason = failure.get("gate_reason")
        raw_code = failure.get("code")
        await svc.bus.publish(
            state["session_key"],
            FailedEvent(
                code=FailureCode(raw_code) if raw_code else FailureCode.INTERNAL,
                reason=failure.get("reason") or "unknown failure",
                gate_reason=GateReason(raw_reason) if raw_reason else None,
                iterations_used=iterations_used,
                generations_spent=generations_spent,
                cost_spent=cost_spent,
            ),
        )
        return {}

    return {
        "analyze": analyze,
        "quality_gate": quality_gate,
        "parse_brief": parse_brief,
        "extract_inventory": extract_inventory,
        "ask": ask,
        "resolve_constraints": resolve_constraints,
        "plan_steps": plan_steps,
        "approve": approve,
        "generate": generate,
        "done": done,
        "fail": fail,
    }
