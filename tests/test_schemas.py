"""Contract tests: serialization round-trips, the event union, and preset validity.

These pin the *shape* — that every model survives a JSON round-trip unchanged,
that the discriminated event union resolves by ``type`` and rejects unknowns, and
that the shipped demo presets validate against the canonical :class:`Preset`.
No domain logic is exercised here; there is none yet.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel, ValidationError

from schemas import (
    ApproveRequest,
    BestResult,
    CostEstimate,
    CostEvent,
    DoneEvent,
    Event,
    EventAdapter,
    FaceProfile,
    FailedEvent,
    FailureCode,
    FrameMetrics,
    FsmState,
    GateReason,
    IngestPhotoRequest,
    IngestPhotoResponse,
    InputAnswerRequest,
    Iteration,
    IterationResultEvent,
    IterationStartEvent,
    NeedInputEvent,
    Plan,
    PlanEvent,
    Preset,
    ResultEvent,
    RetryEvent,
    RiskLevel,
    SessionAck,
    SessionSnapshot,
    SessionState,
    StageEvent,
    StartSessionRequest,
    StartSessionResponse,
    StepResultEvent,
    StepStartedEvent,
    Thresholds,
    Verdict,
)

_EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "presets" / "examples"


# --- fixtures: one fully-populated instance of each composite model ---------


def _frame_metrics() -> FrameMetrics:
    return FrameMetrics(
        face_count=1,
        face_area_ratio=0.42,
        blur_var=180.5,
        yaw=-3.1,
        pitch=2.0,
        roll=0.4,
        brightness=128.0,
        width=1024,
        height=1280,
    )


def _face_profile() -> FaceProfile:
    return FaceProfile(
        face_key="face_abc",
        embedding=[0.01, -0.2, 0.33, 0.9],
        gate_verdict=Verdict.PASSED,
        gate_reason=GateReason.OK,
        metrics=_frame_metrics(),
        photo_ref="s3://faces/face_abc/src.jpg",
    )


def _iteration() -> Iteration:
    return Iteration(
        n=2,
        prompt_hash="deadbeef",
        provider_request_id="req_1",
        result_ref="s3://res/2.png",
        similarity=0.71,
        verdict=Verdict.PASSED,
        charged=True,
        risk_level=RiskLevel.LOW,
    )


def _plan() -> Plan:
    return Plan(
        summary="Formal headshot, off-white background.",
        compositions=[],
        selected_composition="studio_clean",
        planned_generations=2,
    )


def _cost() -> CostEstimate:
    return CostEstimate(
        generations=2,
        budget_limit=Decimal("0.50"),
        per_generation_cost=Decimal("0.069"),
        llm_overhead_cost=Decimal("0.002"),
        total_cost=Decimal("0.140"),
        note="keep-best within budget",
    )


def _best() -> BestResult:
    return BestResult(
        iteration_n=2,
        result_ref="s3://res/2.png",
        similarity=0.71,
        verdict=Verdict.PASSED,
        risk_level=RiskLevel.LOW,
    )


def _session_state() -> SessionState:
    return SessionState(
        session_key="sess_1",
        face_key="face_abc",
        fsm_state=FsmState.GENERATING,
        preset_id="demo_headshot",
        preset_version="1.0.0",
        library_version="examples",
        slots={"purpose": "a resume or CV photo"},
        plan=_plan(),
        cost_estimate=_cost(),
        approved=True,
        iterations=[_iteration()],
        thresholds=Thresholds(similarity_threshold=0.62, identity_floor=0.5, K_max_retries=3),
        best_result=_best(),
        budget_limit=Decimal("0.50"),
        created_at=datetime(2026, 6, 9, 12, 0, tzinfo=UTC),
        updated_at=datetime(2026, 6, 9, 12, 5, tzinfo=UTC),
    )


# Every standalone model with a representative populated instance.
_ROUND_TRIP_CASES: list[BaseModel] = [
    _frame_metrics(),
    _face_profile(),
    _iteration(),
    _plan(),
    _cost(),
    _best(),
    _session_state(),
    IngestPhotoRequest(image_b64="Zm9v", idem_key="k1"),
    IngestPhotoResponse(
        face_key="face_abc",
        accepted=True,
        gate_verdict=Verdict.PASSED,
        gate_reason=GateReason.OK,
        metrics=_frame_metrics(),
    ),
    StartSessionRequest(
        face_key="face_abc",
        brief="a resume photo, navy suit",
        budget_limit=Decimal("0.50"),
        idem_key="k2",
    ),
    StartSessionResponse(
        session_key="sess_1",
        fsm_state=FsmState.NEED_INPUT,
        matched=True,
        preset_id="demo_headshot",
    ),
    InputAnswerRequest(
        session_key="sess_1", slot="purpose", value="a resume or CV photo", idem_key="k3"
    ),
    ApproveRequest(session_key="sess_1", approved=True, idem_key="k4"),
    SessionAck(session_key="sess_1", fsm_state=FsmState.PLANNING),
    SessionSnapshot(state=_session_state(), generations_spent=1),
]


@pytest.mark.parametrize("model", _ROUND_TRIP_CASES, ids=lambda m: type(m).__name__)
def test_json_round_trip(model: BaseModel) -> None:
    """JSON dump → load reconstructs an equal instance of the same type."""
    restored = type(model).model_validate_json(model.model_dump_json())
    assert restored == model


def test_every_aggregate_carries_schema_v() -> None:
    """Aggregate roots expose schema_v; it survives the round-trip as a field."""
    for model in _ROUND_TRIP_CASES:
        if isinstance(model, FrameMetrics | Plan | CostEstimate | BestResult):
            continue  # nested value objects intentionally have no own version
        assert "schema_v" in model.model_dump()


# --- discriminated event union ---------------------------------------------


_EVENT_CASES: list[Event] = [
    StageEvent(stage=FsmState.PLANNING, detail="building plan"),
    NeedInputEvent(slot="purpose", question="What is this for?", options=["a", "b"], default="a"),
    PlanEvent(plan=_plan()),
    CostEvent(cost=_cost()),
    StepStartedEvent(n=1, title="background", targets=["background", "lighting"]),
    StepResultEvent(n=1, status="completed", result_ref="s3://res/1.png", similarity=0.7),
    IterationStartEvent(n=1),
    IterationResultEvent(
        n=1,
        similarity=0.6,
        verdict=Verdict.SOFT,
        risk_level=RiskLevel.MEDIUM,
        charged=True,
        result_ref="s3://res/1.png",
    ),
    RetryEvent(n=2, reason="below target", previous_verdict=Verdict.SOFT),
    ResultEvent(
        best=_best(),
        iterations_used=2,
        generations_spent=2,
        cost_spent=Decimal("0.138"),
        preset_id="demo",
        preset_version="1.0.0",
        library_version="0.4.0",
    ),
    FailedEvent(code=FailureCode.INPUT_REJECTED, reason="no face", gate_reason=GateReason.NO_FACE),
    DoneEvent(detail="delivered", steps_completed=2, steps_total=3),
]


@pytest.mark.parametrize("event", _EVENT_CASES, ids=lambda e: e.type)
def test_event_union_round_trip(event: Event) -> None:
    """Serialized event re-parses through the union to the same member type."""
    restored = EventAdapter.validate_json(EventAdapter.dump_json(event))
    assert restored == event
    assert type(restored) is type(event)


def test_event_union_rejects_unknown_type() -> None:
    with pytest.raises(ValidationError):
        EventAdapter.validate_python({"type": "nope"})


def test_event_union_rejects_extra_key() -> None:
    with pytest.raises(ValidationError):
        EventAdapter.validate_python({"type": "iteration_start", "n": 1, "bogus": 1})


# --- demo presets validate against the canonical schema --------------------


@pytest.mark.parametrize("yaml_path", sorted(_EXAMPLES_DIR.glob("*.yaml")), ids=lambda p: p.stem)
def test_demo_presets_validate(yaml_path: Path) -> None:
    """Each shipped demo preset is valid against :class:`Preset`."""
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    preset = Preset(**data)
    assert preset.id == yaml_path.stem


def test_freeform_ask_slot_validates() -> None:
    """A free-form ask slot — ``ask:true`` with no ``enum`` — is valid (the
    fallback preset's ``scene``). The schema must not require an enum on ask slots."""
    from schemas.presets import Slot

    slot = Slot(required=True, ask=True, default="in a natural setting")
    assert slot.ask is True
    assert slot.enum is None
