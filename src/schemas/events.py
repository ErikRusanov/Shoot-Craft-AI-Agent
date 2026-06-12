"""SSE stream events — the discriminated union the API tails to the client.

Every event is published to the Redis Stream ``events:{session_key}`` and
relayed over SSE. ``type`` is the discriminator; :data:`Event` is the tagged
union and :data:`EventAdapter` parses an unknown payload into the right member
(an unrecognized ``type`` fails loudly — strict union, no silent passthrough).

These narrate the lifecycle in :class:`~schemas.enums.FsmState` order:
``stage`` per transition, then ``need_input`` / ``plan`` / ``cost`` at the
gates, ``iteration_start`` / ``iteration_result`` / ``retry`` inside the loop,
and ``result`` + ``done`` (or ``failed``) at the end.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter

from schemas.base import SchemaModel
from schemas.enums import FailureCode, FsmState, GateReason, RiskLevel, Verdict
from schemas.state import BestResult, CostEstimate, Plan, StepStatus


class StageEvent(SchemaModel):
    """FSM moved to ``stage``. The coarse progress signal."""

    type: Literal["stage"] = "stage"
    stage: FsmState
    detail: str | None = None


class NeedInputEvent(SchemaModel):
    """The single clarifying question (the preset's ``ask:true`` slot)."""

    type: Literal["need_input"] = "need_input"
    slot: str
    question: str
    options: list[str] | None = None  # enum choices, if the slot constrains them
    default: str | None = None


class PlanEvent(SchemaModel):
    """The proposed plan, awaiting approval."""

    type: Literal["plan"] = "plan"
    plan: Plan


class CostEvent(SchemaModel):
    """The paid-generation forecast for the plan."""

    type: Literal["cost"] = "cost"
    cost: CostEstimate


class StepStartedEvent(SchemaModel):
    """Edit/generation step ``n`` of the chain has begun."""

    type: Literal["step_started"] = "step_started"
    n: int
    title: str
    targets: list[str] = Field(default_factory=list)


class StepResultEvent(SchemaModel):
    """Outcome of chain step ``n``.

    ``status`` is ``completed`` (its best frame feeds the next step),
    ``skipped`` (budget-trimmed, never generated) or ``pending`` (ran but reached
    no deliverable — the chain stops here, earlier steps still ship).
    """

    type: Literal["step_result"] = "step_result"
    n: int
    status: StepStatus
    result_ref: str | None = None
    similarity: float | None = None


class IterationStartEvent(SchemaModel):
    """Generation attempt ``n`` has begun."""

    type: Literal["iteration_start"] = "iteration_start"
    n: int


class IterationResultEvent(SchemaModel):
    """Outcome of attempt ``n``.

    Published for *every* attempt, including failed ones, so the stream alone
    counts the real attempts and spend. A measured frame carries
    ``similarity``/``verdict``/``risk_level``; an attempt that produced no
    measured frame (provider error, or a face-check that crashed on a paid
    frame) carries ``error`` instead and leaves those ``None``. ``charged``
    distinguishes a paid-but-unmeasured frame from a transport-failed one, and
    ``cost`` is the dollars this attempt settled (0 when nothing was charged).
    The preset triple pins which thresholds/version produced the score.
    """

    schema_v: int = 3
    type: Literal["iteration_result"] = "iteration_result"
    n: int
    charged: bool
    cost: Decimal = Decimal("0")  # USD settled for this attempt
    similarity: float | None = None
    verdict: Verdict | None = None
    risk_level: RiskLevel | None = None
    result_ref: str | None = None
    error: str | None = None  # set when the attempt produced no measured frame
    preset_id: str | None = None
    preset_version: str | None = None
    library_version: str | None = None


class RetryEvent(SchemaModel):
    """About to retry; ``n`` is the upcoming attempt."""

    type: Literal["retry"] = "retry"
    n: int
    reason: str
    previous_verdict: Verdict | None = None


class ResultEvent(SchemaModel):
    """The delivered keep-best image.

    Self-sufficient: it carries what the run cost (``iterations_used``,
    ``generations_spent``, ``cost_spent``) and the preset triple that
    produced it, so the business service needs no second snapshot read and
    the result stays reproducible after a library update. ``cost_spent`` is the
    real dollars billed (generations plus any auxiliary LLM calls). The image
    bytes themselves live in object storage under ``best.result_ref``.
    """

    schema_v: int = 3
    type: Literal["result"] = "result"
    best: BestResult
    iterations_used: int
    generations_spent: int
    cost_spent: Decimal
    preset_id: str | None = None
    preset_version: str | None = None
    library_version: str | None = None


class FailedEvent(SchemaModel):
    """Terminal failure.

    ``code`` is the machine-readable cause the business service maps to a user
    action; ``reason`` is free-text detail. ``gate_reason`` qualifies an
    ``INPUT_REJECTED`` with the photo signal. The accounting fields mirror
    ``ResultEvent`` so a terminal is self-sufficient even when budget was spent
    before the failure (e.g. a wall-clock timeout mid-generation).
    """

    schema_v: int = 3
    type: Literal["failed"] = "failed"
    code: FailureCode
    reason: str
    gate_reason: GateReason | None = None
    iterations_used: int = 0
    generations_spent: int = 0
    cost_spent: Decimal = Decimal("0")


class DoneEvent(SchemaModel):
    """Terminal success marker — closes the stream after ``result``.

    Carries per-step accounting so the business service can tell a fully executed
    plan from a partial chain (budget-trimmed or a step that found no
    deliverable): ``steps_completed`` of ``steps_total`` ran to a kept-best. Both
    are 0 for the legacy single-shot path that carries no step plan.
    """

    schema_v: int = 2
    type: Literal["done"] = "done"
    detail: str | None = None
    steps_completed: int = 0
    steps_total: int = 0


Event = Annotated[
    StageEvent
    | NeedInputEvent
    | PlanEvent
    | CostEvent
    | StepStartedEvent
    | StepResultEvent
    | IterationStartEvent
    | IterationResultEvent
    | RetryEvent
    | ResultEvent
    | FailedEvent
    | DoneEvent,
    Field(discriminator="type"),
]

# Parse a raw stream payload into the right member; rejects an unknown `type`.
EventAdapter: TypeAdapter[Event] = TypeAdapter(Event)
