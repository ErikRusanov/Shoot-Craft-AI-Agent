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

from typing import Annotated, Literal

from pydantic import Field, TypeAdapter

from schemas.base import SchemaModel
from schemas.enums import FsmState, GateReason, RiskLevel, Verdict
from schemas.state import BestResult, CostEstimate, Plan


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


class IterationStartEvent(SchemaModel):
    """Generation attempt ``n`` has begun."""

    type: Literal["iteration_start"] = "iteration_start"
    n: int


class IterationResultEvent(SchemaModel):
    """Measured outcome of attempt ``n``."""

    type: Literal["iteration_result"] = "iteration_result"
    n: int
    similarity: float
    verdict: Verdict
    risk_level: RiskLevel
    charged: bool
    result_ref: str | None = None


class RetryEvent(SchemaModel):
    """About to retry; ``n`` is the upcoming attempt."""

    type: Literal["retry"] = "retry"
    n: int
    reason: str
    previous_verdict: Verdict | None = None


class ResultEvent(SchemaModel):
    """The delivered keep-best image."""

    type: Literal["result"] = "result"
    best: BestResult


class FailedEvent(SchemaModel):
    """Terminal failure. ``gate_reason`` set when the input photo was the cause."""

    type: Literal["failed"] = "failed"
    reason: str
    gate_reason: GateReason | None = None


class DoneEvent(SchemaModel):
    """Terminal success marker — closes the stream after ``result``."""

    type: Literal["done"] = "done"
    detail: str | None = None


Event = Annotated[
    StageEvent
    | NeedInputEvent
    | PlanEvent
    | CostEvent
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
