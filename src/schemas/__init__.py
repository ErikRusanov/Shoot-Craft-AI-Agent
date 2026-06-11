"""The contract. Every pydantic model the rest of the codebase types against
lives here and **only** here — API in/out, internal state, stream events,
presets, enums. Each aggregate root carries a ``schema_v``.

Import the names from this package, not the submodules, so the contract has one
public surface.
"""

from __future__ import annotations

from schemas.base import SchemaModel, StrictModel
from schemas.contract import (
    ApproveRequest,
    IngestPhotoRequest,
    IngestPhotoResponse,
    InputAnswerRequest,
    SessionAck,
    SessionSnapshot,
    StartSessionRequest,
    StartSessionResponse,
)
from schemas.enums import FailureCode, FsmState, GateReason, Gender, RiskLevel, Verdict
from schemas.events import (
    CostEvent,
    DoneEvent,
    Event,
    EventAdapter,
    FailedEvent,
    IterationResultEvent,
    IterationStartEvent,
    NeedInputEvent,
    PlanEvent,
    ResultEvent,
    RetryEvent,
    StageEvent,
)
from schemas.presets import (
    AppliesTo,
    Composition,
    ConvergenceProfile,
    Generation,
    Preset,
    Slot,
    Thresholds,
)
from schemas.state import (
    BestResult,
    CompositionChoice,
    ConvergenceStats,
    CostEstimate,
    FaceProfile,
    FrameMetrics,
    Iteration,
    Plan,
    SessionState,
)

__all__ = [
    "AppliesTo",
    "ApproveRequest",
    "BestResult",
    "Composition",
    "CompositionChoice",
    "ConvergenceProfile",
    "ConvergenceStats",
    "CostEstimate",
    "CostEvent",
    "DoneEvent",
    "Event",
    "EventAdapter",
    "FaceProfile",
    "FailedEvent",
    "FailureCode",
    "FrameMetrics",
    "FsmState",
    "GateReason",
    "Gender",
    "Generation",
    "IngestPhotoRequest",
    "IngestPhotoResponse",
    "InputAnswerRequest",
    "Iteration",
    "IterationResultEvent",
    "IterationStartEvent",
    "NeedInputEvent",
    "Plan",
    "PlanEvent",
    "Preset",
    "ResultEvent",
    "RetryEvent",
    "RiskLevel",
    "SchemaModel",
    "SessionAck",
    "SessionSnapshot",
    "SessionState",
    "Slot",
    "StageEvent",
    "StartSessionRequest",
    "StartSessionResponse",
    "StrictModel",
    "Thresholds",
    "Verdict",
]
