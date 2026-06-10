"""Internal session state — the contract for what lives in Redis.

Two TTL-bound aggregates keyed independently:

- :class:`FaceProfile` (``face_key``) — the transient biometric profile built
  from one reference photo. Its ``embedding`` is identity data and is **never**
  logged.
- :class:`SessionState` (``session_key``) — the photoshoot's running state,
  referencing a face by ``face_key``.

Everything else here is a value object nested inside one of those two. The
``preset_id`` / ``preset_version`` / ``library_version`` triple on the session is
what makes a delivered result reproducible after the preset library updates.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import Field

from schemas.base import SchemaModel, StrictModel
from schemas.enums import FsmState, GateReason, Gender, RiskLevel, Verdict
from schemas.presets import Thresholds


class FrameMetrics(StrictModel):
    """Quality measurements of the input frame — drives the gate, not biometrics.

    None of these identify a person; they describe the photo (how big/sharp/lit
    the face is) and are safe to log and surface.
    """

    face_count: int
    face_area_ratio: float  # face bbox area / frame area
    blur_var: float  # variance of the Laplacian; higher = sharper
    yaw: float
    pitch: float
    roll: float
    brightness: float  # mean luma, 0..255
    width: int
    height: int


class ConvergenceStats(StrictModel):
    """Running identity-similarity stats across generation attempts.

    Lets the loop decide whether retrying is still buying improvement or has
    plateaued, without re-scanning every iteration.
    """

    attempts: int = 0
    best_similarity: float | None = None
    last_similarity: float | None = None
    improved_last: bool | None = None  # did the latest attempt beat the prior best


class FaceProfile(SchemaModel):
    """Transient, TTL-bound identity profile from one reference photo.

    Holds the biometric embedding (never logged), the quality-gate outcome and
    the frame metrics behind it, the running convergence stats, and the
    object-storage ref of the source photo.
    """

    face_key: str
    # Identity vector — biometric, never logged. Empty when no face was detected
    # (the gate then says NO_FACE and the profile is unusable for generation).
    embedding: list[float]
    gate_verdict: Verdict
    gate_reason: GateReason
    metrics: FrameMetrics
    # Estimated by the CV attribute model; generation hints for preset matching
    # (applies_to.gender / applies_to.age), not identity claims.
    gender: Gender | None = None
    age: int | None = None
    convergence: ConvergenceStats = Field(default_factory=ConvergenceStats)
    photo_ref: str  # object-storage key of the input photo


class Iteration(SchemaModel):
    """One generation attempt and its measured outcome.

    ``charged`` records whether this attempt counted against the budget (a failed
    provider call that produced nothing is not charged). ``prompt_hash`` ties the
    attempt to the exact prompt for reproducibility without storing the text.
    """

    n: int
    prompt_hash: str
    provider_request_id: str | None = None
    result_ref: str | None = None  # object-storage key of the generated image
    similarity: float | None = None
    verdict: Verdict | None = None
    charged: bool = False
    risk_level: RiskLevel | None = None


class CompositionChoice(StrictModel):
    """A composition variant offered to the user in the plan.

    The user-facing projection of a preset ``Composition`` — id, label, optional
    preview — without the internal ``slot_overrides``.
    """

    id: str
    label: str
    preview_asset: str | None = None


class Plan(StrictModel):
    """What the session intends to generate, shown to the user for approval."""

    summary: str
    compositions: list[CompositionChoice] = Field(default_factory=list)
    selected_composition: str | None = None
    planned_generations: int


class CostEstimate(StrictModel):
    """Paid-generation forecast for the plan.

    ``budget_limit`` is the hard ceiling supplied per session by the business
    service; ``generations`` is what the plan expects to spend under it.
    ``unit_price`` / ``total_cost`` are in abstract config units — mapping them
    to user-facing money is the business service's job, not the core's — but
    they are still price-like, so ``Decimal`` keeps the arithmetic exact.
    """

    generations: int
    budget_limit: int
    unit_price: Decimal = Decimal("0")
    total_cost: Decimal = Decimal("0")
    note: str | None = None


class BestResult(StrictModel):
    """The kept-best image — the one the session would deliver right now."""

    iteration_n: int
    result_ref: str
    similarity: float
    verdict: Verdict
    risk_level: RiskLevel


class SessionState(SchemaModel):
    """Running state of a single photoshoot session, stored under ``session_key``.

    The preset triple (``preset_id`` + ``preset_version`` + ``library_version``)
    pins the exact preset content used, so a result stays reproducible across
    library updates. ``thresholds`` is copied from the matched preset so a later
    library change cannot retroactively move this session's bar.
    """

    session_key: str
    face_key: str
    fsm_state: FsmState = FsmState.CREATED
    preset_id: str | None = None
    preset_version: str | None = None
    library_version: str | None = None
    slots: dict[str, str] = Field(default_factory=dict)  # resolved slot values
    plan: Plan | None = None
    cost_estimate: CostEstimate | None = None
    approved: bool = False
    iterations: list[Iteration] = Field(default_factory=list)
    thresholds: Thresholds | None = None  # frozen at match time
    best_result: BestResult | None = None
    budget_limit: int = 0  # paid generations allowed; from the business service
    created_at: datetime | None = None
    updated_at: datetime | None = None
