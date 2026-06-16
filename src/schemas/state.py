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
from typing import Literal

from pydantic import Field

from schemas.base import SchemaModel, StrictModel
from schemas.brief import BriefAnalysis
from schemas.enums import FsmState, GateReason, PaidCallKind, RiskLevel, Verdict
from schemas.inventory import PhotoInventory
from schemas.presets import Thresholds

# Lifecycle of one plan step along the chain: pending → completed, or skipped
# when the planner trimmed it to fit the budget (recorded, never silent).
StepStatus = Literal["pending", "completed", "skipped"]


class FrameMetrics(StrictModel):
    """Quality measurements of the input frame — drives the gate, not biometrics.

    None of these identify a person; they describe the photo (how big/sharp/lit
    the face is) and are safe to log and surface.
    """

    face_count: int
    face_area_ratio: float  # face bbox area / frame area
    # min(w, h) of the primary face bbox, px. The gate thresholds this, not the
    # area ratio: identity quality is about absolute crop resolution — a 450px
    # face in a 2048px frame is a great anchor at only ~3% of the area.
    face_side: float = 0.0
    # Second-largest bbox area / primary bbox area; 0.0 with fewer than two
    # faces. Distinguishes a comparable second face (ambiguous identity) from
    # background passers-by, which are normal and must not fail the gate.
    secondary_face_ratio: float = 0.0
    # Variance of the Laplacian on the median-denoised face crop; higher =
    # sharper. Denoised, because sensor grain reads as high frequency and would
    # let a noisy-but-soft face pass for sharp.
    blur_var: float
    # Pose is observability only — never gated: a turned head is the user's
    # composition, not a rendering defect.
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

    schema_v: int = 3
    face_key: str
    # Identity vector — biometric, never logged. Empty when no face was detected
    # (the gate then says NO_FACE and the profile is unusable for generation).
    embedding: list[float]
    gate_verdict: Verdict
    gate_reason: GateReason
    metrics: FrameMetrics
    # No demographics: the face comes from the reference, so neither age nor
    # gender drives anything here. Preset matching is use_case only.
    convergence: ConvergenceStats = Field(default_factory=ConvergenceStats)
    photo_ref: str  # object-storage key of the input photo
    # What is visible in the reference photo, in words (v3) — extracted once per
    # photo for edit-mode prompts, reused across sessions on the same profile.
    # Appearance text, never logged; None until an edit session extracts it.
    inventory: PhotoInventory | None = None


class ProviderUsage(StrictModel):
    """What the upstream actually billed for one paid call.

    ``cost`` is the dollars OpenRouter reports it charged (``usage.cost``, now
    always returned) — the source of truth the budget settles against. The token
    counts are observability: they explain the cost and feed offline pricing
    calibration. All optional: a provider that omits ``usage`` leaves the meter
    to settle on its reserved estimate instead.
    """

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost: Decimal | None = None  # USD actually billed for this call


class PaidCallRecord(StrictModel):
    """One non-generation paid call (slot fill, use-case classification).

    Generation spend lives on :class:`Iteration`; these are the auxiliary LLM
    calls that share the same dollar budget. Kept on the session so
    ``cost_spent`` accounts for *every* dollar, not just the images.
    """

    kind: PaidCallKind
    cost: Decimal  # USD settled for this call
    usage: ProviderUsage | None = None


class Iteration(SchemaModel):
    """One generation attempt and its measured outcome.

    ``charged`` records whether this attempt counted against the budget (a failed
    provider call that produced nothing is not charged); it stays equivalent to
    ``cost > 0``. ``cost`` is the dollars this attempt settled — actual
    ``usage.cost`` when the provider reported it, else the reserved estimate.
    ``prompt_hash`` ties the attempt to the exact prompt for reproducibility.
    """

    schema_v: int = 3
    n: int
    # Which plan step this attempt belongs to (1 for a single-step / generate
    # run). The chain reference for step N+1 is step N's kept-best result.
    step_n: int = 1
    prompt_hash: str
    # The full prompt text, kept so a surprising result can be debugged from the
    # snapshot rather than blind (the hash alone proves reproducibility but says
    # nothing about *what* was sent). Scene text only — no biometrics — so it is
    # safe on the record, though it is never logged.
    prompt_text: str | None = None
    provider_request_id: str | None = None
    result_ref: str | None = None  # object-storage key of the generated image
    similarity: float | None = None
    verdict: Verdict | None = None
    charged: bool = False
    cost: Decimal = Decimal("0")  # USD settled for this attempt
    usage: ProviderUsage | None = None
    risk_level: RiskLevel | None = None
    # Why a charged frame stayed unmeasured (face-check crash) or an attempt
    # produced nothing (provider error) — so the history explains the gap.
    error: str | None = None


class CompositionChoice(StrictModel):
    """A composition variant offered to the user in the plan.

    The user-facing projection of a preset ``Composition`` — id, label, optional
    preview — without the internal ``slot_overrides``.
    """

    id: str
    label: str
    preview_asset: str | None = None


class EditStep(StrictModel):
    """One ordered step of an edit/generation plan.

    A complex brief decomposes into steps the generator chains: each step's
    kept-best result feeds the next. ``targets`` are the :class:`~schemas.brief.Change`
    targets this step applies (compatible deltas merge into one step). ``status``
    tracks chain progress; ``result_ref`` is the object-storage key of the step's
    kept-best frame once it completes.

    ``applied`` is a short noun phrase naming the changed attribute at its NEW
    value ("the new plain white crew-neck t-shirt") — once the step completes,
    later steps lock this phrase as untouchable so a chained edit cannot undo
    an earlier one. Filled by the LLM planner; empty falls back to a generic
    "result of the earlier edit" phrase in the loop.
    """

    n: int
    title: str
    instruction: str
    targets: list[str] = Field(default_factory=list)
    applied: str = ""
    status: StepStatus = "pending"
    result_ref: str | None = None


class Plan(StrictModel):
    """What the session intends to generate, shown to the user for approval."""

    summary: str
    compositions: list[CompositionChoice] = Field(default_factory=list)
    selected_composition: str | None = None
    # The plan's floor — one generation per step (zero retries). Spending is
    # greedy, so the actual count lands between this and the budget ceiling.
    planned_generations: int
    # The ordered edit/generation steps the user approves. Empty for a legacy
    # single-shot plan; a generate-mode plan carries exactly one step.
    steps: list[EditStep] = Field(default_factory=list)


class CostEstimate(StrictModel):
    """Paid-spend forecast for the plan, in real USD.

    Spending is greedy pay-as-you-go, so the plan is never trimmed to fit:
    ``generations`` is the **floor** — one generation per step, the best case with
    zero retries — and ``total_cost`` is the full ``budget_limit`` the session may
    consume. ``budget_limit`` is the per-session dollar ceiling from the business
    service; ``per_generation_cost`` is the realistic (unpadded) price of one
    generation; ``llm_overhead_cost`` the auxiliary LLM spend (slot fill,
    classification). ``note`` carries the budget ceiling (generations the padded
    reservation admits, retries included), flagging an under-funded chain as
    "may ship partial". All ``Decimal`` so the money arithmetic stays exact.
    """

    generations: int
    budget_limit: Decimal = Decimal("0")
    per_generation_cost: Decimal = Decimal("0")
    llm_overhead_cost: Decimal = Decimal("0")
    total_cost: Decimal = Decimal("0")
    # Optimistic lower bound: floor * per_generation_cost + llm_overhead_cost
    # (zero retries, best case). Use this to display "you need at least $X".
    minimum_cost: Decimal = Decimal("0")
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

    ``budget_limit`` is the per-session dollar ceiling (pay-as-you-go); every
    paid call — generations and the auxiliary LLM calls in ``llm_calls`` —
    settles against it, and :meth:`cost_spent` totals what was actually billed.
    """

    schema_v: int = 3
    session_key: str
    face_key: str
    fsm_state: FsmState = FsmState.CREATED
    preset_id: str | None = None
    preset_version: str | None = None
    library_version: str | None = None
    # The structured reading of the brief (mode, preserve-list, changes,
    # conflicts) — drives resolution, the step plan and the writer.
    brief_analysis: BriefAnalysis | None = None
    slots: dict[str, str] = Field(default_factory=dict)  # resolved slot values
    plan: Plan | None = None
    cost_estimate: CostEstimate | None = None
    approved: bool = False
    iterations: list[Iteration] = Field(default_factory=list)
    llm_calls: list[PaidCallRecord] = Field(default_factory=list)  # auxiliary paid LLM spend
    thresholds: Thresholds | None = None  # frozen at match time
    best_result: BestResult | None = None
    budget_limit: Decimal = Decimal("0")  # USD ceiling; from the business service
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def cost_spent(self) -> Decimal:
        """Total USD actually billed this session — generations plus LLM calls."""
        return sum((it.cost for it in self.iterations), Decimal("0")) + sum(
            (c.cost for c in self.llm_calls), Decimal("0")
        )

    def generations_spent(self) -> int:
        """Paid generations charged (settled cost > 0), not provider calls made."""
        return sum(1 for it in self.iterations if it.charged)
