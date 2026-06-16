"""API request/response models — the wire contract with the business service.

The core is an autonomous worker behind a trusted caller: a request that reaches
here is already authenticated and paid for (see the core boundary). These models
encode *only* the photoshoot contract, never users, money, or limits.

Every mutating request carries ``idem_key`` — mutations are idempotent by it, so
a retried call returns the same outcome instead of doing the work twice.

Endpoints:

- ingest a reference photo → :class:`IngestPhotoRequest` / :class:`IngestPhotoResponse`
- start a session         → :class:`StartSessionRequest` / :class:`StartSessionResponse`
- answer the question     → :class:`InputAnswerRequest` / :class:`SessionAck`
- approve the plan        → :class:`ApproveRequest` / :class:`SessionAck`
- cancel the session      → no body (naturally idempotent) / :class:`SessionAck`
- read the state          → :class:`SessionSnapshot`

A mutation that cannot apply is an HTTP error, not an ack: 404 for a missing
aggregate, 409 for a wrong FSM stage / a run already in flight / a duplicate
start, 503 when the state store is unreachable.
"""

from __future__ import annotations

from decimal import Decimal

from pydantic import Field

from schemas.base import SchemaModel, StrictModel
from schemas.enums import FsmState, GateReason, Verdict
from schemas.state import FrameMetrics, SessionState


class IngestPhotoRequest(SchemaModel):
    """Submit one reference photo. Builds a :class:`~schemas.state.FaceProfile`."""

    image_b64: str
    idem_key: str


class IngestPhotoResponse(SchemaModel):
    """Gate outcome for the submitted photo.

    ``accepted`` is false on a below-floor gate — the caller should ask the user
    to re-shoot rather than start a session on this ``face_key``.
    """

    face_key: str
    accepted: bool
    gate_verdict: Verdict
    gate_reason: GateReason
    metrics: FrameMetrics


class StartSessionRequest(SchemaModel):
    """Open a session for an accepted ``face_key``.

    ``brief`` is the user's own free-text request; ``budget_limit`` is the USD
    ceiling for this session (pay-as-you-go). The core always works in edit mode:
    it reads the brief, extracts changes, and applies them to the reference photo.
    The preset falls back to the library's reserved ``default`` when no curated
    preset matches the brief.
    """

    schema_v: int = 2
    face_key: str
    brief: str | None = None  # the user's own free-text request
    budget_limit: Decimal  # USD ceiling for this session (pay-as-you-go)
    idem_key: str


class PresetAsk(StrictModel):
    """The clarifying question a preset asks (its ``ask:true`` slot).

    ``options`` lists enum choices, ``None`` for a free-form slot (the fallback
    preset's scene). ``default`` is the slot's default when it declares one.
    """

    slot: str
    options: list[str] | None = None
    default: str | None = None


class PresetSummary(StrictModel):
    """One preset as the catalog advertises it — no frozen prompt content.

    ``use_case_tokens`` are the matcher tokens the business service passes back as
    ``use_case``; the reserved ``default`` fallback is flagged (``is_fallback``)
    and is not offered as a choice — it is the implicit fall-through.
    """

    id: str
    version: str
    label: str | None = None
    use_case_tokens: list[str]
    is_fallback: bool = False
    asks: list[PresetAsk] = Field(default_factory=list)


class PresetCatalog(SchemaModel):
    """The ``GET /v1/presets`` response: the curated catalog plus library version."""

    presets: list[PresetSummary]
    library_version: str


class StartSessionResponse(SchemaModel):
    """Session opened. ``matched`` is false (and ``fsm_state`` failed) when no
    preset admits the request."""

    session_key: str
    fsm_state: FsmState
    matched: bool
    preset_id: str | None = None


class InputAnswerRequest(SchemaModel):
    """Answer the clarifying question — fills the preset's ``ask:true`` slot."""

    session_key: str
    slot: str
    value: str
    idem_key: str


class ApproveRequest(SchemaModel):
    """Approve (or reject) the plan and optionally pick a composition.

    ``approved=false`` ends the session without spending budget.
    """

    session_key: str
    approved: bool
    idem_key: str
    composition_id: str | None = None


class SessionAck(SchemaModel):
    """Lean acknowledgement of an accepted mutation — the new state, not the
    full snapshot.

    A mutation that does not apply never produces an ack — it is a 404/409
    (see the module docstring) — so ``accepted`` stays true on the wire; the
    field survives for callers that branch on it rather than on the status.
    """

    session_key: str
    fsm_state: FsmState
    accepted: bool = True
    message: str | None = None


class SessionSnapshot(SchemaModel):
    """Full read model of a session.

    Wraps :class:`~schemas.state.SessionState` (safe to expose — the biometric
    embedding lives on the separately keyed ``FaceProfile``) plus the derived
    spend the caller would otherwise compute from ``iterations``:
    ``generations_spent`` (charged frames) and ``cost_spent`` (real dollars
    billed, generations plus auxiliary LLM calls).
    """

    state: SessionState
    generations_spent: int
    cost_spent: Decimal = Decimal("0")
