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

from schemas.base import SchemaModel
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

    ``use_case`` / ``gender`` drive preset matching; ``budget_limit``
    is the paid-generation ceiling for this session.
    """

    face_key: str
    use_case: str
    gender: str
    budget_limit: int
    idem_key: str


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
    spend counter the caller would otherwise compute from ``iterations``.
    """

    state: SessionState
    generations_spent: int
