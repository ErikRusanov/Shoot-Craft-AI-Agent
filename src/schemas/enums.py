"""Closed vocabularies shared across the contract.

``StrEnum`` so each value serializes to its own string — readable on the wire
and stable in Redis. These are the only places a state, verdict, risk band, or
gate reason is named; services and the graph branch on these, never on raw
strings.
"""

from __future__ import annotations

from enum import StrEnum


class FsmState(StrEnum):
    """Lifecycle of one photoshoot session.

    Linear with one branch: ``created`` → face check → (ask the single
    clarifying question) → plan → approval → generation loop → terminal. Gate
    rejection happens at *ingest*, before a session exists, so there is no
    pre-session ``rejected`` state here — only the terminals below.

    ``cancelled`` is the caller-initiated terminal: distinct from ``failed``
    so the business service can tell "the core gave up" from "we told it to
    stop". It is set outside the graph (the cancel endpoint), never by a node.
    """

    CREATED = "created"
    FACE_CHECK = "face_check"
    NEED_INPUT = "need_input"
    PLANNING = "planning"
    AWAITING_APPROVAL = "awaiting_approval"
    GENERATING = "generating"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Verdict(StrEnum):
    """Outcome of a threshold check against ``similarity_threshold`` /
    ``identity_floor``.

    Used twice with the same shape: the input-photo quality gate and the
    per-iteration identity check. ``soft`` is the band between the floor and
    the target. For a generated frame that means shippable as keep-best when
    retries run out; for an input photo it means usable *with the user's
    confirmation* — quality is not guaranteed, but trying is allowed.
    """

    PASSED = "passed"
    SOFT = "soft"
    BELOW_FLOOR = "below_floor"


class RiskLevel(StrEnum):
    """Confidence band that a generated frame still depicts the same person.

    Distinct from ``Verdict``: a result can pass the similarity target yet carry
    elevated risk (e.g. borderline pose), and the business service may want that
    signal independently of the ship/no-ship verdict.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Gender(StrEnum):
    """Perceived gender of the input face, as estimated by the CV attribute model.

    The values double as preset ``applies_to.gender`` tokens, so matching is a
    plain membership test. It is a generation hint (which presets fit), never an
    identity claim — absent/uncertain stays ``None`` on the profile.
    """

    MALE = "male"
    FEMALE = "female"


class GateReason(StrEnum):
    """Why the input-photo quality gate produced its verdict.

    ``ok`` accompanies a clean pass; the rest explain a soft/below-floor
    outcome so the business service can tell the user what to re-shoot or what
    risk they are confirming. Rendering reasons (resolution, sharpness,
    exposure) can reject; ``extreme_pose`` is the one composition signal and
    it only ever marks the ``soft`` band — a strong profile weakens the
    identity anchor, so the user is warned, never refused.
    """

    OK = "ok"
    NO_FACE = "no_face"
    MULTIPLE_FACES = "multiple_faces"
    FACE_TOO_SMALL = "face_too_small"
    LOW_RESOLUTION = "low_resolution"
    BLURRY = "blurry"
    OCCLUDED = "occluded"
    POOR_LIGHTING = "poor_lighting"
    EXTREME_POSE = "extreme_pose"


class FailureCode(StrEnum):
    """Why a session reached the ``failed`` terminal — the machine-readable axis.

    The business service maps a terminal to a user action (re-shoot, re-ask,
    retry, contact support) on ``code``; the ``FailedEvent.reason`` string is
    human-readable detail only, free to change without breaking that mapping.
    ``gate_reason`` further qualifies ``INPUT_REJECTED`` with the photo signal.
    """

    INPUT_REJECTED = "input_rejected"  # the input photo cannot anchor the identity
    NO_PRESET = "no_preset"  # no preset admits the request and no fallback ships
    SCENE_REJECTED = "scene_rejected"  # free-form scene read as injection, re-asks spent
    PLAN_REJECTED = "plan_rejected"  # the user declined the plan at approval
    BUDGET_EXHAUSTED = "budget_exhausted"  # budget spent before reaching the floor
    NO_DELIVERABLE = "no_deliverable"  # retries spent, no attempt reached the floor
    GENERATION_FAILED = "generation_failed"  # provider/face-check produced no measured frame
    REFERENCE_MISSING = "reference_missing"  # the reference photo is gone from storage
    WALL_CLOCK = "wall_clock"  # the run exceeded the wall-clock limit
    CANCELLED = "cancelled"  # the caller stopped the session
    INTERNAL = "internal"  # an unclassified internal error
