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
    pre-session ``rejected`` state here — only the terminal pair below.
    """

    CREATED = "created"
    FACE_CHECK = "face_check"
    NEED_INPUT = "need_input"
    PLANNING = "planning"
    AWAITING_APPROVAL = "awaiting_approval"
    GENERATING = "generating"
    DONE = "done"
    FAILED = "failed"


class Verdict(StrEnum):
    """Outcome of a threshold check against ``similarity_threshold`` /
    ``identity_floor``.

    Used twice with the same shape: the input-photo quality gate and the
    per-iteration identity check. ``soft`` is the band between the floor and the
    target — shippable as keep-best when retries run out, but not a clean pass.
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

    ``ok`` accompanies a pass; the rest explain a soft/below-floor outcome so the
    business service can tell the user what to re-shoot.
    """

    OK = "ok"
    NO_FACE = "no_face"
    MULTIPLE_FACES = "multiple_faces"
    FACE_TOO_SMALL = "face_too_small"
    LOW_RESOLUTION = "low_resolution"
    BLURRY = "blurry"
    OCCLUDED = "occluded"
    EXTREME_POSE = "extreme_pose"
    POOR_LIGHTING = "poor_lighting"
