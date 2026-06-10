"""Quality gate — is this input photo good enough to anchor a paid session?

A pure threshold check over :class:`~schemas.state.FrameMetrics`: no I/O, no
model calls, every number it compares against arrives via
:class:`GateThresholds` from config (deps assembles it from ``Settings``), never
hardcoded here.

Three bands, calibrated on labeled real photos:

- ``PASSED`` — a clean anchor, proceed silently.
- ``SOFT`` — usable *at the user's risk*: the metric sits between the hard
  floor and the clean-pass level (a shadowed face, heavy retouching, a small
  but sharp face, an extreme profile). The business service is expected to
  warn that quality cannot be guaranteed and ask for confirmation before
  spending budget on it.
- ``BELOW_FLOOR`` — hopeless as an identity anchor: no face, an ambiguous
  comparable second face, or rendering below the hard floor. Re-shooting is
  the only fix.

The first failing check names the :class:`~schemas.enums.GateReason`, ordered
so the user is told the most fundamental problem first (no face beats blurry).

The gate *rejects* only on how well the face is rendered — resolution,
sharpness, exposure — never on how the shot is composed. Pose is the one
composition signal that can mark a photo ``SOFT``: an extreme profile is a
weak identity anchor (the embedding degrades), so the user is warned — but it
can never reject.
"""

from __future__ import annotations

from dataclasses import dataclass

from schemas import FrameMetrics, GateReason, Verdict


@dataclass(frozen=True, slots=True)
class GateThresholds:
    """Config-supplied limits; no defaults so wiring stays explicit.

    ``min_*``/``max_*`` are the clean-pass levels; ``floor_*`` are the hard
    floors below which the photo is rejected outright. Between the two the
    verdict is ``SOFT`` — try with confirmation.
    """

    min_side: int  # min(width, height) of the frame, px — hard
    max_secondary_face_ratio: float  # second face area / primary area — hard
    min_face_side: float  # min side of the primary face bbox, px
    floor_face_side: float
    min_blur_var: float  # Laplacian variance on the *denoised* face crop
    floor_blur_var: float
    min_brightness: float  # mean luma 0..255 on the face crop
    max_brightness: float
    floor_min_brightness: float
    floor_max_brightness: float
    # Risk-only: beyond this |yaw| the photo is SOFT, never rejected.
    risk_max_abs_yaw: float


@dataclass(frozen=True, slots=True)
class GateResult:
    verdict: Verdict
    reason: GateReason


class QualityGate:
    """Bands an input photo as pass / at-risk / rejected from its metrics."""

    def __init__(self, thresholds: GateThresholds) -> None:
        self._t = thresholds

    def evaluate(self, metrics: FrameMetrics) -> GateResult:
        """First failing check wins; hard floors before risk flags."""
        hard = self._hard_failure(metrics)
        if hard is not None:
            return GateResult(verdict=Verdict.BELOW_FLOOR, reason=hard)
        risk = self._risk(metrics)
        if risk is not None:
            return GateResult(verdict=Verdict.SOFT, reason=risk)
        return GateResult(verdict=Verdict.PASSED, reason=GateReason.OK)

    def _hard_failure(self, m: FrameMetrics) -> GateReason | None:
        t = self._t
        if m.face_count == 0:
            return GateReason.NO_FACE
        # Ambiguity about *whose* identity to anchor exists only when a second
        # face is comparable in size to the primary — small background
        # passers-by are a fact of real photos and must not fail the gate.
        # No risk band here: confirmation cannot resolve which face to use.
        if m.face_count > 1 and m.secondary_face_ratio > t.max_secondary_face_ratio:
            return GateReason.MULTIPLE_FACES
        if min(m.width, m.height) < t.min_side:
            return GateReason.LOW_RESOLUTION
        # Absolute crop resolution, not fraction of the frame: identity quality
        # comes from the pixels on the face, however the shot is composed.
        if m.face_side < t.floor_face_side:
            return GateReason.FACE_TOO_SMALL
        if m.blur_var < t.floor_blur_var:
            return GateReason.BLURRY
        if not t.floor_min_brightness <= m.brightness <= t.floor_max_brightness:
            return GateReason.POOR_LIGHTING
        return None

    def _risk(self, m: FrameMetrics) -> GateReason | None:
        t = self._t
        if m.face_side < t.min_face_side:
            return GateReason.FACE_TOO_SMALL
        if m.blur_var < t.min_blur_var:
            return GateReason.BLURRY
        if not t.min_brightness <= m.brightness <= t.max_brightness:
            return GateReason.POOR_LIGHTING
        if abs(m.yaw) > t.risk_max_abs_yaw:
            return GateReason.EXTREME_POSE
        return None
