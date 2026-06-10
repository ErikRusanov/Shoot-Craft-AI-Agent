"""Quality gate — is this input photo good enough to anchor a paid session?

A pure threshold check over :class:`~schemas.state.FrameMetrics`: no I/O, no
model calls, every number it compares against arrives via
:class:`GateThresholds` from config (deps assembles it from ``Settings``), never
hardcoded here.

The gate is binary — ``PASSED`` or ``BELOW_FLOOR`` — with no ``SOFT`` band on
purpose: a borderline *input* photo costs the user nothing to re-shoot, while
letting it through risks burning the whole generation budget on a weak anchor.
``SOFT`` exists for generated frames (keep-best), not here. The first failing
check names the :class:`~schemas.enums.GateReason`, ordered so the user is told
the most fundamental problem first (no face beats blurry).
"""

from __future__ import annotations

from dataclasses import dataclass

from schemas import FrameMetrics, GateReason, Verdict


@dataclass(frozen=True, slots=True)
class GateThresholds:
    """Config-supplied limits; no defaults so wiring stays explicit."""

    min_side: int  # min(width, height) of the frame, px
    min_face_area_ratio: float  # face bbox area / frame area
    min_blur_var: float  # Laplacian variance on the face crop
    max_yaw: float  # degrees, absolute
    max_pitch: float  # degrees, absolute
    max_roll: float  # degrees, absolute
    min_brightness: float  # mean luma 0..255 on the face crop
    max_brightness: float


@dataclass(frozen=True, slots=True)
class GateResult:
    verdict: Verdict
    reason: GateReason


class QualityGate:
    """Decides pass/fail for an input photo from its measured metrics."""

    def __init__(self, thresholds: GateThresholds) -> None:
        self._t = thresholds

    def evaluate(self, metrics: FrameMetrics) -> GateResult:
        """First failing check wins; all checks green means ``PASSED``/``OK``."""
        reason = self._first_failure(metrics)
        if reason is None:
            return GateResult(verdict=Verdict.PASSED, reason=GateReason.OK)
        return GateResult(verdict=Verdict.BELOW_FLOOR, reason=reason)

    def _first_failure(self, m: FrameMetrics) -> GateReason | None:
        t = self._t
        if m.face_count == 0:
            return GateReason.NO_FACE
        # Any extra detection is ambiguity about *whose* identity to anchor, so
        # one face is the rule — the detector's own confidence cut already
        # filtered background noise.
        if m.face_count > 1:
            return GateReason.MULTIPLE_FACES
        if min(m.width, m.height) < t.min_side:
            return GateReason.LOW_RESOLUTION
        if m.face_area_ratio < t.min_face_area_ratio:
            return GateReason.FACE_TOO_SMALL
        if m.blur_var < t.min_blur_var:
            return GateReason.BLURRY
        if abs(m.yaw) > t.max_yaw or abs(m.pitch) > t.max_pitch or abs(m.roll) > t.max_roll:
            return GateReason.EXTREME_POSE
        if not t.min_brightness <= m.brightness <= t.max_brightness:
            return GateReason.POOR_LIGHTING
        return None
