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

The gate judges only how well the face is *rendered* — resolution, sharpness,
exposure — never how the shot is composed. A turned head, a laugh, a tilted
camera are the user's photo, not a defect: generation is reference-conditioned
and the identity embedding tolerates pose. Pose angles stay in the metrics for
observability but are deliberately not thresholded here.
"""

from __future__ import annotations

from dataclasses import dataclass

from schemas import FrameMetrics, GateReason, Verdict


@dataclass(frozen=True, slots=True)
class GateThresholds:
    """Config-supplied limits; no defaults so wiring stays explicit."""

    min_side: int  # min(width, height) of the frame, px
    min_face_side: float  # min side of the primary face bbox, px
    max_secondary_face_ratio: float  # second-largest face area / primary face area
    min_blur_var: float  # Laplacian variance on the *denoised* face crop
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
        # Ambiguity about *whose* identity to anchor exists only when a second
        # face is comparable in size to the primary — small background
        # passers-by are a fact of real photos and must not fail the gate.
        if m.face_count > 1 and m.secondary_face_ratio > t.max_secondary_face_ratio:
            return GateReason.MULTIPLE_FACES
        if min(m.width, m.height) < t.min_side:
            return GateReason.LOW_RESOLUTION
        # Absolute crop resolution, not fraction of the frame: identity quality
        # comes from the pixels on the face, however the shot is composed.
        if m.face_side < t.min_face_side:
            return GateReason.FACE_TOO_SMALL
        if m.blur_var < t.min_blur_var:
            return GateReason.BLURRY
        if not t.min_brightness <= m.brightness <= t.max_brightness:
            return GateReason.POOR_LIGHTING
        return None
