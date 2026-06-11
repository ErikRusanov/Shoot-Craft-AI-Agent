"""Port: face analyzer — detection plus per-face attributes for the input photo.

Distinct from :class:`~protocols.embedder.Embedder` on purpose: the embedder is
the minimal surface face-check needs per generated frame (one vector), while
ingest needs the full detection picture — how many faces, where, posed how —
to fill :class:`~schemas.state.FrameMetrics`. One concrete model
(InsightFace) implements both ports from a single inference pass; tests swap in
a stub that fabricates :class:`DetectedFace` values.

``DetectedFace.embedding`` is biometric data — same rule as the embedder port:
it lives only on the face profile and is **never** logged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DetectedFace:
    """One detected face: geometry, pose, identity vector.

    ``bbox`` is ``(x1, y1, x2, y2)`` in pixels of the source frame; pose angles
    are degrees. Demographics (age, gender) are deliberately *not* surfaced: the
    age estimate proved wildly wrong on real photos, and matching no longer keys
    on gender — the face comes from the reference, not a preset choice.
    """

    bbox: tuple[float, float, float, float]
    det_score: float
    yaw: float
    pitch: float
    roll: float
    embedding: NDArray[np.float32]  # L2-normalized; biometric, never logged


@runtime_checkable
class FaceAnalyzer(Protocol):
    """Detect every face in a photo and describe each one."""

    async def analyze(self, image: bytes) -> list[DetectedFace]:
        """All detected faces, largest bbox first (``[0]`` is the primary face).

        ``image`` is the encoded photo bytes. Empty list when nothing is
        detected — that is a normal outcome the quality gate turns into
        ``NO_FACE``, not an error.
        """
        ...
