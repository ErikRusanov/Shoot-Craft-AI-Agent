"""Vision — turns one input photo into a :class:`~schemas.state.FaceProfile`.

The ingest front of the pipeline, before anything costs money: detect faces
through the :class:`~protocols.face_analyzer.FaceAnalyzer` port, measure the
frame (sharpness, exposure, pose, face size — pure pixel math from
``utils.images``), run the quality gate, and assemble the profile. The caller
stores the result under ``face_key`` with a TTL; later sessions reuse the
stored profile instead of re-running detection.

Blur and brightness are measured on the *face crop* (with margin), not the
whole frame — a sharp face in front of a creamy bokeh background must pass, a
sharp background behind a motion-blurred face must not. With no face detected
the metrics fall back to the full frame: the gate fails on ``NO_FACE`` anyway,
the numbers are only for observability.
"""

from __future__ import annotations

from PIL import Image

from protocols import DetectedFace, FaceAnalyzer
from schemas import FaceProfile, FrameMetrics
from services.quality_gate import QualityGate
from utils import images


class VisionService:
    """Builds the face profile for an input photo: metrics, gate verdict, identity."""

    def __init__(self, analyzer: FaceAnalyzer, gate: QualityGate) -> None:
        self._analyzer = analyzer
        self._gate = gate

    async def build_face_profile(
        self, image: bytes, *, face_key: str, photo_ref: str
    ) -> FaceProfile:
        """Analyze ``image`` and return the profile to store under ``face_key``.

        Always returns a profile — a failed gate is encoded in
        ``gate_verdict`` / ``gate_reason``, not raised — so the caller can tell
        the business service exactly what to ask the user to re-shoot.
        """
        frame = images.decode_rgb(image)
        faces = await self._analyzer.analyze(image)
        primary = faces[0] if faces else None  # analyzer contract: largest first

        metrics = self._measure(frame, primary, face_count=len(faces))
        gate = self._gate.evaluate(metrics)

        return FaceProfile(
            face_key=face_key,
            embedding=primary.embedding.tolist() if primary else [],
            gate_verdict=gate.verdict,
            gate_reason=gate.reason,
            metrics=metrics,
            gender=primary.gender if primary else None,
            age=primary.age if primary else None,
            photo_ref=photo_ref,
        )

    def _measure(
        self, frame: Image.Image, primary: DetectedFace | None, *, face_count: int
    ) -> FrameMetrics:
        if primary is not None:
            x1, y1, x2, y2 = primary.bbox
            # Clamp before computing the ratio: detectors may overshoot the frame.
            area = max(0.0, min(x2, frame.width) - max(x1, 0.0)) * max(
                0.0, min(y2, frame.height) - max(y1, 0.0)
            )
            face_area_ratio = area / (frame.width * frame.height)
            region = images.crop_bbox(frame, primary.bbox, margin=0.25)
        else:
            face_area_ratio = 0.0
            region = frame

        gray = images.grayscale(region)
        return FrameMetrics(
            face_count=face_count,
            face_area_ratio=face_area_ratio,
            blur_var=images.laplacian_variance(gray),
            yaw=primary.yaw if primary else 0.0,
            pitch=primary.pitch if primary else 0.0,
            roll=primary.roll if primary else 0.0,
            brightness=images.mean_brightness(gray),
            width=frame.width,
            height=frame.height,
        )
