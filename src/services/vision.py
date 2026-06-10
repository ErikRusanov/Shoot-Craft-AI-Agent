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

        metrics = self._measure(frame, faces)
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

    def _measure(self, frame: Image.Image, faces: list[DetectedFace]) -> FrameMetrics:
        primary = faces[0] if faces else None
        face_area_ratio = 0.0
        face_side = 0.0
        secondary_face_ratio = 0.0
        region = frame
        if primary is not None:
            w, h = self._clamped_size(frame, primary)
            face_area_ratio = (w * h) / (frame.width * frame.height)
            face_side = min(w, h)
            region = images.crop_bbox(frame, primary.bbox, margin=0.25)
            if len(faces) > 1 and w * h > 0:
                w2, h2 = self._clamped_size(frame, faces[1])
                secondary_face_ratio = (w2 * h2) / (w * h)

        gray = images.grayscale(region)
        # Sharpness on the denoised crop (grain reads as "sharp" otherwise);
        # brightness on the raw one (denoising doesn't move the mean, but the
        # exposure number should describe the actual pixels).
        denoised = images.grayscale(images.denoise_median(region))
        return FrameMetrics(
            face_count=len(faces),
            face_area_ratio=face_area_ratio,
            face_side=face_side,
            secondary_face_ratio=secondary_face_ratio,
            blur_var=images.laplacian_variance(denoised),
            yaw=primary.yaw if primary else 0.0,
            pitch=primary.pitch if primary else 0.0,
            roll=primary.roll if primary else 0.0,
            brightness=images.mean_brightness(gray),
            width=frame.width,
            height=frame.height,
        )

    @staticmethod
    def _clamped_size(frame: Image.Image, face: DetectedFace) -> tuple[float, float]:
        """Bbox width/height with the out-of-frame overshoot detectors allow cut off."""
        x1, y1, x2, y2 = face.bbox
        w = max(0.0, min(x2, frame.width) - max(x1, 0.0))
        h = max(0.0, min(y2, frame.height) - max(y1, 0.0))
        return w, h
