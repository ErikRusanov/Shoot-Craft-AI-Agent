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

import asyncio
from typing import NamedTuple

from PIL import Image

from protocols import DetectedFace, FaceAnalyzer
from schemas import FaceProfile, FrameMetrics
from services.quality_gate import QualityGate
from utils import images

# Margin (fraction of the bbox) around the stored reference crop: enough
# context for the generator to read the face, still face-dominated.
_CROP_MARGIN = 0.25


def photo_ref(face_key: str) -> str:
    """Object-storage key convention for the input photo behind ``face_key``."""
    return f"photos/{face_key}"


def face_crop_ref(face_key: str) -> str:
    """Object-storage key convention for the tight face crop behind ``face_key``."""
    return f"faces/{face_key}/crop"


class FaceIngest(NamedTuple):
    """One ingest pass: the profile plus the tight crop of the primary face.

    The crop is produced here because only ingest still holds the detection
    bbox; the generation loop later attaches it to retries to strengthen the
    identity reference. ``None`` when no face was detected.
    """

    profile: FaceProfile
    face_crop: bytes | None


class VisionService:
    """Builds the face profile for an input photo: metrics, gate verdict, identity."""

    def __init__(self, analyzer: FaceAnalyzer, gate: QualityGate) -> None:
        self._analyzer = analyzer
        self._gate = gate

    async def ingest(self, image: bytes, *, face_key: str, photo_ref: str) -> FaceIngest:
        """Analyze ``image``: the profile to store under ``face_key`` plus the face crop.

        Always returns a profile — a failed gate is encoded in
        ``gate_verdict`` / ``gate_reason``, not raised — so the caller can tell
        the business service exactly what to ask the user to re-shoot.
        """
        faces = await self._analyzer.analyze(image)
        primary = faces[0] if faces else None  # analyzer contract: largest first

        # The pixel pass (decode, median denoise, crop) is CPU-bound; off the
        # loop, or parallel ingests serialize the whole API behind it.
        metrics, face_crop = await asyncio.to_thread(self._pixel_pass, image, faces)
        gate = self._gate.evaluate(metrics)

        profile = FaceProfile(
            face_key=face_key,
            embedding=primary.embedding.tolist() if primary else [],
            gate_verdict=gate.verdict,
            gate_reason=gate.reason,
            metrics=metrics,
            gender=primary.gender if primary else None,
            photo_ref=photo_ref,
        )
        return FaceIngest(profile=profile, face_crop=face_crop)

    def _pixel_pass(
        self, image: bytes, faces: list[DetectedFace]
    ) -> tuple[FrameMetrics, bytes | None]:
        frame = images.decode_rgb(image)
        return self._measure(frame, faces), self._crop(frame, faces[0] if faces else None)

    async def build_face_profile(
        self, image: bytes, *, face_key: str, photo_ref: str
    ) -> FaceProfile:
        """The profile alone — see :meth:`ingest` for the full result."""
        ingest = await self.ingest(image, face_key=face_key, photo_ref=photo_ref)
        return ingest.profile

    @staticmethod
    def _crop(frame: Image.Image, primary: DetectedFace | None) -> bytes | None:
        if primary is None:
            return None
        try:
            return images.encode_jpeg(images.crop_bbox(frame, primary.bbox, margin=_CROP_MARGIN))
        except ValueError:
            # A degenerate bbox (fully out of frame) — no crop, never a crash:
            # the crop only strengthens retries, the pipeline runs without it.
            return None

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
