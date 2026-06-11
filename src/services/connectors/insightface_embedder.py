"""InsightFace connector — one model pack behind two ports.

Implements both :class:`~protocols.embedder.Embedder` and
:class:`~protocols.face_analyzer.FaceAnalyzer` from a single inference pass:
detection, pose, gender/age and the 512-d identity embedding all come from the
same ``FaceAnalysis`` pack (default ``buffalo_l``), CPU ``onnxruntime``.

Weights are **not** committed and are **not** auto-downloaded at startup —
`make models` fetches the pack into ``insightface_root`` explicitly, and the
constructor fails fast with that instruction when the directory is missing.
A silent multi-hundred-MB download inside a prod boot is the failure mode this
guards against.

``insightface`` is imported lazily inside the constructor: the import drags in
onnxruntime + OpenCV (seconds of import time), and merely importing this module
(e.g. by test collection that then skips) must stay cheap.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps

from protocols import DetectedFace
from utils import images


def model_pack_dir(root: str, model_pack: str) -> Path:
    """Where ``FaceAnalysis`` expects the pack's ``*.onnx`` files."""
    return Path(root).expanduser() / "models" / model_pack


def weights_present(root: str, model_pack: str) -> bool:
    """True when the pack is downloaded; tests use this to skip cleanly."""
    return any(model_pack_dir(root, model_pack).glob("*.onnx"))


class InsightFaceEmbedder:
    """CPU InsightFace at the ``Embedder`` + ``FaceAnalyzer`` ports."""

    def __init__(self, *, model_pack: str, root: str, det_size: int = 640) -> None:
        if not weights_present(root, model_pack):
            raise RuntimeError(
                f"InsightFace pack '{model_pack}' not found at {model_pack_dir(root, model_pack)}"
                " — run `make models` to download it (weights are never fetched implicitly)"
            )
        from insightface.app import FaceAnalysis

        self._app: Any = FaceAnalysis(
            name=model_pack, root=root, providers=["CPUExecutionProvider"]
        )
        # ctx_id is a GPU index; ignored on the CPU provider.
        self._app.prepare(ctx_id=0, det_size=(det_size, det_size))

    async def analyze(self, image: bytes) -> list[DetectedFace]:
        # onnxruntime inference holds the GIL only in spurts but takes hundreds
        # of ms — off the event loop it goes.
        return await asyncio.to_thread(self._analyze_sync, image)

    async def embed(self, image: bytes) -> NDArray[np.float32]:
        faces = await self.analyze(image)
        if not faces:
            raise ValueError("no face detected — cannot embed")
        return faces[0].embedding

    # Border (fraction of the longer side) for the close-up detection retry.
    _PAD_FRACTION = 0.25

    def _analyze_sync(self, image: bytes) -> list[DetectedFace]:
        frame = images.decode_rgb(image)
        faces = self._detect(frame)
        if not faces:
            # SCRFD has no anchors for a face larger than the frame, so an
            # extreme close-up (selfie filling the whole shot) detects as
            # "no face". A neutral border shrinks the face relative to the
            # canvas; bboxes are mapped back to frame coordinates. Doubles
            # inference only on the otherwise-rejected zero-face path.
            pad = round(max(frame.size) * self._PAD_FRACTION)
            padded = ImageOps.expand(frame, border=pad, fill=(127, 127, 127))
            faces = self._detect(padded, offset=pad)
        return faces

    def _detect(self, frame: Image.Image, *, offset: int = 0) -> list[DetectedFace]:
        rgb = np.asarray(frame)
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])  # insightface is cv2-conventioned
        detected = self._app.get(bgr)

        faces = [self._to_face(f, offset=offset) for f in detected]
        # Port contract: largest bbox first, so [0] is the primary face.
        faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        return faces

    @staticmethod
    def _to_face(face: Any, *, offset: int = 0) -> DetectedFace:
        # A bbox found on a padded canvas may overhang the original frame
        # after the shift — metrics clamp it (vision) and crops clamp it
        # (utils.images), so it is kept as-is here.
        x1, y1, x2, y2 = (float(v) - offset for v in face.bbox)
        # landmark_3d_68 sets pose as [pitch, yaw, roll] degrees; it may be
        # absent from a slim pack. The pack's age/gender estimates are dropped at
        # this boundary — see DetectedFace.
        pose = face.get("pose")
        pitch, yaw, roll = (float(v) for v in pose) if pose is not None else (0.0, 0.0, 0.0)
        return DetectedFace(
            bbox=(x1, y1, x2, y2),
            det_score=float(face.det_score),
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            embedding=np.asarray(face.normed_embedding, dtype=np.float32),
        )
