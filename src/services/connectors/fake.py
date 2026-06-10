"""Dev-mode fake connectors — the ``FAKE_CONNECTORS=true`` wiring.

Deterministic in-process stand-ins for the three model-shaped ports (image
generator, face analyzer/embedder, slot filler is just the deterministic
service), so the **full** pipeline — gate, interrupts, planning, budget,
generation loop, face-check — runs with no OpenRouter key, no InsightFace
weights and no money spent.

These are *not* test doubles: the scripted/recording fakes for unit tests live
in ``tests/fakes``. These two are honest dev connectors — the analyzer measures
the real pixels of the real photo (so the quality gate behaves truthfully) and
only the *identity* is faked: every image embeds to one constant vector, which
makes face-check converge on the first attempt.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from protocols import DetectedFace, GeneratedImage
from schemas import Generation
from utils import images

# buffalo_l's embedding dimension, so swapping fake ↔ real changes nothing else.
_EMBED_DIM = 512


def _constant_identity(dim: int = _EMBED_DIM) -> NDArray[np.float32]:
    vec = np.zeros(dim, dtype=np.float32)
    vec[0] = 1.0
    return vec


class FakeFaceEngine:
    """``FaceAnalyzer`` + ``Embedder`` with one constant identity.

    ``analyze`` reports a single face covering the central half of the frame —
    frame metrics (sharpness, exposure, resolution) are still measured from the
    actual pixels, so the gate stays honest. ``embed`` returns the same unit
    vector for any image: every generated frame is "the same person" (cosine
    1.0), which is exactly what a converging dev run should look like.
    """

    def __init__(self) -> None:
        self._identity = _constant_identity()

    async def analyze(self, image: bytes) -> list[DetectedFace]:
        frame = images.decode_rgb(image)
        side = min(frame.width, frame.height) / 2
        cx, cy = frame.width / 2, frame.height / 2
        bbox = (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
        return [
            DetectedFace(
                bbox=bbox,
                det_score=0.99,
                yaw=0.0,
                pitch=0.0,
                roll=0.0,
                gender=None,
                embedding=self._identity.copy(),
            )
        ]

    async def embed(self, image: bytes) -> NDArray[np.float32]:
        images.decode_rgb(image)  # keep the port's "must be an image" contract
        return self._identity.copy()


def _noise_jpeg(side: int = 64) -> bytes:
    """A fixed-seed noise JPEG — decodable, embeddable, never the same as input."""
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 256, size=(side, side, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(pixels, mode="RGB").save(buf, format="JPEG")
    return buf.getvalue()


class FakeImageGenerator:
    """Returns one fixed frame per call; the request id is the prompt's hash.

    Free and instant, but shaped exactly like the real connector: real
    decodable bytes plus a provider id stable across retries of the same prompt.
    """

    def __init__(self) -> None:
        self._image = _noise_jpeg()

    async def generate(
        self,
        *,
        prompt: str,
        params: Generation,
        reference_images: Sequence[bytes],
        face_crop: bytes | None = None,
    ) -> GeneratedImage:
        if not reference_images:
            raise ValueError("reference-conditioned edit requires at least one reference image")
        rid = "fake-" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        return GeneratedImage(self._image, rid)
