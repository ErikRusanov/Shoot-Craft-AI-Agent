"""Image generator fake — always returns the same valid PNG.

Hands back a tiny but genuinely decodable PNG (so callers that re-open or
re-embed the result work unchanged) and a request id derived from the prompt, so
the id is stable across runs. Records every call for assertions on what the loop
actually asked for.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Sequence
from dataclasses import dataclass

from PIL import Image

from protocols.generator import GeneratedImage
from schemas import Generation


def _pixel_png() -> bytes:
    """A 1x1 PNG — smallest genuinely decodable image, built once at import."""
    buf = io.BytesIO()
    Image.new("RGB", (1, 1), (127, 127, 127)).save(buf, format="PNG")
    return buf.getvalue()


_PIXEL_PNG = _pixel_png()


@dataclass
class GenerateCall:
    """One recorded :meth:`FixedImageGenerator.generate` invocation."""

    prompt: str
    params: Generation
    reference_count: int
    face_crop: bytes | None = None


class FixedImageGenerator:
    """Returns ``image`` for every call; id is ``fake-<prompt hash>``."""

    def __init__(self, image: bytes = _PIXEL_PNG) -> None:
        self.image = image
        self.calls: list[GenerateCall] = []

    async def generate(
        self,
        *,
        prompt: str,
        params: Generation,
        reference_images: Sequence[bytes],
        face_crop: bytes | None = None,
    ) -> GeneratedImage:
        self.calls.append(GenerateCall(prompt, params, len(reference_images), face_crop))
        rid = "fake-" + hashlib.sha256(prompt.encode()).hexdigest()[:16]
        return GeneratedImage(self.image, rid)
