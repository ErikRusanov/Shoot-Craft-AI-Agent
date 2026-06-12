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
from decimal import Decimal

from PIL import Image

from protocols.generator import GeneratedImage
from schemas import Generation, ProviderUsage


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
    # The working image this call edited (reference_images[0]) — lets a chained
    # multi-step test assert step N edited step N-1's result, not the original.
    reference_image: bytes | None = None


class FixedImageGenerator:
    """Returns ``image`` for every call; id is ``fake-<prompt hash>``.

    ``cost`` (when set) is reported as the provider's billed ``usage.cost`` so the
    budget settles to a known value; left ``None``, the meter settles on its
    reserved estimate (the real-provider "no usage" path).
    """

    def __init__(self, image: bytes = _PIXEL_PNG, *, cost: Decimal | None = None) -> None:
        self.image = image
        self._cost = cost
        self.calls: list[GenerateCall] = []

    async def generate(
        self,
        *,
        prompt: str,
        params: Generation,
        reference_images: Sequence[bytes],
        face_crop: bytes | None = None,
    ) -> GeneratedImage:
        self.calls.append(
            GenerateCall(
                prompt,
                params,
                len(reference_images),
                face_crop,
                reference_images[0] if reference_images else None,
            )
        )
        rid = "fake-" + hashlib.sha256(prompt.encode()).hexdigest()[:16]
        usage = None if self._cost is None else ProviderUsage(cost=self._cost)
        return GeneratedImage(self.image, rid, usage)


class FlakyImageGenerator(FixedImageGenerator):
    """Fails the first ``failures`` calls with ``ConnectionError``, then succeeds.

    The stand-in for a provider whose transport gave out after the connector's
    own transient retries — the loop must treat such an attempt as unpaid.
    Failed calls are recorded in ``calls`` like successful ones.
    """

    def __init__(self, *, failures: int, image: bytes = _PIXEL_PNG) -> None:
        super().__init__(image)
        self._failures_left = failures

    async def generate(
        self,
        *,
        prompt: str,
        params: Generation,
        reference_images: Sequence[bytes],
        face_crop: bytes | None = None,
    ) -> GeneratedImage:
        if self._failures_left > 0:
            self._failures_left -= 1
            self.calls.append(
                GenerateCall(
                    prompt,
                    params,
                    len(reference_images),
                    face_crop,
                    reference_images[0] if reference_images else None,
                )
            )
            raise ConnectionError("provider unreachable")
        return await super().generate(
            prompt=prompt,
            params=params,
            reference_images=reference_images,
            face_crop=face_crop,
        )
