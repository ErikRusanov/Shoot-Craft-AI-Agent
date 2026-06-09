"""Port: image generator — the reference-conditioned edit model.

Nano Banana 2 (Gemini flash-image) via OpenRouter is **not** text2img: it edits
toward the supplied reference faces, so a call without at least one reference is
meaningless. There is no denoise/strength knob — the only controls are the
preset's :class:`~schemas.presets.Generation` block, passed through verbatim as
``params`` so the contract for "what knobs exist" lives in one place.

The provider's request id is returned alongside the bytes and recorded on
:class:`~schemas.state.Iteration` so a delivered frame can be traced back to the
exact upstream call.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple, Protocol, runtime_checkable

from schemas import Generation


class GeneratedImage(NamedTuple):
    """One generated frame plus the upstream call id, unpackable as a tuple."""

    image_bytes: bytes
    provider_request_id: str


@runtime_checkable
class ImageGenerator(Protocol):
    """Generate one frame, conditioned on the reference face image(s)."""

    async def generate(
        self,
        *,
        prompt: str,
        params: Generation,
        reference_images: Sequence[bytes],
    ) -> GeneratedImage:
        """Return ``(image_bytes, provider_request_id)`` for one attempt.

        ``reference_images`` holds the encoded reference photo(s) the edit is
        conditioned on; it is expected to be non-empty. ``params`` are the
        frozen preset generation knobs — the caller never invents its own.
        """
        ...
