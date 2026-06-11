"""Port: image generator — the reference-conditioned edit model.

Nano Banana 2 (Gemini flash-image) via OpenRouter is **not** text2img: it edits
toward the supplied reference faces, so a call without at least one reference is
meaningless. There is no denoise/strength knob — the only controls are the
preset's :class:`~schemas.presets.Generation` block, passed through verbatim as
``params`` so the contract for "what knobs exist" lives in one place.

The provider's request id is returned alongside the bytes and recorded on
:class:`~schemas.state.Iteration` so a delivered frame can be traced back to the
exact upstream call. ``usage`` carries what the provider billed, so the budget
can settle the reservation to the real cost.

The port defines exactly one error: :class:`GenerationRefusedError` — a paid
response that carried no usable image (a text-only refusal may still be billed).
The budget settles its spend. Every other failure (transport, 4xx) is a plain
exception the loop refunds, so "paid but no image" and "never charged" stay
distinguishable without the loop importing a concrete connector.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple, Protocol, runtime_checkable

from schemas import Generation, ProviderUsage


class GeneratedImage(NamedTuple):
    """One generated frame, the upstream call id, and what it billed."""

    image_bytes: bytes
    provider_request_id: str
    usage: ProviderUsage | None = None


class GenerationRefusedError(RuntimeError):
    """A 2xx response that returned no usable image — possibly still billed.

    Carries the provider's ``usage`` so the budget settles the real spend
    instead of refunding. Distinct from transport/4xx failures (plain
    exceptions): those never charged and the loop refunds the reservation.
    """

    def __init__(self, detail: str, *, usage: ProviderUsage | None = None) -> None:
        super().__init__(detail)
        self.usage = usage


@runtime_checkable
class ImageGenerator(Protocol):
    """Generate one frame, conditioned on the reference face image(s)."""

    async def generate(
        self,
        *,
        prompt: str,
        params: Generation,
        reference_images: Sequence[bytes],
        face_crop: bytes | None = None,
    ) -> GeneratedImage:
        """Return ``(image_bytes, provider_request_id)`` for one attempt.

        ``reference_images`` holds the encoded reference photo(s) the edit is
        conditioned on; it is expected to be non-empty. ``face_crop`` is the
        tight crop of the anchor face, sent as its own part at the preset's
        ``face_media_resolution`` so identity is read at full detail — explicit
        here rather than smuggled into ``reference_images`` by position.
        ``params`` are the frozen preset generation knobs — the caller never
        invents its own.
        """
        ...
