"""Backpressure on the image-generator port — a per-process concurrency cap.

Parallel sessions all funnel their paid calls through one
:class:`~protocols.generator.ImageGenerator`; without a cap a burst of sessions
stampedes the upstream (rate-limit storms, head-of-line timeouts). The wrapper
is a plain semaphore around the port, so callers queue *at the model call* —
the cheap parts of the loop (face-check, storage, events) stay concurrent.

Wraps any generator (fake or OpenRouter): backpressure is wiring, not a
property of one connector.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from protocols import GeneratedImage, ImageGenerator
from schemas import Generation


class ThrottledImageGenerator:
    """``ImageGenerator`` that admits at most ``max_concurrent`` calls at once."""

    def __init__(self, inner: ImageGenerator, *, max_concurrent: int) -> None:
        self._inner = inner
        self._sem = asyncio.Semaphore(max_concurrent)

    async def generate(
        self,
        *,
        prompt: str,
        params: Generation,
        reference_images: Sequence[bytes],
        face_crop: bytes | None = None,
    ) -> GeneratedImage:
        async with self._sem:
            return await self._inner.generate(
                prompt=prompt,
                params=params,
                reference_images=reference_images,
                face_crop=face_crop,
            )
