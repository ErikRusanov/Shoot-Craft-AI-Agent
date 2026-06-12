"""Encoding images into OpenRouter ``image_url`` message parts.

Shared by every connector that sends a photo upstream (the image generator, the
inventory extractor). OpenRouter exposes Gemini's per-part media resolution
through the OpenAI-style ``detail`` field — there is no native
``media_resolution`` in its schema.
"""

from __future__ import annotations

import base64
from typing import Any

_MAGIC_MIME: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
)


def sniff_mime(image: bytes) -> str:
    if image[:4] == b"RIFF" and image[8:12] == b"WEBP":
        return "image/webp"
    for magic, mime in _MAGIC_MIME:
        if image[: len(magic)] == magic:
            return mime
    # The pipeline encodes references as JPEG (utils.images.encode_jpeg);
    # unrecognized bytes default to that and the upstream rejects true garbage.
    return "image/jpeg"


def image_part(image: bytes, *, detail: str | None = None) -> dict[str, Any]:
    image_url: dict[str, Any] = {
        "url": f"data:{sniff_mime(image)};base64,{base64.b64encode(image).decode('ascii')}"
    }
    if detail is not None:
        image_url["detail"] = detail
    return {"type": "image_url", "image_url": image_url}
