"""OpenRouter-backed :class:`~protocols.generator.ImageGenerator` (Nano Banana 2).

One ``chat/completions`` call per attempt: text prompt first (OpenRouter's
recommended part order), then the reference photo(s), then the face crop as its
own part at the preset's ``face_media_resolution``. OpenRouter exposes Gemini's
per-part media resolution through the OpenAI-style ``detail`` field on an image
part — there is no native ``media_resolution`` in its schema — so the preset
value maps onto ``detail`` verbatim and must stay within that vocabulary.

The model is a reference-conditioned edit with no denoise/strength: the only
knobs are the preset's :class:`~schemas.presets.Generation` block, mapped to
``temperature`` and ``image_config.aspect_ratio``.

A 200 response **without** an image is deliberately not retried here: unlike a
network failure or 5xx, the upstream may already have charged for it (e.g. a
text-only refusal), and a verbatim retry would pay again for the same outcome —
the generation loop owns that decision.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Sequence
from typing import Any

from protocols.generator import GeneratedImage, GenerationRefusedError
from schemas import Generation
from services.connectors.openrouter_client import OpenRouterClient, parse_usage

# OpenRouter's per-part detail vocabulary; the preset's face_media_resolution
# must be one of these or the request is a config bug, failed before any spend.
_DETAIL_LEVELS = frozenset({"auto", "low", "high"})

_MAGIC_MIME: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
)


class NoImageGeneratedError(GenerationRefusedError):
    """Upstream answered 2xx but returned no usable image.

    Possibly a refusal, possibly already paid — never auto-retried (see module
    docstring); the caller decides whether another attempt is worth its price.
    A :class:`~protocols.generator.GenerationRefusedError`, so it carries the
    billed ``usage`` and the loop settles its spend rather than refunding.
    """


def _sniff_mime(image: bytes) -> str:
    if image[:4] == b"RIFF" and image[8:12] == b"WEBP":
        return "image/webp"
    for magic, mime in _MAGIC_MIME:
        if image[: len(magic)] == magic:
            return mime
    # The pipeline encodes references as JPEG (utils.images.encode_jpeg);
    # unrecognized bytes default to that and the upstream rejects true garbage.
    return "image/jpeg"


def _image_part(image: bytes, *, detail: str | None = None) -> dict[str, Any]:
    image_url: dict[str, Any] = {
        "url": f"data:{_sniff_mime(image)};base64,{base64.b64encode(image).decode('ascii')}"
    }
    if detail is not None:
        image_url["detail"] = detail
    return {"type": "image_url", "image_url": image_url}


def _extract_image(body: dict[str, Any]) -> bytes:
    """Pull the generated image out of ``choices[0].message.images[0]``.

    Any malformed shape collapses into :class:`NoImageGeneratedError` — for the
    caller "no usable image came back" is one condition, however it manifested.
    """
    try:
        url: str = body["choices"][0]["message"]["images"][0]["image_url"]["url"]
        _, _, encoded = url.partition("base64,")
        image = base64.b64decode(encoded, validate=True) if encoded else b""
    except (KeyError, IndexError, TypeError, binascii.Error) as exc:
        raise NoImageGeneratedError(f"no decodable image in the response: {exc!r}") from exc
    if not image:
        raise NoImageGeneratedError("the response carried an empty or non-base64 image part")
    return image


class OpenRouterImageGenerator:
    """Nano Banana 2 (Gemini flash image) behind the ImageGenerator port."""

    def __init__(self, client: OpenRouterClient, *, model: str) -> None:
        self._client = client
        self._model = model

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
        if params.face_media_resolution not in _DETAIL_LEVELS:
            raise ValueError(
                f"face_media_resolution {params.face_media_resolution!r} is not one of "
                f"{sorted(_DETAIL_LEVELS)} — OpenRouter's per-part detail vocabulary"
            )

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        content += [_image_part(ref) for ref in reference_images]
        if face_crop is not None:
            content.append(_image_part(face_crop, detail=params.face_media_resolution))

        body = await self._client.chat_completion(
            {
                "model": self._model,
                "messages": [{"role": "user", "content": content}],
                "modalities": ["image", "text"],
                "temperature": params.temperature,
                "image_config": {"aspect_ratio": params.aspect_ratio},
            }
        )

        # Parse usage first: a no-image response is still billed, and the refusal
        # error must carry the cost so the loop settles rather than refunds.
        usage = parse_usage(body)
        try:
            image = _extract_image(body)
            request_id = str(body.get("id") or "")
            if not request_id:
                # An image we cannot trace back upstream is unusable: Iteration
                # records the provider id, and keep-best must stay auditable.
                raise NoImageGeneratedError("the response carried an image but no request id")
        except NoImageGeneratedError as exc:
            exc.usage = usage
            raise
        return GeneratedImage(image, request_id, usage)
