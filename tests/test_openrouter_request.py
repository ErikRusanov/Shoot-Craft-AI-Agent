"""OpenRouterImageGenerator — request shape and retry/payment semantics.

All on mocked httpx. The two invariants money depends on:

- a retry happens **only** when no result was delivered (network/429/5xx), so
  it can never double-pay;
- a 2xx — with or without an image — is never re-sent by the connector: a
  no-image 200 may already be paid and repeating it is the loop's decision.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx
import pytest

from schemas import Generation
from services.connectors import NoImageGeneratedError, OpenRouterImageGenerator
from tests.openrouter_mock import (
    API_KEY,
    PIXEL_PNG,
    ScriptedTransport,
    image_completion_body,
    scripted_client,
    text_completion_body,
)
from utils.retry import TransientUpstreamError, UpstreamRequestError

MODEL = "google/gemini-3.1-flash-image-preview"
PROMPT = "a studio headshot"
REFERENCE = PIXEL_PNG
FACE_CROP = b"\xff\xd8\xff\xe0fake-jpeg-crop"


@pytest.fixture
def params() -> Generation:
    return Generation(temperature=0.15, aspect_ratio="4:5", face_media_resolution="high")


def _generator(
    *script: httpx.Response | Exception, attempts: int = 4
) -> tuple[OpenRouterImageGenerator, ScriptedTransport]:
    client, transport = scripted_client(*script, attempts=attempts)
    return OpenRouterImageGenerator(client, model=MODEL), transport


def _ok() -> httpx.Response:
    return httpx.Response(200, json=image_completion_body(request_id="gen-abc"))


def _sent_body(transport: ScriptedTransport, index: int = -1) -> dict[str, Any]:
    body: dict[str, Any] = json.loads(transport.requests[index].content)
    return body


async def test_request_shape(params: Generation) -> None:
    generator, transport = _generator(_ok())

    await generator.generate(
        prompt=PROMPT,
        params=params,
        reference_images=[REFERENCE, REFERENCE],
        face_crop=FACE_CROP,
    )

    request = transport.requests[0]
    assert request.method == "POST"
    assert request.url.path.endswith("/chat/completions")
    assert request.headers["Authorization"] == f"Bearer {API_KEY}"

    body = _sent_body(transport)
    assert body["model"] == MODEL
    assert body["modalities"] == ["image", "text"]
    assert body["temperature"] == params.temperature
    assert body["image_config"] == {"aspect_ratio": "4:5"}

    content = body["messages"][0]["content"]
    # Text first (OpenRouter's recommended order), then refs, face crop last.
    assert content[0] == {"type": "text", "text": PROMPT}
    assert [part["type"] for part in content[1:]] == ["image_url"] * 3


async def test_face_part_carries_detail_references_do_not(params: Generation) -> None:
    generator, transport = _generator(_ok())

    await generator.generate(
        prompt=PROMPT, params=params, reference_images=[REFERENCE], face_crop=FACE_CROP
    )

    content = _sent_body(transport)["messages"][0]["content"]
    reference_part, face_part = content[1], content[2]
    # The preset's face_media_resolution maps onto OpenRouter's per-part
    # `detail` field, on the face crop only.
    assert face_part["image_url"]["detail"] == "high"
    assert "detail" not in reference_part["image_url"]


async def test_reference_encoded_as_data_url_with_sniffed_mime(params: Generation) -> None:
    generator, transport = _generator(_ok())

    await generator.generate(
        prompt=PROMPT, params=params, reference_images=[REFERENCE], face_crop=FACE_CROP
    )

    content = _sent_body(transport)["messages"][0]["content"]
    ref_url: str = content[1]["image_url"]["url"]
    assert ref_url.startswith("data:image/png;base64,")
    assert base64.b64decode(ref_url.split(",", 1)[1]) == REFERENCE
    # The crop's JPEG magic bytes are sniffed too.
    assert content[2]["image_url"]["url"].startswith("data:image/jpeg;base64,")


async def test_unknown_bytes_default_to_jpeg(params: Generation) -> None:
    generator, transport = _generator(_ok())

    await generator.generate(prompt=PROMPT, params=params, reference_images=[b"opaque-bytes"])

    content = _sent_body(transport)["messages"][0]["content"]
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


async def test_no_face_crop_sends_no_extra_part(params: Generation) -> None:
    generator, transport = _generator(_ok())

    await generator.generate(prompt=PROMPT, params=params, reference_images=[REFERENCE])

    content = _sent_body(transport)["messages"][0]["content"]
    assert len(content) == 2  # text + one reference, nothing else


async def test_parses_base64_image_and_request_id(params: Generation) -> None:
    generator, _ = _generator(_ok())

    image_bytes, request_id = await generator.generate(
        prompt=PROMPT, params=params, reference_images=[REFERENCE]
    )

    assert image_bytes == PIXEL_PNG
    assert request_id == "gen-abc"


async def test_5xx_then_success_retries_and_delivers_once(params: Generation) -> None:
    generator, transport = _generator(
        httpx.Response(500, text="boom"),
        httpx.Response(503, text="busy"),
        _ok(),
    )

    result = await generator.generate(prompt=PROMPT, params=params, reference_images=[REFERENCE])

    # Three requests went out, but only one ever produced a (paid) image —
    # the failed attempts never returned a result to pay for.
    assert len(transport.requests) == 3
    assert result.image_bytes == PIXEL_PNG


async def test_network_error_then_success_retries(params: Generation) -> None:
    generator, transport = _generator(httpx.ConnectError("connection dropped"), _ok())

    result = await generator.generate(prompt=PROMPT, params=params, reference_images=[REFERENCE])

    assert len(transport.requests) == 2
    assert result.provider_request_id == "gen-abc"


async def test_persistent_5xx_raises_transient_after_attempts(params: Generation) -> None:
    generator, transport = _generator(httpx.Response(502, text="bad gateway"), attempts=3)

    with pytest.raises(TransientUpstreamError):
        await generator.generate(prompt=PROMPT, params=params, reference_images=[REFERENCE])

    assert len(transport.requests) == 3  # bounded by the attempt budget


async def test_4xx_fails_immediately_without_retry(params: Generation) -> None:
    generator, transport = _generator(httpx.Response(400, text="bad request"))

    with pytest.raises(UpstreamRequestError):
        await generator.generate(prompt=PROMPT, params=params, reference_images=[REFERENCE])

    assert len(transport.requests) == 1


async def test_success_without_image_raises_and_never_resends(params: Generation) -> None:
    # A 200 with no image may be a refusal that is already charged; the
    # connector must surface it, not silently pay again for a repeat.
    generator, transport = _generator(httpx.Response(200, json=text_completion_body("sorry, no")))

    with pytest.raises(NoImageGeneratedError):
        await generator.generate(prompt=PROMPT, params=params, reference_images=[REFERENCE])

    assert len(transport.requests) == 1


async def test_empty_references_rejected_before_any_call(params: Generation) -> None:
    generator, transport = _generator(_ok())

    with pytest.raises(ValueError, match="reference"):
        await generator.generate(prompt=PROMPT, params=params, reference_images=[])

    assert transport.requests == []


async def test_unsupported_media_resolution_rejected_before_any_call() -> None:
    generator, transport = _generator(_ok())
    params = Generation(temperature=0.15, aspect_ratio="1:1", face_media_resolution="ultra")

    with pytest.raises(ValueError, match="face_media_resolution"):
        await generator.generate(prompt=PROMPT, params=params, reference_images=[REFERENCE])

    assert transport.requests == []
