"""Contract: :class:`~protocols.generator.ImageGenerator`.

Pins the shape every generator must honor: decodable image bytes plus a
non-empty provider request id, unpackable as a tuple, conditioned on the
reference image(s) it is handed.
"""

from __future__ import annotations

import io
from collections.abc import Callable

import httpx
import pytest
from PIL import Image

from protocols import GeneratedImage, ImageGenerator
from schemas import Generation
from services.connectors import OpenRouterImageGenerator
from tests.fakes import FixedImageGenerator
from tests.openrouter_mock import image_completion_body, scripted_client


def _openrouter_generator() -> ImageGenerator:
    """The real connector over mocked httpx — same contract, no network."""
    client, _ = scripted_client(httpx.Response(200, json=image_completion_body()))
    return OpenRouterImageGenerator(client, model="google/gemini-3.1-flash-image-preview")


GENERATOR_FACTORIES = [
    pytest.param(FixedImageGenerator, id="fixed"),
    pytest.param(_openrouter_generator, id="openrouter-mocked"),
]


@pytest.fixture(params=GENERATOR_FACTORIES)
def generator(request: pytest.FixtureRequest) -> ImageGenerator:
    factory: Callable[[], ImageGenerator] = request.param
    return factory()


@pytest.fixture
def params() -> Generation:
    return Generation(temperature=0.15, aspect_ratio="1:1", face_media_resolution="high")


async def test_is_an_image_generator(generator: ImageGenerator) -> None:
    assert isinstance(generator, ImageGenerator)


async def test_returns_decodable_image_and_request_id(
    generator: ImageGenerator, params: Generation
) -> None:
    result = await generator.generate(
        prompt="a studio headshot",
        params=params,
        reference_images=[b"reference-photo-bytes"],
    )

    assert isinstance(result, GeneratedImage)
    image_bytes, request_id = result  # must unpack as a 2-tuple
    assert isinstance(request_id, str) and request_id
    # Bytes must be a real image the rest of the pipeline can re-open.
    Image.open(io.BytesIO(image_bytes)).verify()


async def test_accepts_multiple_reference_images(
    generator: ImageGenerator, params: Generation
) -> None:
    result = await generator.generate(
        prompt="a studio headshot",
        params=params,
        reference_images=[b"ref-a", b"ref-b", b"ref-c"],
    )
    assert result.image_bytes


async def test_accepts_an_explicit_face_crop(generator: ImageGenerator, params: Generation) -> None:
    result = await generator.generate(
        prompt="a studio headshot",
        params=params,
        reference_images=[b"reference-photo-bytes"],
        face_crop=b"face-crop-bytes",
    )
    assert result.image_bytes
