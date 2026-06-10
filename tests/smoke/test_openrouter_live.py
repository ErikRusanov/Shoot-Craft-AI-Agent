"""Live smoke against the real OpenRouter API — **spends real money**.

Skipped unless ``OPENROUTER_LIVE=1`` (and an ``OPENROUTER_API_KEY``) is set, so
neither CI nor a plain ``make test`` ever pays for a generation. Run explicitly:

    OPENROUTER_LIVE=1 make test ARGS=tests/smoke/test_openrouter_live.py

Assertions are deliberately loose — the point is "the wire format we built is
the one the real API accepts today", not output quality. Everything the API
returned is dropped into ``.smoke/`` (gitignored) so a human can eyeball what
the money actually bought.
"""

from __future__ import annotations

import io
import json
import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from config import Settings
from schemas import Generation, Preset
from services.connectors import (
    OpenRouterClient,
    OpenRouterImageGenerator,
    OpenRouterSlotFiller,
)
from services.preset_matcher import load_library

pytestmark = pytest.mark.skipif(
    os.environ.get("OPENROUTER_LIVE") != "1",
    reason="live OpenRouter smoke (paid) — set OPENROUTER_LIVE=1 to run",
)

ARTIFACTS = Path(".smoke")


def _save(name: str, data: bytes) -> Path:
    ARTIFACTS.mkdir(exist_ok=True)
    path = ARTIFACTS / name
    path.write_bytes(data)
    return path


@pytest.fixture
def settings() -> Settings:
    settings = Settings()
    if not settings.openrouter_api_key:
        pytest.skip("OPENROUTER_API_KEY is not configured")
    return settings


@pytest.fixture
async def client(settings: Settings) -> AsyncIterator[OpenRouterClient]:
    client = OpenRouterClient(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        timeout_seconds=settings.openrouter_timeout_seconds,
        attempts=settings.openrouter_retry_attempts,
    )
    yield client
    await client.aclose()


def _synthetic_portrait() -> bytes:
    """A simple face-like drawing — enough for the edit model to anchor on."""
    img = Image.new("RGB", (512, 512), (210, 200, 190))
    draw = ImageDraw.Draw(img)
    draw.ellipse((128, 96, 384, 416), fill=(224, 172, 140))  # head
    draw.ellipse((192, 208, 224, 240), fill=(60, 40, 30))  # left eye
    draw.ellipse((288, 208, 320, 240), fill=(60, 40, 30))  # right eye
    draw.arc((208, 288, 304, 352), start=20, end=160, fill=(120, 60, 50), width=6)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


async def test_generate_one_image(settings: Settings, client: OpenRouterClient) -> None:
    generator = OpenRouterImageGenerator(client, model=settings.generation_model)
    reference = _synthetic_portrait()

    image_bytes, request_id = await generator.generate(
        prompt=(
            "A photorealistic studio portrait of the person in the reference "
            "image, neutral grey background, soft frontal lighting."
        ),
        params=Generation(temperature=0.4, aspect_ratio="1:1", face_media_resolution="high"),
        reference_images=[reference],
        face_crop=reference,
    )

    assert request_id
    Image.open(io.BytesIO(image_bytes)).verify()

    _save("reference.png", reference)
    generated = _save("generated.png", image_bytes)
    print(f"\ngenerated image ({request_id}) -> {generated.resolve()}")


async def test_slot_filler_fills_a_demo_preset(
    settings: Settings, client: OpenRouterClient
) -> None:
    preset: Preset | None = load_library(Settings(_env_file=None)).get("demo_headshot")
    assert preset is not None
    filler = OpenRouterSlotFiller(client, model=settings.slot_filler_model)

    fill = await filler.fill(
        preset=preset,
        user_answer="I need a photo for my resume, I work in finance",
        photo_analysis=None,
    )

    assert set(fill.slots) == set(preset.slots)
    for name, value in fill.slots.items():
        enum = preset.slots[name].enum
        if enum is not None:
            assert value in {str(o) for o in enum}, f"{name}: {value!r}"

    dump = {"slots": fill.slots, "addendum": fill.addendum}
    path = _save("slot_fill.json", json.dumps(dump, indent=2, ensure_ascii=False).encode())
    print(f"\nslot fill -> {path.resolve()}")
