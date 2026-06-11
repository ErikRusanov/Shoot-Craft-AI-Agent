"""OpenRouterSlotFiller — structured-output contract and degrade-to-default.

Two invariants under test, both on mocked httpx:

- the LLM is shown the slot dictionary only — the frozen preset blocks
  (identity_instruction / prompt_structure / negative_prompt) never appear in
  the request, so there is nothing to leak or edit;
- any misbehavior (garbage JSON, out-of-vocabulary value, transport failure,
  4xx) degrades to :class:`DefaultSlotFiller` output instead of raising.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from config import Settings
from protocols.slot_filler import SlotFill, SlotFiller
from schemas import FrameMetrics, Preset
from services.connectors import OpenRouterSlotFiller
from services.preset_matcher import PresetLibrary, load_library
from services.slot_filler import DefaultSlotFiller
from tests.openrouter_mock import ScriptedTransport, scripted_client, text_completion_body

MODEL = "google/gemini-3.1-flash-lite"
ANSWER = "I need a photo for my resume"


@pytest.fixture(scope="module")
def library() -> PresetLibrary:
    return load_library(Settings(_env_file=None))


@pytest.fixture(scope="module")
def headshot(library: PresetLibrary) -> Preset:
    preset = library.get("demo_headshot")
    assert preset is not None
    return preset


@pytest.fixture(scope="module")
def fallback_preset(library: PresetLibrary) -> Preset:
    preset = library.get("default")
    assert preset is not None
    return preset


@pytest.fixture
def metrics() -> FrameMetrics:
    return FrameMetrics(
        face_count=1,
        face_area_ratio=0.2,
        face_side=300.0,
        blur_var=80.0,
        yaw=5.0,
        pitch=0.0,
        roll=0.0,
        brightness=120.0,
        width=1024,
        height=1024,
    )


def _filler(
    *script: httpx.Response | Exception, attempts: int = 4
) -> tuple[OpenRouterSlotFiller, ScriptedTransport]:
    client, transport = scripted_client(*script, attempts=attempts)
    return OpenRouterSlotFiller(client, model=MODEL), transport


def _valid_fill(preset: Preset) -> dict[str, Any]:
    """A well-formed LLM answer: last enum option per slot, free-form otherwise."""
    slots = {
        name: str(slot.enum[-1]) if slot.enum else "on a quiet rooftop at dusk"
        for name, slot in preset.slots.items()
    }
    return {"slots": slots, "addendum": "soft window light"}


def _llm_ok(preset: Preset) -> httpx.Response:
    return httpx.Response(200, json=text_completion_body(json.dumps(_valid_fill(preset))))


async def _default_fill(preset: Preset, answer: str | None, metrics: FrameMetrics) -> SlotFill:
    return await DefaultSlotFiller().fill(preset=preset, user_answer=answer, photo_analysis=metrics)


def test_port_conformance() -> None:
    filler, _ = _filler(httpx.Response(500))
    assert isinstance(filler, SlotFiller)


async def test_valid_response_maps_to_slots(headshot: Preset, metrics: FrameMetrics) -> None:
    filler, _ = _filler(_llm_ok(headshot))

    fill = await filler.fill(preset=headshot, user_answer=ANSWER, photo_analysis=metrics)

    expected = _valid_fill(headshot)
    assert fill.slots == expected["slots"]
    assert fill.addendum == "soft window light"


async def test_request_is_strict_structured_output(headshot: Preset, metrics: FrameMetrics) -> None:
    filler, transport = _filler(_llm_ok(headshot))

    await filler.fill(preset=headshot, user_answer=ANSWER, photo_analysis=metrics)

    body = json.loads(transport.requests[0].content)
    assert body["model"] == MODEL
    response_format = body["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True

    schema = response_format["json_schema"]["schema"]
    slot_props = schema["properties"]["slots"]["properties"]
    assert set(slot_props) == set(headshot.slots)
    for name, slot in headshot.slots.items():
        if slot.enum is not None:
            # Every enum slot is pinned to the preset's own vocabulary.
            assert slot_props[name]["enum"] == [str(o) for o in slot.enum]


async def test_user_answer_and_metrics_reach_the_llm(
    headshot: Preset, metrics: FrameMetrics
) -> None:
    filler, transport = _filler(_llm_ok(headshot))

    await filler.fill(preset=headshot, user_answer=ANSWER, photo_analysis=metrics)

    raw = transport.requests[0].content.decode()
    assert ANSWER in raw
    assert "blur_var" in raw  # FrameMetrics is photo statistics, safe to send


async def test_frozen_blocks_never_sent(headshot: Preset, metrics: FrameMetrics) -> None:
    filler, transport = _filler(_llm_ok(headshot))

    await filler.fill(preset=headshot, user_answer=ANSWER, photo_analysis=metrics)

    raw = transport.requests[0].content.decode()
    # Sample a distinctive span of each frozen block, not the whole text —
    # JSON escaping could mask an exact full-string match.
    for frozen in (
        headshot.identity_instruction,
        headshot.prompt_structure,
        headshot.negative_prompt,
    ):
        assert frozen[:40] not in raw


async def test_garbage_content_falls_back(headshot: Preset, metrics: FrameMetrics) -> None:
    filler, transport = _filler(httpx.Response(200, json=text_completion_body("not json at all")))

    fill = await filler.fill(preset=headshot, user_answer=ANSWER, photo_analysis=metrics)

    assert fill == await _default_fill(headshot, ANSWER, metrics)
    assert len(transport.requests) == 1


async def test_out_of_vocabulary_value_falls_back(headshot: Preset, metrics: FrameMetrics) -> None:
    poisoned = _valid_fill(headshot)
    enum_slot = next(name for name, s in headshot.slots.items() if s.enum)
    poisoned["slots"][enum_slot] = "a value the preset never declared"
    filler, _ = _filler(httpx.Response(200, json=text_completion_body(json.dumps(poisoned))))

    fill = await filler.fill(preset=headshot, user_answer=ANSWER, photo_analysis=metrics)

    assert fill == await _default_fill(headshot, ANSWER, metrics)


async def test_missing_slot_falls_back(headshot: Preset, metrics: FrameMetrics) -> None:
    partial = _valid_fill(headshot)
    partial["slots"].popitem()
    filler, _ = _filler(httpx.Response(200, json=text_completion_body(json.dumps(partial))))

    fill = await filler.fill(preset=headshot, user_answer=ANSWER, photo_analysis=metrics)

    assert fill == await _default_fill(headshot, ANSWER, metrics)


async def test_transport_failure_exhausts_retries_then_falls_back(
    headshot: Preset, metrics: FrameMetrics
) -> None:
    filler, transport = _filler(httpx.ConnectError("network down"), attempts=2)

    fill = await filler.fill(preset=headshot, user_answer=ANSWER, photo_analysis=metrics)

    assert len(transport.requests) == 2  # transient → retried, then degraded
    assert fill == await _default_fill(headshot, ANSWER, metrics)


async def test_4xx_falls_back_without_retry(headshot: Preset, metrics: FrameMetrics) -> None:
    filler, transport = _filler(httpx.Response(400, text="bad request"))

    fill = await filler.fill(preset=headshot, user_answer=ANSWER, photo_analysis=metrics)

    assert len(transport.requests) == 1
    assert fill == await _default_fill(headshot, ANSWER, metrics)


async def test_no_inputs_skips_the_llm_entirely(headshot: Preset) -> None:
    filler, transport = _filler(_llm_ok(headshot))

    fill = await filler.fill(preset=headshot, user_answer=None, photo_analysis=None)

    assert transport.requests == []  # nothing to interpret → no paid call
    assert fill == await DefaultSlotFiller().fill(
        preset=headshot, user_answer=None, photo_analysis=None
    )


async def test_freeform_scene_is_the_users_words_not_the_llm_paraphrase(
    fallback_preset: Preset, metrics: FrameMetrics
) -> None:
    filler, _ = _filler(_llm_ok(fallback_preset))

    answer = "на крыше на закате"
    fill = await filler.fill(preset=fallback_preset, user_answer=answer, photo_analysis=metrics)

    # The LLM answered "on a quiet rooftop at dusk" for the free-form scene, but
    # the user's own words win — paraphrasing is exactly the bug this prevents.
    assert fill.slots["scene"] == answer


async def test_edit_request_brief_passes_through_verbatim(
    fallback_preset: Preset, metrics: FrameMetrics
) -> None:
    # The sandbox regression: an edit request must reach the model verbatim, not
    # be rewritten into a generic "flattering setting" scene.
    filler, _ = _filler(_llm_ok(fallback_preset))

    answer = "do not change the photo, make the background blue light"
    fill = await filler.fill(preset=fallback_preset, user_answer=answer, photo_analysis=metrics)

    assert fill.slots["scene"] == answer
