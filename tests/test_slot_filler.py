"""DefaultSlotFiller — deterministic answer→vocabulary mapping, no LLM.

The filler may only pick values from the preset's own slot dictionary; the
user's answer steers nothing but the single ``ask:true`` slot. Both the real
fallback and the test fake are pinned to the SlotFiller port here.
"""

from __future__ import annotations

import pytest

from config import Settings
from protocols.slot_filler import SlotFill, SlotFiller
from schemas import Preset
from services.preset_matcher import PresetLibrary, load_library
from services.slot_filler import DefaultSlotFiller, apply_composition
from tests.fakes import FixedSlotFiller

RESUME_ANSWER = "It is for my resume"
UNKNOWN_ANSWER = "я инженер-нефтяник"


@pytest.fixture(scope="module")
def library() -> PresetLibrary:
    return load_library(Settings(_env_file=None))


@pytest.fixture(scope="module")
def headshot(library: PresetLibrary) -> Preset:
    preset = library.get("demo_headshot")
    assert preset is not None
    return preset


def _defaults(preset: Preset) -> dict[str, str]:
    return {name: str(slot.default) for name, slot in preset.slots.items()}


def test_port_conformance() -> None:
    assert isinstance(DefaultSlotFiller(), SlotFiller)
    assert isinstance(FixedSlotFiller(), SlotFiller)


async def test_known_answer_maps_to_vocabulary(headshot: Preset) -> None:
    fill = await DefaultSlotFiller().fill(
        preset=headshot, user_answer=RESUME_ANSWER, photo_analysis=None
    )
    assert fill.slots["purpose"] == "a resume or CV photo"


async def test_unknown_answer_falls_back_to_defaults(headshot: Preset) -> None:
    fill = await DefaultSlotFiller().fill(
        preset=headshot, user_answer=UNKNOWN_ANSWER, photo_analysis=None
    )
    assert fill.slots == _defaults(headshot)


async def test_no_answer_falls_back_to_defaults(headshot: Preset) -> None:
    fill = await DefaultSlotFiller().fill(preset=headshot, user_answer=None, photo_analysis=None)
    assert fill.slots == _defaults(headshot)
    assert fill.addendum == ""


async def test_answer_never_steers_non_ask_slots(headshot: Preset) -> None:
    # "blazer" matches an attire enum option, but attire is not the asked slot.
    fill = await DefaultSlotFiller().fill(
        preset=headshot, user_answer="dark blazer please", photo_analysis=None
    )
    assert fill.slots["attire"] == str(headshot.slots["attire"].default)


async def test_every_slot_filled_and_in_vocabulary(library: PresetLibrary) -> None:
    filler = DefaultSlotFiller()
    for preset_id in library.ids:
        preset = library.get(preset_id)
        assert preset is not None
        fill = await filler.fill(preset=preset, user_answer=RESUME_ANSWER, photo_analysis=None)
        assert set(fill.slots) == set(preset.slots)
        for name, value in fill.slots.items():
            enum = preset.slots[name].enum
            if enum is not None:
                assert value in {str(o) for o in enum}, f"{preset.id}/{name}: {value!r}"


async def test_deterministic(headshot: Preset) -> None:
    filler = DefaultSlotFiller()
    first = await filler.fill(preset=headshot, user_answer=RESUME_ANSWER, photo_analysis=None)
    second = await filler.fill(preset=headshot, user_answer=RESUME_ANSWER, photo_analysis=None)
    assert first == second


def test_apply_composition_overlays_overrides(headshot: Preset) -> None:
    merged = apply_composition(headshot, _defaults(headshot), "studio_clean")
    assert merged["background"] == "a clean light-grey studio backdrop"
    assert merged["expression"] == "a neutral, composed expression"
    # Slots the composition does not override keep their resolved values.
    assert merged["attire"] == str(headshot.slots["attire"].default)


def test_apply_composition_unknown_id_raises(headshot: Preset) -> None:
    with pytest.raises(ValueError, match="no composition"):
        apply_composition(headshot, _defaults(headshot), "nope")


async def test_fixed_fake_returns_configured_fill_and_records(headshot: Preset) -> None:
    configured = SlotFill(slots={"purpose": "x"}, addendum="warm light")
    fake = FixedSlotFiller(configured)
    fill = await fake.fill(preset=headshot, user_answer="hi", photo_analysis=None)
    assert fill == configured
    assert len(fake.calls) == 1
    assert fake.calls[0].preset_id == "demo_headshot"
    assert fake.calls[0].user_answer == "hi"


async def test_fixed_fake_defaults_to_preset_defaults(headshot: Preset) -> None:
    fill = await FixedSlotFiller().fill(preset=headshot, user_answer=None, photo_analysis=None)
    assert fill.slots == _defaults(headshot)
