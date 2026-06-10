"""Prompt builder — frozen blocks verbatim, slots validated, output golden.

The golden snapshot is a literal, not a re-derivation: if the builder's layout
or the demo preset's text changes, this test must be consciously updated — that
is the point, the prompt is part of the reproducibility contract.
"""

from __future__ import annotations

import hashlib

import pytest

from config import Settings
from schemas import Preset
from services.preset_matcher import PresetLibrary, load_library
from services.prompt_builder import build_prompt

GOLDEN_AVATAR_PROMPT = (
    "Preserve the exact facial identity of the person in the reference photo: the "
    "same face geometry, eyes, nose, lips, jawline, skin tone, and any moles or "
    "scars. Do not beautify, slim, age, or restyle the face. Reproduce a faithful, "
    "recognizable likeness of the same individual."
    "\n\n"
    "Clean head-and-shoulders profile avatar of the person, with a relaxed, friendly "
    "closed-mouth smile, looking at the camera, photographed against a plain, softly "
    "lit neutral background. Intended for a general profile avatar. Photorealistic, "
    "natural skin texture, soft even lighting, sharp focus."
    "\n\n"
    "Strictly avoid: cartoon, illustration, 3d render, plastic or waxy skin, beauty "
    "filter, distorted face, extra fingers, watermark, text, logo, low resolution, "
    "blurry, different person."
)


@pytest.fixture(scope="module")
def library() -> PresetLibrary:
    return load_library(Settings(_env_file=None))


@pytest.fixture(scope="module")
def avatar(library: PresetLibrary) -> Preset:
    preset = library.get("demo_avatar")
    assert preset is not None
    return preset


def _defaults(preset: Preset) -> dict[str, str]:
    return {name: str(slot.default) for name, slot in preset.slots.items()}


def test_golden_snapshot(avatar: Preset) -> None:
    built = build_prompt(avatar, _defaults(avatar))
    assert built.text == GOLDEN_AVATAR_PROMPT


def test_deterministic_text_and_hash(avatar: Preset) -> None:
    first = build_prompt(avatar, _defaults(avatar))
    second = build_prompt(avatar, _defaults(avatar))
    assert first == second
    assert first.prompt_hash == hashlib.sha256(first.text.encode("utf-8")).hexdigest()


def test_frozen_blocks_verbatim(library: PresetLibrary) -> None:
    for preset_id in library.ids:
        preset = library.get(preset_id)
        assert preset is not None
        built = build_prompt(preset, _defaults(preset))
        assert preset.identity_instruction in built.text
        assert built.text.endswith("Strictly avoid: " + preset.negative_prompt)
        assert "{" not in built.text and "}" not in built.text


def test_generation_params_pass_through(avatar: Preset) -> None:
    built = build_prompt(avatar, _defaults(avatar))
    assert built.params is avatar.generation
    assert built.params.aspect_ratio == "1:1"
    assert built.params.face_media_resolution == "high"
    assert built.params.temperature == pytest.approx(0.50)


def test_addendum_between_structure_and_exclusions(avatar: Preset) -> None:
    built = build_prompt(avatar, _defaults(avatar), addendum="Warm golden-hour light.")
    base = build_prompt(avatar, _defaults(avatar))
    head, _, tail = base.text.rpartition("\n\n")
    assert built.text == head + "\n\nWarm golden-hour light.\n\n" + tail
    assert built.prompt_hash != base.prompt_hash


def test_blank_addendum_changes_nothing(avatar: Preset) -> None:
    assert build_prompt(avatar, _defaults(avatar), addendum="  \n") == build_prompt(
        avatar, _defaults(avatar)
    )


def test_different_slots_change_the_hash(avatar: Preset) -> None:
    slots = _defaults(avatar)
    slots["background"] = "a clean light-grey backdrop"
    assert (
        build_prompt(avatar, slots).prompt_hash
        != build_prompt(avatar, _defaults(avatar)).prompt_hash
    )


def test_missing_slot_raises(avatar: Preset) -> None:
    slots = _defaults(avatar)
    del slots["background"]
    with pytest.raises(ValueError, match="missing values"):
        build_prompt(avatar, slots)


def test_unknown_slot_raises(avatar: Preset) -> None:
    with pytest.raises(ValueError, match="unknown slots"):
        build_prompt(avatar, {**_defaults(avatar), "mood": "joyful"})


def test_value_outside_vocabulary_raises(avatar: Preset) -> None:
    with pytest.raises(ValueError, match="not in its vocabulary"):
        build_prompt(avatar, {**_defaults(avatar), "background": "a neon cyberpunk alley"})
