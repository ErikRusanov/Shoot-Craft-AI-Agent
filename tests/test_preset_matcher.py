"""Loader + schema tests against the in-repo demo presets (PRESET_SOURCE=examples).

This exercises the exact path the public core runs on out of the box. The same
structural rules are enforced in the private presets repo against the real
library; keeping them here guards the demo set and the schema itself.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

from config import Settings
from schemas.presets import Preset
from services.preset_matcher import PresetLibrary, load_library

_PLACEHOLDER = re.compile(r"\{(\w+)\}")
_EXAMPLES = Path(__file__).resolve().parents[1] / "presets" / "examples"


@pytest.fixture(scope="module")
def library() -> PresetLibrary:
    # Defaults select PRESET_SOURCE=examples; do not read a developer's .env.
    return load_library(Settings(_env_file=None))


def test_loads_examples(library: PresetLibrary) -> None:
    assert library.source == "examples"
    assert library.library_version == "examples"
    assert len(library) >= 2
    assert "demo_avatar" in library.ids


def test_get_and_match(library: PresetLibrary) -> None:
    assert library.get("demo_avatar") is not None
    assert library.get("nope") is None

    hit = library.match(use_case="resume", gender="female", age=30)
    assert hit is not None and hit.id == "demo_headshot"

    # Age outside every preset's range yields no match (not an exception).
    assert library.match(use_case="headshot", gender="male", age=5) is None


def _presets(library: PresetLibrary) -> list[Preset]:
    return [p for p in (library.get(i) for i in library.ids) if p is not None]


def test_placeholders_match_slots(library: PresetLibrary) -> None:
    for preset in _presets(library):
        slots = set(preset.slots)
        used = set(_PLACEHOLDER.findall(preset.prompt_structure))
        assert used == slots, f"{preset.id}: placeholders {used} != slots {slots}"


def test_single_ask_slot(library: PresetLibrary) -> None:
    for preset in _presets(library):
        ask = [name for name, slot in preset.slots.items() if slot.ask]
        assert len(ask) == 1, f"{preset.id}: expected one ask:true slot, got {ask}"


def test_overrides_reference_declared_slots(library: PresetLibrary) -> None:
    for preset in _presets(library):
        for comp in preset.compositions:
            unknown = set(comp.slot_overrides) - set(preset.slots)
            assert not unknown, f"{preset.id}/{comp.id}: unknown override slots {unknown}"


def test_path_source_requires_path() -> None:
    with pytest.raises(ValueError, match="preset_library_path is required"):
        Settings(_env_file=None, preset_source="path")


def test_match_is_deterministic(library: PresetLibrary) -> None:
    fresh = load_library(Settings(_env_file=None))
    for use_case, gender, age in [("avatar", "male", 30), ("resume", "female", 40)]:
        a = library.match(use_case=use_case, gender=gender, age=age)
        b = fresh.match(use_case=use_case, gender=gender, age=age)
        assert a is not None and b is not None and a.id == b.id


def test_path_source(tmp_path: Path) -> None:
    shutil.copy(_EXAMPLES / "demo_avatar.yaml", tmp_path / "demo_avatar.yaml")
    settings = Settings(_env_file=None, preset_source="path", preset_library_path=str(tmp_path))
    library = load_library(settings)
    assert library.source == "path"
    assert library.library_version.startswith("path:")
    assert library.ids == ("demo_avatar",)


def test_duplicate_preset_id_fails_loudly(tmp_path: Path) -> None:
    shutil.copy(_EXAMPLES / "demo_avatar.yaml", tmp_path / "a.yaml")
    shutil.copy(_EXAMPLES / "demo_avatar.yaml", tmp_path / "b.yaml")
    settings = Settings(_env_file=None, preset_source="path", preset_library_path=str(tmp_path))
    with pytest.raises(ValueError, match="duplicate preset id"):
        load_library(settings)
