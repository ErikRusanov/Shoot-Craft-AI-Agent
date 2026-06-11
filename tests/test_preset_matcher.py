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
import yaml

from config import Settings
from schemas.presets import Preset
from services.preset_matcher import PresetLibrary, _check_min_version, load_library

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

    hit = library.match(use_case="resume")
    assert hit is not None and hit.id == "demo_headshot"

    # An unknown use_case yields no match (not an exception); the bare filter
    # does not fall back — resolve() does.
    assert library.match(use_case="underwater-ballet") is None


def test_resolve_falls_back_to_default(library: PresetLibrary) -> None:
    # Nothing curated admits this request, so resolve() returns the fallback.
    assert library.match(use_case="underwater-ballet") is None
    resolved = library.resolve(use_case="underwater-ballet")
    assert resolved is not None and resolved.id == "default"


def test_resolve_prefers_a_real_match(library: PresetLibrary) -> None:
    # When something curated matches, resolve() returns it, not the fallback.
    resolved = library.resolve(use_case="avatar")
    assert resolved is not None and resolved.id == "demo_avatar"


def test_default_token_is_never_keyword_matched(library: PresetLibrary) -> None:
    # The reserved token is excluded from the filter: even asking for it by name
    # reaches the fallback only through resolve(), never as a keyword match.
    assert library.match(use_case="default") is None
    assert library.fallback is not None and library.fallback.id == "default"
    resolved = library.resolve(use_case="default")
    assert resolved is not None and resolved.id == "default"


def test_no_fallback_returns_none(tmp_path: Path) -> None:
    # A minimal custom library without a `default` preset: resolve() returns None
    # rather than inventing a fallback.
    shutil.copy(_EXAMPLES / "demo_avatar.yaml", tmp_path / "demo_avatar.yaml")
    settings = Settings(_env_file=None, preset_source="path", preset_library_path=str(tmp_path))
    library = load_library(settings)
    assert library.fallback is None
    assert library.resolve(use_case="nope") is None


def test_reserved_token_on_non_default_fails_loudly(tmp_path: Path) -> None:
    data = yaml.safe_load((_EXAMPLES / "demo_avatar.yaml").read_text(encoding="utf-8"))
    data["applies_to"]["use_case"].append("default")
    (tmp_path / "demo_avatar.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    settings = Settings(_env_file=None, preset_source="path", preset_library_path=str(tmp_path))
    with pytest.raises(ValueError, match="reserved use_case token"):
        load_library(settings)


def test_default_preset_without_reserved_token_fails_loudly(tmp_path: Path) -> None:
    data = yaml.safe_load((_EXAMPLES / "default.yaml").read_text(encoding="utf-8"))
    data["applies_to"]["use_case"] = ["avatar"]
    (tmp_path / "default.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    settings = Settings(_env_file=None, preset_source="path", preset_library_path=str(tmp_path))
    with pytest.raises(ValueError, match="must declare the reserved"):
        load_library(settings)


def test_min_version_check() -> None:
    # Older than the required minimum → loud failure; equal/newer → fine.
    with pytest.raises(ValueError, match="older than the required"):
        _check_min_version("0.2.9", "0.3.0")
    _check_min_version("0.3.0", "0.3.0")
    _check_min_version("1.0.0", "0.3.0")
    # An unparseable version is tolerated (logged, not fatal).
    _check_min_version("unknown", "0.3.0")


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
    for use_case in ("avatar", "resume"):
        a = library.match(use_case=use_case)
        b = fresh.match(use_case=use_case)
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
