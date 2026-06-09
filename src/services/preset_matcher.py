"""Preset library loader and matcher.

Resolves the preset source **from config** (never a hardcoded path), reads every
YAML, validates it against :class:`schemas.presets.Preset`, and holds the result
immutably for the lifetime of the process. The library is loaded once at startup.

Three sources, selected by ``settings.preset_source``:

- ``package`` (prod) — ``importlib.resources.files(preset_package) / "library"``.
  The core does **not** depend on that package; it is installed only in the
  private image. The package version becomes ``library_version``.
- ``path`` (dev / custom) — read ``preset_library_path`` directly. Useful for
  local work against a sibling checkout of the private library; there is no
  package metadata, so ``library_version`` is a non-reproducible ``path:`` marker.
- ``examples`` (default) — the in-repo ``presets/examples/`` demo set, so the
  public core runs out of the box with no private library present.

Only the package **name** (a config string) and this loader interface ever name
the private library — no URL, no content, no hard dependency.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.resources
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import structlog
import yaml

from config import Settings
from schemas.presets import Preset

log = structlog.get_logger()

# Repo root: src/services/preset_matcher.py -> parents[2] == repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_DIR = _REPO_ROOT / "presets" / "examples"


@dataclass(frozen=True)
class PresetLibrary:
    """Immutable in-memory index of the loaded presets.

    ``library_version`` is recorded on ``SessionState`` so a result can be
    reproduced after a library update; in ``path`` mode it is a ``path:`` marker,
    not a reproducible semver.
    """

    library_version: str
    source: str
    _by_id: Mapping[str, Preset]

    def get(self, preset_id: str) -> Preset | None:
        return self._by_id.get(preset_id)

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(self._by_id)

    def __len__(self) -> int:
        return len(self._by_id)

    def match(self, *, use_case: str, gender: str, age: int) -> Preset | None:
        """First preset whose ``applies_to`` admits the request, else ``None``.

        Intentionally a simple linear filter — the seam for real ranking later.
        ``any`` in a preset's ``gender`` (or ``use_case``) acts as a wildcard.
        """
        for preset in self._by_id.values():
            a = preset.applies_to
            if use_case not in a.use_case and "any" not in a.use_case:
                continue
            if gender not in a.gender and "any" not in a.gender:
                continue
            lo, hi = a.age
            if not lo <= age <= hi:
                continue
            return preset
        return None


def load_library(settings: Settings) -> PresetLibrary:
    """Resolve the source, load + validate every preset, return an immutable index."""
    root, version, source = _resolve_source(settings)

    by_id: dict[str, Preset] = {}
    for path in _iter_yaml(root):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        preset = Preset(**data)  # raises on any schema violation
        if preset.id in by_id:
            raise ValueError(f"duplicate preset id {preset.id!r} in {source} library")
        by_id[preset.id] = preset

    if not by_id:
        raise ValueError(f"no presets found in {source} library at {root!r}")

    log.info(
        "preset_library_loaded",
        source=source,
        library_version=version,
        count=len(by_id),
        ids=sorted(by_id),
    )
    return PresetLibrary(library_version=version, source=source, _by_id=MappingProxyType(by_id))


def _resolve_source(settings: Settings) -> tuple[Path, str, str]:
    """Return ``(library_root, library_version, source_label)`` per config."""
    source = settings.preset_source

    if source == "package":
        pkg = settings.preset_package
        # Coerce the package resource to a real path (the library ships unpacked
        # as wheel data); same idiom as photocore_presets.library_path().
        root = Path(str(importlib.resources.files(pkg).joinpath("library")))
        if not root.is_dir():
            raise ValueError(f"package {pkg!r} has no library/ directory at {root!r}")
        return root, _package_version(pkg), f"package:{pkg}"

    if source == "path":
        # Guarded by Settings validation, but assert for the type checker.
        assert settings.preset_library_path is not None
        root = Path(settings.preset_library_path).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"preset_library_path {root!r} is not a directory")
        return root, f"path:{root}", "path"

    return _EXAMPLES_DIR, "examples", "examples"


def _package_version(pkg: str) -> str:
    """Distribution version for ``library_version``; tolerant of name/install form."""
    for name in (pkg, pkg.replace("_", "-")):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    # Editable installs without dist metadata: fall back to the module's own.
    module = importlib.import_module(pkg)
    return str(getattr(module, "__version__", "unknown"))


def _iter_yaml(root: Path) -> list[Path]:
    """Sorted ``*.yaml`` files directly under ``root``."""
    return sorted(root.glob("*.yaml"))
