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
import re
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

# The reserved fallback preset (library convention, ≥ photocore-presets 0.3.0):
# its id is `default` and its sole ``applies_to.use_case`` token is `default`.
# That token is reserved — it must not appear in any other preset, and the
# keyword matcher must never reach the fallback through it. The fallback is
# reachable only via ``resolve()`` when nothing else matches.
_FALLBACK_ID = "default"
_RESERVED_USE_CASE = "default"

_SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)")


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

    @property
    def use_case_tokens(self) -> tuple[str, ...]:
        """Curated ``use_case`` tokens across the library, minus the reserved one.

        The vocabulary the classifier maps free text onto and the ``GET /presets``
        endpoint advertises; the reserved ``default`` fallback token is excluded —
        it is never a choice, only a fall-through.
        """
        tokens: set[str] = set()
        for preset in self._by_id.values():
            tokens.update(t for t in preset.applies_to.use_case if t != _RESERVED_USE_CASE)
        return tuple(sorted(tokens))

    def match(self, *, use_case: str) -> Preset | None:
        """First preset whose ``applies_to`` admits ``use_case``, else ``None``.

        Intentionally a simple linear filter — the seam for real ranking later.
        ``any`` in a preset's ``use_case`` acts as a wildcard. The reserved
        fallback is skipped here: it is *never* keyword-matched, only reached via
        :meth:`resolve` when nothing else admits the request — even a literal
        ``use_case="default"`` falls through to the fallback, not a match.
        """
        for preset in self._by_id.values():
            a = preset.applies_to
            if _RESERVED_USE_CASE in a.use_case:
                continue  # reserved fallback — reachable only through resolve()
            if use_case in a.use_case or "any" in a.use_case:
                return preset
        return None

    @property
    def fallback(self) -> Preset | None:
        """The reserved ``default`` fallback preset, or ``None`` if absent.

        Always present in a ``package``-mode library (guaranteed from
        photocore-presets 0.3.0); a minimal custom ``path`` library may omit it.
        """
        return self._by_id.get(_FALLBACK_ID)

    def resolve(self, *, use_case: str) -> Preset | None:
        """Best ``applies_to`` match, else the reserved ``default`` fallback.

        This is the entry point the orchestration uses: a request that no curated
        preset admits (including an empty/unknown ``use_case``) resolves to the
        fallback rather than failing. Returns ``None`` only when nothing matches
        *and* the library ships no fallback.
        """
        return self.match(use_case=use_case) or self.fallback


def load_library(settings: Settings) -> PresetLibrary:
    """Resolve the source, load + validate every preset, return an immutable index."""
    root, version, source = _resolve_source(settings)

    by_id: dict[str, Preset] = {}
    for path in _iter_yaml(root):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        preset = Preset(**data)  # raises on any schema violation
        if preset.id in by_id:
            raise ValueError(f"duplicate preset id {preset.id!r} in {source} library")
        _check_reserved_token(preset, source)
        by_id[preset.id] = preset

    if not by_id:
        raise ValueError(f"no presets found in {source} library at {root!r}")

    if settings.preset_source == "package":
        _check_min_version(version, settings.preset_min_library_version)
        if _FALLBACK_ID not in by_id:
            raise ValueError(
                f"{source} library has no reserved {_FALLBACK_ID!r} fallback preset; "
                f"the fallback convention requires photocore-presets "
                f">= {settings.preset_min_library_version}"
            )

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


def _check_reserved_token(preset: Preset, source: str) -> None:
    """Enforce the reserved-token convention at load time.

    The ``default`` use_case token may appear only on the fallback preset (whose
    id is ``default``), and the fallback must claim exactly that token — so a
    typo'd or mis-scoped preset fails loudly instead of silently shadowing the
    fallback or leaking it into keyword matching.
    """
    declares_token = _RESERVED_USE_CASE in preset.applies_to.use_case
    is_fallback = preset.id == _FALLBACK_ID
    if declares_token and not is_fallback:
        raise ValueError(
            f"{source} library: preset {preset.id!r} declares the reserved "
            f"use_case token {_RESERVED_USE_CASE!r}, allowed only on the "
            f"{_FALLBACK_ID!r} fallback preset"
        )
    if is_fallback and not declares_token:
        raise ValueError(
            f"{source} library: fallback preset {_FALLBACK_ID!r} must declare the "
            f"reserved use_case token {_RESERVED_USE_CASE!r}"
        )


def _check_min_version(version: str, minimum: str) -> None:
    """Fail fast if a ``package``-mode library predates the fallback convention.

    Only meaningful for real semver package versions; the ``path``/``examples``
    markers never reach here. A version that does not parse as semver is logged
    and allowed through rather than blocking startup on an odd version string.
    """
    have, want = _SEMVER.match(version), _SEMVER.match(minimum)
    if have is None:
        log.warning("preset_library_version_unparsed", version=version, minimum=minimum)
        return
    assert want is not None  # minimum is a controlled constant/setting
    if tuple(map(int, have.groups())) < tuple(map(int, want.groups())):
        raise ValueError(
            f"photocore-presets {version} is older than the required >= {minimum}: "
            f"the reserved {_FALLBACK_ID!r} fallback convention needs >= {minimum}"
        )


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
