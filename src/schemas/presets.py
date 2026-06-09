"""Preset schema — the canonical contract for the preset library.

This is the **single source of truth** for a preset's shape. The external
``photocore-presets`` package (the private library) authors YAML against this
schema and keeps a local mirror (``tests/_schema_mirror.py`` there) only because
the core ships flat (``package=false``) and is not importable as ``photocore``.
If the two ever diverge, **this file wins** — sync the mirror and the YAML to it.

Versioning note: a preset is authored externally and versioned by its own
``version`` (semver), with the package version recorded separately as
``library_version`` on ``SessionState``. ``schema_v`` here versions the *contract*
itself (this shape), independently of any individual preset's ``version``.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

_SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


class _Strict(BaseModel):
    """Reject unknown keys so a typo in a YAML preset fails loudly at load."""

    model_config = ConfigDict(extra="forbid")


class Slot(_Strict):
    required: bool = False
    # ask:true marks the single clarifying question the agent asks the user.
    ask: bool = False
    default: object | None = None
    enum: list[object] | None = None


class AppliesTo(_Strict):
    use_case: list[str]
    gender: list[str]
    age: tuple[int, int]  # [min, max]

    @model_validator(mode="after")
    def _age_ordered(self) -> AppliesTo:
        lo, hi = self.age
        if lo > hi:
            raise ValueError(f"age min {lo} > max {hi}")
        return self


class Generation(_Strict):
    # Low temperature keeps identity reproducible.
    temperature: float
    aspect_ratio: str
    # The model has no denoise/strength — only these knobs exist.
    face_media_resolution: str

    @field_validator("temperature")
    @classmethod
    def _temp_range(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError(f"temperature {v} out of range [0, 2]")
        return v


class Thresholds(_Strict):
    similarity_threshold: float
    identity_floor: float
    K_max_retries: int


class Composition(_Strict):
    id: str
    label: str
    preview_asset: str | None = None
    slot_overrides: dict[str, object] = {}


class Preset(_Strict):
    # Contract version of this shape; absent in YAML → defaults. Bump only on a
    # breaking change to the preset schema itself, not on a library content edit.
    schema_v: int = 1
    id: str
    version: str
    applies_to: AppliesTo
    identity_instruction: str
    prompt_structure: str
    # Stored data only — Nano Banana / Gemini has no negative-prompt API field;
    # the prompt builder inlines these terms into the prompt text as exclusions.
    negative_prompt: str
    slots: dict[str, Slot]
    generation: Generation
    thresholds: Thresholds
    compositions: list[Composition] = []
    anchor_examples: list[str] = []

    @field_validator("version")
    @classmethod
    def _semver(cls, v: str) -> str:
        if not _SEMVER.match(v):
            raise ValueError(f"version {v!r} is not semver MAJOR.MINOR.PATCH")
        return v
