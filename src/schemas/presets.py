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
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

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
    # locked = a deterministic, non-negotiable preset attribute (passport white
    # background, frontal pose): it wins over a user delta and the conflict is
    # surfaced to the user. default = the preset's default, a user delta wins.
    policy: Literal["locked", "default"] = "default"


class AppliesTo(_Strict):
    use_case: list[str]


class Generation(_Strict):
    # Low temperature keeps identity reproducible.
    temperature: float
    aspect_ratio: str
    # The model has no denoise/strength — only these knobs exist.
    face_media_resolution: str
    # Output resolution tier passed to the image model (e.g. "1K", "2K", "4K").
    # Defaults to "2K" for better portrait quality; "1K" was the old implicit default.
    output_size: str = "2K"

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
    # v2: dropped applies_to.age. v3: dropped applies_to.gender — the face comes
    # from the reference, so matching is use_case only. v4: added `mode`,
    # `style_notes` and slot `policy` for the brief-driven writer pipeline;
    # `prompt_structure` demoted to the no-LLM fallback template. v5: dropped the
    # `convergence` block — budget is spent greedily (reserve-per-generation), not
    # forecast from a per-step attempt count.
    schema_v: int = 5
    id: str
    version: str
    applies_to: AppliesTo
    # generate = target-driven template; edit = delta-driven on the user's photo;
    # both = usable either way. Drives mode-aware preset resolution.
    mode: Literal["generate", "edit", "both"] = "generate"
    identity_instruction: str
    # Demoted from "the prompt" to the deterministic no-LLM fallback template:
    # the prompt writer composes the body per situation, and this template is
    # what the pipeline degrades to when the writer LLM is unavailable.
    prompt_structure: str
    # Free-text style guidance fed to the prompt writer (informational, never a
    # frozen block). Empty for template-only presets.
    style_notes: str = ""
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
