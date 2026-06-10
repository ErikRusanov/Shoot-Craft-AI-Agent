"""Prompt assembly — the only place preset text becomes a generation prompt.

Frozen-block rule: ``identity_instruction`` and ``prompt_structure`` go in
**verbatim** — the single transformation ever applied is substituting declared
``{slot}`` placeholders with validated values. The addendum is the one
sanctioned free-text extension and is appended after the structure, never
inside it. ``negative_prompt`` is inlined into the text as an exclusion clause
because Nano Banana (Gemini flash-image) has no negative-prompt API parameter.

Slot values are re-validated against the preset's vocabulary here, so even a
misbehaving (LLM) filler cannot smuggle arbitrary text into the frozen blocks.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import NamedTuple

from schemas import Generation, Preset

_PLACEHOLDER = re.compile(r"\{(\w+)\}")
_EXCLUSION_PREFIX = "Strictly avoid: "


class BuiltPrompt(NamedTuple):
    """The final prompt text, its hash, and the preset's generation knobs.

    ``prompt_hash`` is what :class:`~schemas.state.Iteration` records;
    ``params`` are the frozen preset knobs the generator port expects — the
    builder passes them through so callers never assemble their own.
    """

    text: str
    prompt_hash: str
    params: Generation


def build_prompt(preset: Preset, slots: Mapping[str, str], *, addendum: str = "") -> BuiltPrompt:
    """Assemble the deterministic prompt for one generation attempt.

    Raises ``ValueError`` on any slot that is unknown to the preset, missing
    for a placeholder, or outside the slot's declared ``enum``.
    """
    _validate_slots(preset, slots)
    structure = _PLACEHOLDER.sub(lambda m: slots[m.group(1)], preset.prompt_structure)

    parts = [preset.identity_instruction, structure]
    if addendum.strip():
        parts.append(addendum.strip())
    parts.append(_EXCLUSION_PREFIX + preset.negative_prompt)

    text = "\n\n".join(parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return BuiltPrompt(text=text, prompt_hash=digest, params=preset.generation)


def _validate_slots(preset: Preset, slots: Mapping[str, str]) -> None:
    unknown = set(slots) - set(preset.slots)
    if unknown:
        raise ValueError(f"preset {preset.id!r}: unknown slots {sorted(unknown)}")

    missing = set(_PLACEHOLDER.findall(preset.prompt_structure)) - set(slots)
    if missing:
        raise ValueError(f"preset {preset.id!r}: missing values for slots {sorted(missing)}")

    for name, value in slots.items():
        enum = preset.slots[name].enum
        if enum is not None and value not in {str(option) for option in enum}:
            raise ValueError(
                f"preset {preset.id!r}: slot {name!r} value {value!r} is not in its vocabulary"
            )
