"""Prompt assembly — the only place preset text becomes a generation prompt.

Frozen-block rule: ``identity_instruction`` and ``prompt_structure`` go in
**verbatim** — the single transformation ever applied is substituting declared
``{slot}`` placeholders with validated values. The addendum is the one
sanctioned free-text extension and is appended after the structure, never
inside it. ``negative_prompt`` is inlined into the text as an exclusion clause
because Nano Banana (Gemini flash-image) has no negative-prompt API parameter.

Slot values are re-validated against the preset's vocabulary here, so even a
misbehaving (LLM) filler cannot smuggle arbitrary text into the frozen blocks.
A slot without an ``enum`` (the fallback preset's free-form ``scene``) has no
vocabulary to check against; instead its user-supplied text is sanitized — read
as a scene description only, with prompt-injection attempts (altering the face /
identity, or overriding the frozen instructions) rejected so the caller re-asks.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import NamedTuple

from schemas import Generation, Preset

_PLACEHOLDER = re.compile(r"\{(\w+)\}")
_EXCLUSION_PREFIX = "Strictly avoid: "

# Prompt-injection guards for free-form (enum-less) slot text. A free-form slot
# describes a *scene*; it must never carry instructions that reach past the scene
# to edit the frozen identity block or the frozen structure. These patterns catch
# the two attack shapes — (1) override/ignore the surrounding instructions, and
# (2) change/replace the face or identity — and the matched value is rejected so
# the orchestration re-asks rather than feeding a poisoned prompt to the model.
# Erring toward rejection is deliberate: a false reject costs a re-ask; a false
# accept costs a wrong face in a paid generation.
_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Override / disregard / leak the surrounding (frozen) instructions.
    re.compile(
        r"\b(ignore|disregard|forget|override|overrule|bypass|skip|cancel|remove|delete|drop)\b"
        r"[^.]{0,40}\b(instruction|instructions|prompt|prompts|rule|rules|directive|directives"
        r"|guideline|guidelines|constraint|constraints|context|wording|everything|above"
        r"|previous|prior|system)\b"
    ),
    # Bare "ignore the above / previous", "new instructions", system-prompt talk.
    re.compile(r"\b(ignore|disregard|forget)\b[^.]{0,20}\b(above|previous|prior|earlier)\b"),
    re.compile(r"\bnew\b[^.]{0,20}\b(instruction|instructions|prompt|prompts|rules)\b"),
    re.compile(r"\bsystem\s+prompt\b"),
    # Edit / replace / disable the face or identity.
    re.compile(
        r"\b(change|alter|swap|switch|replace|modify|edit|morph|reshape|remove|delete|drop"
        r"|disable|beautify|slim|de-?age|fix|improve|enhance)\b"
        r"[^.]{0,40}\b(face|facial|identity|likeness|features|jaw|jawline|nose|eyes|lips"
        r"|skin tone|complexion|appearance)\b"
    ),
    # Make it a different / another person; face-swap onto someone else.
    re.compile(
        r"\b(different|another|other|new|someone else'?s?|somebody else'?s?|a different)\b"
        r"[^.]{0,20}\b(face|person|identity|man|woman|guy|girl|people|individual)\b"
    ),
    re.compile(
        r"\b(look like|turn me into|make me look like|become)\b"
        r"[^.]{0,30}\b(someone|else|a different|another)\b"
    ),
    # Direct references to the frozen blocks by name.
    re.compile(r"\b(identity[_ ]instruction|prompt[_ ]structure|negative[_ ]prompt)\b"),
)


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
        if enum is None:
            # Free-form slot: no vocabulary to check, sanitize the text instead.
            _reject_injection(preset, name, value)
        elif value not in {str(option) for option in enum}:
            raise ValueError(
                f"preset {preset.id!r}: slot {name!r} value {value!r} is not in its vocabulary"
            )


def _reject_injection(preset: Preset, name: str, value: str) -> None:
    """Reject free-form slot text that reads as a prompt injection.

    A free-form value is scene description only. If it instead tries to alter the
    face/identity or override the frozen instructions, fail loudly so the caller
    re-asks the question — the text must not reach the frozen blocks.
    """
    lowered = value.lower()
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(lowered):
            raise ValueError(
                f"preset {preset.id!r}: slot {name!r} free-form text rejected — it reads "
                f"as an instruction to alter identity or override the prompt; "
                f"describe only the scene"
            )
