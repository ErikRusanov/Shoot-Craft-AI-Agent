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
from collections.abc import Mapping, Sequence
from typing import NamedTuple

from schemas import Generation, PhotoInventory, Preset

_PLACEHOLDER = re.compile(r"\{(\w+)\}")
_EXCLUSION_PREFIX = "Strictly avoid: "
# The locked-attribute clause: deterministic, placed after the writer's body and
# stated as overriding it, so a lock wins over anything the body says (passport
# white background beats a user's "make it green"). Rendered by the builder, never
# by the writer — the writer only sees lock values as informational context.
_LOCK_PREFIX = "These must hold exactly, overriding any conflicting detail above: "

# --- Edit-mode lock block ---------------------------------------------------
# The deterministic wrapper that makes a chained edit conservative: enumerate
# everything that must survive ("LOCKED … copy pixel-for-pixel"), scope the step
# to a single change ("the ONLY change allowed"), and pin the face texture.
# Rendered by the builder, never by the writer — exactly like the identity and
# exclusion blocks.

# The face/skin/body lock is unconditional: it opens every enumeration and is
# never excludable, so an empty inventory still locks the person (the
# degradation path is generic, not absent).
_GENERIC_PERSON_LOCK = (
    "the person — entire face, every facial feature, expression, eyes, nose, "
    "mouth, jaw, skin texture and skin tone, hair, body, and head angle"
)
_LOCKED_PREFIX = (
    "LOCKED — the following must be copied exactly from the provided image, unchanged: "
)
_LOCKED_SUFFIX = (
    " Do NOT regenerate, repaint, smooth, retouch, denoise or beautify any locked "
    "part — copy it pixel-for-pixel from the input image. Treat the person as "
    "untouchable. Do NOT reorient the face, head, or gaze — preserve the exact "
    "head yaw, pitch, and roll from the reference. Never normalize or correct the "
    "head position toward a more frontal view. Do NOT resize the head, narrow the "
    "shoulders, or alter the head-to-shoulder width ratio — preserve the exact body "
    "proportions and shoulder span from the reference image. Exception: if the step "
    "changes the background or ambient lighting, natural rim light or color spill "
    "from the new light source may fall on the subject's hair outline, shoulders and "
    "clothing edges — that is physics, not a change to the person."
)
_ONLY_CHANGE_PREFIX = "The ONLY change allowed in this edit is: "
_TEXTURE_BLOCK_HEAD = (
    "Keep the face at the exact original sharpness, grain and raw skin texture — no "
    "plastic or waxy skin, no airbrushing, no blur on the face. "
)
_TEXTURE_BLOCK_TAIL = (
    " and keep shadows, reflections and grain consistent with the rest of the photo."
)


def _texture_block(lighting: str | None) -> str:
    """Integration block, optionally anchored to the photo's extracted lighting."""
    if lighting and lighting.strip():
        mid = f"Where the change meets the scene, match the {lighting.strip()}"
    else:
        mid = "Where the change meets the scene, match the existing light direction"
    return _TEXTURE_BLOCK_HEAD + mid + _TEXTURE_BLOCK_TAIL


_APPLIED_SUFFIX = " (already final — keep it exactly)"

# Inventory fields in render order, with their human labels. Only the fields in
# _FIELD_REGIONS are ever excluded from the lock (when the step edits them);
# pose, hands and facial hair stay locked no matter what the step targets —
# they sit too close to the identity to ever unlock by keyword.
_FIELD_LABELS: tuple[tuple[str, str], ...] = (
    ("pose", "the pose"),
    # Framing (shoulder span, scale, crop) is second — composition drift is as
    # identity-destructive as pose drift and must be locked early in the list.
    ("framing", "the framing"),
    ("hands", "the hands"),
    ("clothing", "the clothing"),
    ("hair", "the hair"),
    ("facial_hair", "the facial hair"),
    ("lighting", "the lighting"),
    ("background", "the background"),
)
# Synonyms a step target may use for an editable inventory region. Intentionally
# a simple keyword map: a miss only over-locks (the step instruction still wins
# as "the ONLY change allowed"), never under-locks. If it misfires in practice,
# promote it to a planner-emitted `unlocks` list on EditStep.
_FIELD_REGIONS: dict[str, frozenset[str]] = {
    "background": frozenset({"background", "backdrop", "setting", "scene", "wall"}),
    "lighting": frozenset({"lighting", "light", "grade", "tone", "color", "colour"}),
    "clothing": frozenset(
        {
            "clothing",
            "clothes",
            "shirt",
            "t-shirt",
            "tshirt",
            "outfit",
            "attire",
            "garment",
            "top",
            "jacket",
            "dress",
            "suit",
        }
    ),
    "hair": frozenset({"hair", "hairstyle", "haircut"}),
    "framing": frozenset({"framing", "crop", "zoom"}),
}
# Tokens too generic to identify an accessory: shared placement words must not
# unlock "wedding ring on the right hand" because the instruction says "in the
# left ear", and shared colors must not unlock "white earbud" because the step
# asks for a "white t-shirt" — only the accessory's own nouns count.
_ACCESSORY_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "in",
        "on",
        "with",
        "and",
        "or",
        "of",
        "to",
        "at",
        "his",
        "her",
        "their",
        "left",
        "right",
        "hand",
        "ear",
        "ears",
        "wrist",
        "neck",
        "head",
        "finger",
        "new",
        "existing",
        "white",
        "black",
        "beige",
        "blue",
        "red",
        "green",
        "grey",
        "gray",
        "brown",
        "pink",
        "yellow",
        "orange",
        "purple",
        "golden",
        "gold",
        "silver",
        "dark",
        "light",
        "navy",
    }
)
_TOKEN = re.compile(r"[a-z][a-z-]{2,}")

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


class FreeFormRejectedError(ValueError):
    """Free-form slot text read as a prompt injection.

    Its own type (not a bare ``ValueError``) because the orchestration reacts
    differently: a vocabulary violation is a programming/filler bug and crashes
    the run, while this error means "re-ask the user for a scene description".
    """


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
    _validate_slots(preset, slots, sanitize_freeform=True)
    structure = _PLACEHOLDER.sub(lambda m: slots[m.group(1)], preset.prompt_structure)

    parts = [preset.identity_instruction, structure]
    if addendum.strip():
        parts.append(addendum.strip())
    parts.append(_EXCLUSION_PREFIX + preset.negative_prompt)

    text = "\n\n".join(parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return BuiltPrompt(text=text, prompt_hash=digest, params=preset.generation)


def fill_template(preset: Preset, slots: Mapping[str, str], *, addendum: str = "") -> str:
    """The deterministic body — the filled ``prompt_structure``, no frozen wrap.

    This is what :class:`~services.prompt_writer.DeterministicPromptWriter` returns
    and what the LLM writer falls back to: the same substitution
    :func:`build_prompt` does, minus the identity/exclusion blocks (those are the
    builder's to add in :func:`assemble_prompt`). Slots are validated exactly as
    the legacy path validates them.
    """
    # Structural validation only — the body's free-form text is sanitized once,
    # downstream, by :func:`assemble_prompt`, not here (else the scene would be
    # checked twice and could reject early before assembly).
    _validate_slots(preset, slots, sanitize_freeform=False)
    structure = _PLACEHOLDER.sub(lambda m: slots[m.group(1)], preset.prompt_structure)
    if addendum.strip():
        return f"{structure}\n\n{addendum.strip()}"
    return structure


def assemble_prompt(
    preset: Preset, body: str, *, locks: Mapping[str, str] | None = None
) -> BuiltPrompt:
    """Assemble the final prompt around a writer-composed ``body``.

    The new assembly path (the writer's, not the template's): ``identity (frozen)
    + body (writer, sanitized) + locks (deterministic) + exclusion (frozen)``. The
    body is sanitized with the same injection guards the free-form slot used, so a
    misbehaving writer cannot reach past the scene to edit the identity block or
    override the exclusions. Locked attributes are rendered here, deterministically,
    so they win over the body. ``prompt_hash`` is the same sha256-over-text the
    legacy path produces, so reproducibility is unchanged.
    """
    _reject_body_injection(preset, body)
    parts = [preset.identity_instruction, body.strip()]
    if locks:
        rendered = "; ".join(f"{name} — {value}" for name, value in sorted(locks.items()))
        parts.append(f"{_LOCK_PREFIX}{rendered}.")
    parts.append(_EXCLUSION_PREFIX + preset.negative_prompt)

    text = "\n\n".join(parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return BuiltPrompt(text=text, prompt_hash=digest, params=preset.generation)


def _content_tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text.lower()) if t not in _ACCESSORY_STOPWORDS}


def _is_injection(text: str) -> bool:
    lowered = text.lower()
    return any(pattern.search(lowered) for pattern in _INJECTION_PATTERNS)


def edit_lock_items(
    inventory: PhotoInventory | None,
    *,
    preserve: Sequence[str] = (),
    applied: Sequence[str] = (),
    edited_targets: Sequence[str] = (),
    edited_text: str = "",
) -> list[str]:
    """The lock enumeration for one edit step, most identity-critical first.

    The generic person lock always opens the list (the whole degradation path
    when ``inventory`` is empty/None). ``edited_targets``/``edited_text`` cover
    the current step **and** every completed step: the inventory describes the
    *original* photo, so a region an earlier step already changed must not be
    re-locked at its stale value — its ``applied`` phrase (the new value) locks
    it instead. Fields are excluded via the region keyword map; an accessory is
    excluded when it shares a content token with the edited text ("replace the
    existing earbud…" unlocks the earbud item, never the ring). ``preserve``
    entries (user-derived) append after the inventory.

    Items are *enumerated text*, not instructions: a preserve/inventory entry
    that reads as an injection is silently dropped — the generic person lock
    already covers the face, so dropping costs specificity, not safety, and a
    benign "don't change my face" must not fail the step.
    """
    items = [_GENERIC_PERSON_LOCK]
    target_tokens = {t.strip().lower() for t in edited_targets if t.strip()}
    edited_regions = {
        field
        for field, synonyms in _FIELD_REGIONS.items()
        if any(token in synonyms for token in target_tokens)
    }
    unlock_tokens = _content_tokens(edited_text) | _content_tokens(" ".join(edited_targets))

    if inventory is not None:
        for field, label in _FIELD_LABELS:
            if field in edited_regions:
                continue
            value = getattr(inventory, field).strip()
            if value:
                items.append(f"{label}: {value}")
        for accessory in inventory.accessories:
            if _content_tokens(accessory) & unlock_tokens:
                continue  # this accessory is what the step edits
            items.append(accessory)

    items += [p.strip() for p in preserve if p.strip()]
    items += [f"{a.strip()}{_APPLIED_SUFFIX}" for a in applied if a.strip()]
    return [item for item in items if not _is_injection(item)]


def assemble_edit_prompt(
    preset: Preset,
    body: str,
    *,
    only_change: str,
    lock_items: Sequence[str],
    locks: Mapping[str, str] | None = None,
    lighting: str | None = None,
) -> BuiltPrompt:
    """Assemble the edit-mode prompt: lock block + single-change scope + body.

    The edit-mode sibling of :func:`assemble_prompt` — same frozen blocks, same
    sanitization, plus the deterministic lock enumeration and the texture
    block. ``only_change`` is the step's instruction (user-derived, so it goes
    through the injection guards; the caller degrades to :func:`assemble_prompt`
    on rejection). Layout::

        identity (frozen)
        LOCKED — … copy pixel-for-pixel … untouchable.
        The ONLY change allowed in this edit is: …
        body (writer, sanitized)
        face texture + integration block
        preset locks (deterministic, win over the body)
        Strictly avoid: … (frozen)
    """
    change = only_change.strip()
    if _is_injection(change):
        raise FreeFormRejectedError(
            f"preset {preset.id!r}: step instruction rejected — it reads as an "
            f"instruction to alter identity or override the prompt"
        )
    _reject_body_injection(preset, body)

    items = list(lock_items) or [_GENERIC_PERSON_LOCK]
    parts = [
        preset.identity_instruction,
        _LOCKED_PREFIX + "; ".join(items) + "." + _LOCKED_SUFFIX,
        f"{_ONLY_CHANGE_PREFIX}{change.rstrip('.')}.",
        body.strip(),
        _texture_block(lighting),
    ]
    if locks:
        rendered = "; ".join(f"{name} — {value}" for name, value in sorted(locks.items()))
        parts.append(f"{_LOCK_PREFIX}{rendered}.")
    parts.append(_EXCLUSION_PREFIX + preset.negative_prompt)

    text = "\n\n".join(parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return BuiltPrompt(text=text, prompt_hash=digest, params=preset.generation)


def _reject_body_injection(preset: Preset, body: str) -> None:
    """Reject a writer-composed body that reads as a prompt injection.

    Same guards as the free-form slot, applied to the whole body: the writer
    describes a scene/edit and must never carry instructions that reach past it to
    alter the frozen identity block or override the exclusions.
    """
    lowered = body.lower()
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(lowered):
            raise FreeFormRejectedError(
                f"preset {preset.id!r}: composed body rejected — it reads as an "
                f"instruction to alter identity or override the prompt"
            )


def _validate_slots(preset: Preset, slots: Mapping[str, str], *, sanitize_freeform: bool) -> None:
    unknown = set(slots) - set(preset.slots)
    if unknown:
        raise ValueError(f"preset {preset.id!r}: unknown slots {sorted(unknown)}")

    missing = set(_PLACEHOLDER.findall(preset.prompt_structure)) - set(slots)
    if missing:
        raise ValueError(f"preset {preset.id!r}: missing values for slots {sorted(missing)}")

    for name, value in slots.items():
        enum = preset.slots[name].enum
        if enum is None:
            # Free-form slot: no vocabulary to check. The legacy path sanitizes the
            # text here; the writer path defers it to assemble_prompt on the body.
            if sanitize_freeform:
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
            raise FreeFormRejectedError(
                f"preset {preset.id!r}: slot {name!r} free-form text rejected — it reads "
                f"as an instruction to alter identity or override the prompt; "
                f"describe only the scene"
            )
