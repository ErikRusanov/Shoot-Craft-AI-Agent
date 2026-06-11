"""LLM-backed :class:`~protocols.slot_filler.SlotFiller` via a cheap text model.

The LLM's authority is exactly the port's: pick a value for every slot from the
preset's **own vocabulary** (plus the one free-form slot and a short addendum).
It is shown the slot dictionary, the user's answer and the safe frame metrics —
**never** ``identity_instruction``, ``prompt_structure`` or ``negative_prompt``,
so even a fully compromised model cannot see (let alone edit) the frozen blocks,
and the prompt builder re-validates every value anyway.

Failure policy: this filler must degrade, not break. Any LLM misbehavior —
transport failure after retries, a 4xx, unparseable JSON, a value outside the
slot's enum — falls back to :class:`~services.slot_filler.DefaultSlotFiller`
instead of raising. Structured output (``response_format: json_schema`` with
``strict``) makes the happy path conform; the fallback covers providers that
ignore it.

When there is neither a user answer nor photo analysis the LLM has nothing the
deterministic filler doesn't, so the call (and its cost) is skipped outright.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from protocols.budget import BudgetMeter
from protocols.slot_filler import SlotFill, SlotFiller
from schemas import FrameMetrics, PaidCallKind, Preset
from services.connectors.openrouter_client import OpenRouterClient, parse_usage
from services.slot_filler import DefaultSlotFiller

log = structlog.get_logger(__name__)

_SYSTEM_PROMPT = (
    "You resolve styling slots for a photo-generation prompt. For every slot that "
    "lists allowed options, choose exactly one of them, the best fit for the user's "
    "answer and the photo metrics. A slot without options is a free-form scene "
    "description: copy the user's own words into it verbatim, unchanged. The "
    "addendum is at most one short sentence of extra scene detail (lighting, mood); "
    "leave it empty unless the user's answer clearly asks for it. Never describe or "
    "modify the person's face or identity."
)

# Hard ceilings via the schema, not trust: the addendum lands in the prompt
# after the frozen structure, so a rambling model must be cut off by shape.
_FREEFORM_MAX_LEN = 300
_ADDENDUM_MAX_LEN = 200


class _InvalidLLMFill(Exception):
    """Internal: the LLM's output failed validation — triggers the fallback."""


def _response_schema(preset: Preset) -> dict[str, Any]:
    """Strict JSON schema mirroring the preset's slot dictionary."""
    properties: dict[str, Any] = {}
    for name, slot in preset.slots.items():
        if slot.enum is not None:
            properties[name] = {"type": "string", "enum": [str(o) for o in slot.enum]}
        else:
            properties[name] = {"type": "string", "maxLength": _FREEFORM_MAX_LEN}
    return {
        "type": "object",
        "properties": {
            "slots": {
                "type": "object",
                "properties": properties,
                "required": sorted(preset.slots),
                "additionalProperties": False,
            },
            "addendum": {"type": "string", "maxLength": _ADDENDUM_MAX_LEN},
        },
        "required": ["slots", "addendum"],
        "additionalProperties": False,
    }


def _slot_catalog(preset: Preset) -> dict[str, Any]:
    """The slot dictionary as the LLM sees it — names, options, defaults only."""
    return {
        name: {
            "options": [str(o) for o in slot.enum] if slot.enum is not None else None,
            "default": None if slot.default is None else str(slot.default),
            "asked_to_user": slot.ask,
        }
        for name, slot in preset.slots.items()
    }


def _parse_fill(preset: Preset, body: dict[str, Any]) -> SlotFill:
    try:
        content = body["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        raw_slots = parsed["slots"]
        addendum = parsed.get("addendum", "")
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise _InvalidLLMFill(f"unparseable LLM response: {exc!r}") from exc

    if not isinstance(raw_slots, dict) or set(raw_slots) != set(preset.slots):
        raise _InvalidLLMFill("LLM did not return exactly the preset's slots")

    slots: dict[str, str] = {}
    for name, value in raw_slots.items():
        if not isinstance(value, str) or not value.strip():
            raise _InvalidLLMFill(f"slot {name!r} is not a non-empty string")
        enum = preset.slots[name].enum
        if enum is not None and value not in {str(o) for o in enum}:
            raise _InvalidLLMFill(f"slot {name!r} value is outside the preset vocabulary")
        slots[name] = value
    if not isinstance(addendum, str):
        raise _InvalidLLMFill("addendum is not a string")
    return SlotFill(slots=slots, addendum=addendum.strip()[:_ADDENDUM_MAX_LEN])


def _verbatim_freeform(
    preset: Preset, slots: dict[str, str], user_answer: str | None
) -> dict[str, str]:
    """Replace every free-form (enum-less) slot with the user's own words.

    The LLM resolves enum slots and the addendum; a free-form slot must carry the
    user's text *verbatim*, never the model's paraphrase — paraphrasing is exactly
    how "do not change photo, make background blue light" became a generic scene.
    The prompt builder's injection sanitizer is the only filter on this text.
    """
    out = dict(slots)
    for name, slot in preset.slots.items():
        if slot.enum is not None:
            continue
        if slot.ask and user_answer is not None:
            out[name] = user_answer
        elif slot.default is not None:
            out[name] = str(slot.default)
    return out


class OpenRouterSlotFiller:
    """SlotFiller that asks a cheap LLM, falling back to the deterministic filler."""

    def __init__(
        self,
        client: OpenRouterClient,
        *,
        model: str,
        fallback: SlotFiller | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._fallback = fallback if fallback is not None else DefaultSlotFiller()

    async def fill(
        self,
        *,
        preset: Preset,
        user_answer: str | None,
        photo_analysis: FrameMetrics | None,
        meter: BudgetMeter | None = None,
    ) -> SlotFill:
        if user_answer is None and photo_analysis is None:
            # Nothing to interpret — the LLM would only restate the defaults.
            return await self._fallback.fill(
                preset=preset, user_answer=user_answer, photo_analysis=photo_analysis, meter=meter
            )
        # Reserve the dollar slot before the paid call; a refused budget degrades
        # to the free deterministic fill rather than failing the session.
        reservation = None if meter is None else await meter.reserve(PaidCallKind.SLOT_FILL)
        if meter is not None and reservation is None:
            log.warning("slot_filler_budget_exhausted", preset_id=preset.id)
            return await self._fallback.fill(
                preset=preset, user_answer=user_answer, photo_analysis=photo_analysis, meter=meter
            )
        try:
            payload = self._payload(preset, user_answer, photo_analysis)
            body = await self._client.chat_completion(payload)
            fill = _parse_fill(preset, body)
            # Free-form slots are taken verbatim from the user, never the LLM's
            # rephrasing — even when structured output complied.
            fill = fill._replace(slots=_verbatim_freeform(preset, fill.slots, user_answer))
            usage = parse_usage(body)
            if reservation is None:
                return fill
            cost = await reservation.settle(usage.cost if usage is not None else None)
            return fill._replace(usage=usage, cost=cost)
        # Deliberately broad: whatever the LLM path did wrong, the session
        # degrades to the deterministic filler instead of breaking. The provider
        # did not deliver a usable result, so the reservation is refunded.
        except Exception as exc:
            if reservation is not None:
                await reservation.cancel()
            log.warning("slot_filler_llm_fallback", preset_id=preset.id, error=repr(exc))
            return await self._fallback.fill(
                preset=preset, user_answer=user_answer, photo_analysis=photo_analysis, meter=meter
            )

    def _payload(
        self, preset: Preset, user_answer: str | None, photo_analysis: FrameMetrics | None
    ) -> dict[str, Any]:
        # FrameMetrics is photo statistics, not biometrics — safe to surface.
        user_message = {
            "slots": _slot_catalog(preset),
            "user_answer": user_answer,
            "photo_metrics": photo_analysis.model_dump() if photo_analysis else None,
        }
        return {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_message, ensure_ascii=False)},
            ],
            "temperature": 0.0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "slot_fill",
                    "strict": True,
                    "schema": _response_schema(preset),
                },
            },
        }
