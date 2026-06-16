"""VLM-backed :class:`~protocols.inventory.InventoryExtractor`.

One vision call per reference photo: the image goes up as a data-URI part, the
model returns a structured :class:`~schemas.inventory.PhotoInventory` via
``response_format: json_schema`` (``strict``). The system prompt restricts the
output to what is *visible* — never facial geometry or identity, which stay in
the embedding's domain.

Failure policy mirrors the brief parser: any misbehavior — budget refusal,
transport failure, a 4xx, unparseable output — degrades to an empty inventory
instead of raising. The session must never fail because cataloguing did; an
empty inventory only makes the edit prompts less specific.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from protocols.budget import BudgetMeter
from protocols.inventory import InventoryResult
from schemas import PaidCallKind, PhotoInventory
from services.connectors.image_parts import image_part
from services.connectors.openrouter_client import OpenRouterClient, parse_usage

log = structlog.get_logger(__name__)

_SYSTEM_PROMPT = (
    "You catalogue exactly what is visible in one photo of a person, so a photo "
    "editor can keep everything unchanged. Report only what you can actually see — "
    "never guess, never embellish; leave a field empty when it is not visible. Use "
    "short, concrete noun phrases.\n"
    "\n"
    "For the 'pose' field, start with head yaw using EXACTLY one of these categories: "
    "'frontal (0-15°)' | 'slight turn (15-30°)' | 'three-quarter view (30-50°)' | "
    "'strong three-quarter (50-70°)' | 'near-profile (70-80°)' | 'profile (80-90°)'. "
    "Always add direction: 'to the left' or 'to the right'. "
    "Visual check: if the far ear is NOT visible → use at least 'strong three-quarter'; "
    "if the far ear IS visible but the far cheek is partially obscured → 'three-quarter view'; "
    "if both ears are clearly visible → 'slight turn' or 'frontal'. "
    "After the head yaw, add body orientation, posture, and arm/hand placement.\n"
    "\n"
    "Other fields: hands (visibility, gesture, held objects), accessories (each item with "
    "its placement, e.g. 'wedding ring on the right hand', 'white earbud in the left ear'), "
    "clothing (garment, color, fit, neckline), hair, facial_hair, framing (crop, camera "
    "distance and angle; for head-and-shoulders or half-body shots also state the shoulder "
    "span as a fraction of frame width, e.g. 'shoulders span ~75% of frame width', and how "
    "much of the lower frame is body, e.g. 'body visible to mid-chest'), "
    "lighting (light direction, quality, overall color grade), "
    "background (a one-sentence summary). Do not describe facial geometry or identity — "
    "only what sits on or around the person."
)

_TEXT_FIELDS = (
    "pose",
    "hands",
    "clothing",
    "hair",
    "facial_hair",
    "framing",
    "lighting",
    "background",
)


class _InvalidInventory(Exception):
    """Internal: the VLM's output failed validation — triggers the fallback."""


def _response_schema() -> dict[str, Any]:
    properties: dict[str, Any] = {field: {"type": "string"} for field in _TEXT_FIELDS}
    properties["accessories"] = {"type": "array", "items": {"type": "string"}}
    return {
        "type": "object",
        "properties": properties,
        "required": [*_TEXT_FIELDS, "accessories"],
        "additionalProperties": False,
    }


def _parse_inventory(body: dict[str, Any]) -> PhotoInventory:
    try:
        content = body["choices"][0]["message"]["content"]
        parsed = json.loads(content)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise _InvalidInventory(f"unparseable inventory response: {exc!r}") from exc
    if not isinstance(parsed, dict):
        raise _InvalidInventory("inventory response is not an object")

    fields: dict[str, Any] = {}
    for field in _TEXT_FIELDS:
        value = parsed.get(field)
        if not isinstance(value, str):
            raise _InvalidInventory(f"{field} is not a string")
        fields[field] = value.strip()

    raw_accessories = parsed.get("accessories")
    if not isinstance(raw_accessories, list) or not all(
        isinstance(x, str) for x in raw_accessories
    ):
        raise _InvalidInventory("accessories is not a list of strings")
    fields["accessories"] = [x.strip() for x in raw_accessories if x.strip()]

    return PhotoInventory(**fields)


class OpenRouterInventoryExtractor:
    """InventoryExtractor that asks a VLM, falling back to an empty inventory."""

    def __init__(self, client: OpenRouterClient, *, model: str) -> None:
        self._client = client
        self._model = model

    async def extract(
        self,
        image: bytes,
        *,
        meter: BudgetMeter | None = None,
    ) -> InventoryResult:
        # Reserve before the paid call; a refused budget degrades to the free
        # empty inventory rather than failing the session.
        reservation = None if meter is None else await meter.reserve(PaidCallKind.INVENTORY)
        if meter is not None and reservation is None:
            log.warning("inventory_budget_exhausted")
            return InventoryResult(inventory=PhotoInventory())
        try:
            body = await self._client.chat_completion(self._payload(image))
            inventory = _parse_inventory(body)
            usage = parse_usage(body)
            if reservation is None:
                return InventoryResult(inventory=inventory)
            cost = await reservation.settle(usage.cost if usage is not None else None)
            return InventoryResult(inventory=inventory, usage=usage, cost=cost)
        # Deliberately broad: whatever the VLM path did wrong, degrade to the
        # empty inventory. The provider delivered no usable result, so the
        # reservation is refunded.
        except Exception as exc:
            if reservation is not None:
                await reservation.cancel()
            log.warning("inventory_llm_fallback", error=repr(exc))
            return InventoryResult(inventory=PhotoInventory())

    def _payload(self, image: bytes) -> dict[str, Any]:
        return {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": [image_part(image)]},
            ],
            "temperature": 0.0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "photo_inventory",
                    "strict": True,
                    "schema": _response_schema(),
                },
            },
        }
