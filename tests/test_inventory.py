"""Inventory extractor: the empty fallback and the budgeted VLM connector.

The inventory only makes edit prompts more specific — it must never fail a
session. Any misbehavior on the VLM path (budget refusal, transport failure,
unparseable output) degrades to an empty inventory, the same shape the free
fallback returns.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from decimal import Decimal

import httpx

from protocols import BudgetMeter
from schemas import PaidCallKind, PhotoInventory
from services.budget import BudgetService
from services.connectors import InMemoryStateStore, OpenRouterInventoryExtractor
from services.inventory import EmptyInventoryExtractor
from services.pricing import PricingTable
from tests.openrouter_mock import PIXEL_PNG, scripted_client, text_completion_body

MODEL = "vision-model"

FULL_PAYLOAD: dict[str, object] = {
    "pose": "standing square to the camera, hands together at the waist",
    "hands": "fingertips touching in a relaxed triangle",
    "accessories": ["wedding ring on the right hand", "white earbud in the left ear"],
    "clothing": "beige short-sleeve t-shirt, relaxed fit, crew neckline",
    "hair": "short wavy light-brown hair",
    "facial_hair": "short beard and moustache",
    "framing": "waist-up, eye-level, slightly off-center",
    "lighting": "soft window light from the left, neutral grade",
    "background": "a dim living room with shelving and a pendant lamp",
}


def _inventory_body(payload: Mapping[str, object], *, cost: float | None = None) -> httpx.Response:
    body = text_completion_body(json.dumps(payload))
    if cost is not None:
        body["usage"] = {"prompt_tokens": 1200, "completion_tokens": 150, "cost": cost}
    return httpx.Response(200, json=body)


def _meter(limit: str = "1") -> BudgetMeter:
    store = InMemoryStateStore()
    pricing = PricingTable.default(generation_model="gen", lite_model="lite")
    return BudgetService(store, pricing).meter("sess-1", limit=Decimal(limit), ttl_seconds=60)


# --- schema ---


def test_empty_inventory_knows_it_is_empty() -> None:
    assert PhotoInventory().is_empty()
    assert not PhotoInventory(accessories=["ring"]).is_empty()
    assert not PhotoInventory(pose="standing").is_empty()


# --- deterministic fallback ---


async def test_empty_extractor_is_free_and_empty() -> None:
    result = await EmptyInventoryExtractor().extract(PIXEL_PNG)
    assert result.inventory.is_empty()
    assert result.cost == Decimal("0")
    assert result.usage is None


# --- VLM connector ---


async def test_vlm_extracts_and_settles_cost() -> None:
    client, transport = scripted_client(_inventory_body(FULL_PAYLOAD, cost=0.003))
    extractor = OpenRouterInventoryExtractor(client, model=MODEL)
    meter = _meter()

    result = await extractor.extract(PIXEL_PNG, meter=meter)

    assert result.inventory.clothing.startswith("beige")
    assert result.inventory.accessories == [
        "wedding ring on the right hand",
        "white earbud in the left ear",
    ]
    assert result.cost == Decimal("0.003000")
    # The photo went up as a data-URI image part under the cataloguing prompt.
    sent = json.loads(transport.requests[0].content)
    assert sent["model"] == MODEL
    part = sent["messages"][1]["content"][0]
    assert part["type"] == "image_url"
    assert part["image_url"]["url"].startswith("data:image/png;base64,")


async def test_garbage_response_falls_back_and_refunds() -> None:
    client, _ = scripted_client(httpx.Response(200, json=text_completion_body("not json")))
    extractor = OpenRouterInventoryExtractor(client, model=MODEL)
    # Room for exactly one INVENTORY reservation ($0.005): if the failed call
    # were not refunded, the follow-up reserve below would be refused.
    meter = _meter(limit="0.005")

    result = await extractor.extract(PIXEL_PNG, meter=meter)

    assert result.inventory.is_empty()
    assert result.cost == Decimal("0")
    follow_up = await meter.reserve(PaidCallKind.INVENTORY)
    assert follow_up is not None
    await follow_up.cancel()


async def test_wrong_shape_falls_back_to_empty() -> None:
    payload = dict(FULL_PAYLOAD, accessories="a ring")  # not a list
    client, _ = scripted_client(_inventory_body(payload))
    extractor = OpenRouterInventoryExtractor(client, model=MODEL)

    result = await extractor.extract(PIXEL_PNG)
    assert result.inventory.is_empty()


async def test_transport_failure_falls_back_to_empty() -> None:
    client, _ = scripted_client(httpx.ConnectError("down"))
    extractor = OpenRouterInventoryExtractor(client, model=MODEL)

    result = await extractor.extract(PIXEL_PNG)
    assert result.inventory.is_empty()
    assert result.cost == Decimal("0")


async def test_budget_refusal_degrades_without_calling() -> None:
    client, transport = scripted_client(_inventory_body(FULL_PAYLOAD, cost=0.003))
    extractor = OpenRouterInventoryExtractor(client, model=MODEL)
    # A limit below the INVENTORY flat estimate ($0.005) refuses the reservation.
    meter = _meter(limit="0.0001")

    result = await extractor.extract(PIXEL_PNG, meter=meter)

    assert transport.requests == []  # never called — budget refused up front
    assert result.inventory.is_empty()
    assert result.cost == Decimal("0")
