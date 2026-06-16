"""Brief parser: deterministic fallback and the budgeted LLM connector.

The pipeline is always edit mode. The deterministic parse puts the whole brief
as a single scene change; the LLM extracts structured changes. Any LLM
misbehavior degrades to the deterministic parse — parsing must never fail the
session.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from decimal import Decimal

import httpx

from protocols import BudgetMeter
from services.brief_parser import SCENE_TARGET, DeterministicBriefParser, deterministic_analysis
from services.budget import BudgetService
from services.connectors import InMemoryStateStore, OpenRouterBriefParser
from services.pricing import PricingTable
from tests.openrouter_mock import scripted_client, text_completion_body

MODEL = "lite-model"


def _analysis_body(payload: Mapping[str, object], *, cost: float | None = None) -> httpx.Response:
    body = text_completion_body(json.dumps(payload))
    if cost is not None:
        body["usage"] = {"prompt_tokens": 60, "completion_tokens": 20, "cost": cost}
    return httpx.Response(200, json=body)


def _meter(limit: str = "1") -> tuple[BudgetService, BudgetMeter]:
    store = InMemoryStateStore()
    pricing = PricingTable.default(generation_model="gen", lite_model=MODEL)
    svc = BudgetService(store, pricing)
    return svc, svc.meter("sess-1", limit=Decimal(limit), ttl_seconds=60)


# --- deterministic fallback ---


def test_brief_becomes_a_single_scene_change() -> None:
    a = deterministic_analysis("keep my face, make the background blue")
    assert a.mode == "edit"
    assert a.use_case is None
    assert len(a.changes) == 1
    assert a.changes[0].target == SCENE_TARGET
    assert a.changes[0].instruction == "keep my face, make the background blue"


def test_empty_brief_has_no_changes() -> None:
    a = deterministic_analysis("   ")
    assert a.mode == "edit"
    assert a.changes == []


async def test_deterministic_parser_is_free() -> None:
    result = await DeterministicBriefParser().parse(brief="a headshot for my resume")
    assert result.analysis.mode == "edit"
    assert result.cost == Decimal("0")


# --- LLM connector ---


async def test_llm_parses_changes_and_settles_cost() -> None:
    payload = {
        "preserve": ["pose", "framing"],
        "changes": [{"target": "background", "instruction": "make it solid blue"}],
        "conflicts": [],
    }
    client, _ = scripted_client(_analysis_body(payload, cost=0.0004))
    parser = OpenRouterBriefParser(client, model=MODEL)
    _svc, meter = _meter()

    result = await parser.parse(
        brief="keep my face the same, replace the background with blue",
        meter=meter,
    )

    assert result.analysis.mode == "edit"
    assert result.analysis.preserve == ["pose", "framing"]
    assert [c.target for c in result.analysis.changes] == ["background"]
    assert result.cost == Decimal("0.000400")


async def test_empty_brief_skips_the_llm() -> None:
    empty_payload: dict[str, object] = {"preserve": [], "changes": [], "conflicts": []}
    client, transport = scripted_client(_analysis_body(empty_payload))
    parser = OpenRouterBriefParser(client, model=MODEL)

    result = await parser.parse(brief="  ")

    assert transport.requests == []  # no paid call for an empty brief
    assert result.analysis.mode == "edit"


async def test_garbage_response_falls_back_to_deterministic() -> None:
    client, _ = scripted_client(httpx.Response(200, json=text_completion_body("not json")))
    parser = OpenRouterBriefParser(client, model=MODEL)

    result = await parser.parse(brief="replace background with blue")
    # Degraded to deterministic: whole brief as one scene change.
    assert result.analysis.mode == "edit"
    assert len(result.analysis.changes) == 1
    assert result.cost == Decimal("0")


async def test_budget_refusal_degrades_without_calling() -> None:
    payload: dict[str, object] = {"preserve": [], "changes": [], "conflicts": []}
    client, transport = scripted_client(_analysis_body(payload, cost=0.0004))
    parser = OpenRouterBriefParser(client, model=MODEL)
    # A limit below the CLASSIFY flat estimate ($0.002) refuses the reservation.
    _svc, meter = _meter(limit="0.0001")

    result = await parser.parse(brief="a fun edit", meter=meter)

    assert transport.requests == []  # never called — budget refused up front
    assert result.cost == Decimal("0")
