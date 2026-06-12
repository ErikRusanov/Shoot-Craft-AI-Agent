"""Brief parser: deterministic fallback and the budgeted LLM connector.

The deterministic parse is exactly today's classifier behavior in the new shape;
the LLM path must never fail the session — any misbehavior degrades to that
fallback — and it reserves through the budget, degrading the same way when the
budget refuses.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from decimal import Decimal

import httpx

from protocols import BudgetMeter
from services.brief_parser import SCENE_TARGET, DeterministicBriefParser, deterministic_analysis
from services.budget import BudgetService
from services.classifier import FALLBACK_USE_CASE, best_use_case
from services.connectors import InMemoryStateStore, OpenRouterBriefParser
from services.pricing import PricingTable
from tests.openrouter_mock import scripted_client, text_completion_body

USE_CASES = ["avatar", "headshot", "formal_portrait"]
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


# --- token overlap (the deterministic engine) ---


def test_best_use_case_matches_on_words() -> None:
    assert best_use_case("I need a professional headshot", USE_CASES) == "headshot"
    # Multi-word tokens split on the underscore.
    assert best_use_case("a formal portrait for documents", USE_CASES) == "formal_portrait"


def test_best_use_case_no_overlap_is_default() -> None:
    assert best_use_case("подводный балет", USE_CASES) == FALLBACK_USE_CASE


# --- deterministic fallback ---


def test_known_use_case_is_generate() -> None:
    a = deterministic_analysis("anything", "headshot", USE_CASES)
    assert a.mode == "generate"
    assert a.use_case == "headshot"
    assert a.changes == []


def test_overlap_resolves_a_curated_generate() -> None:
    a = deterministic_analysis("I need a professional headshot", None, USE_CASES)
    assert a.mode == "generate"
    assert a.use_case == "headshot"


def test_no_match_is_an_edit_with_the_whole_brief() -> None:
    a = deterministic_analysis("keep my face, make the background blue", None, USE_CASES)
    assert a.mode == "edit"
    assert a.use_case is None
    assert len(a.changes) == 1
    assert a.changes[0].target == SCENE_TARGET
    assert a.changes[0].instruction == "keep my face, make the background blue"


def test_empty_brief_is_an_edit_with_nothing_actionable() -> None:
    a = deterministic_analysis("   ", None, USE_CASES)
    assert a.mode == "edit"
    assert a.changes == []


async def test_deterministic_parser_is_free() -> None:
    result = await DeterministicBriefParser().parse(
        brief="a headshot for my resume", use_case=None, use_cases=USE_CASES
    )
    assert result.analysis.use_case == "headshot"
    assert result.cost == Decimal("0")


# --- LLM connector ---


async def test_llm_parses_an_edit_and_settles_cost() -> None:
    payload = {
        "mode": "edit",
        "use_case": None,
        "preserve": ["pose", "framing"],
        "changes": [{"target": "background", "instruction": "make it solid blue"}],
        "conflicts": [],
    }
    client, _ = scripted_client(_analysis_body(payload, cost=0.0004))
    parser = OpenRouterBriefParser(client, model=MODEL)
    _svc, meter = _meter()

    result = await parser.parse(
        brief="keep my face the same, replace the background with blue",
        use_case=None,
        use_cases=USE_CASES,
        meter=meter,
    )

    assert result.analysis.mode == "edit"
    assert result.analysis.preserve == ["pose", "framing"]
    assert [c.target for c in result.analysis.changes] == ["background"]
    assert result.cost == Decimal("0.000400")


async def test_llm_edit_with_a_use_case_is_nulled() -> None:
    # An edit never targets a curated use case, even if the model names one.
    payload = {
        "mode": "edit",
        "use_case": "headshot",
        "preserve": [],
        "changes": [{"target": "lighting", "instruction": "warmer"}],
        "conflicts": [],
    }
    client, _ = scripted_client(_analysis_body(payload))
    parser = OpenRouterBriefParser(client, model=MODEL)

    result = await parser.parse(brief="warmer light please", use_case=None, use_cases=USE_CASES)
    assert result.analysis.mode == "edit"
    assert result.analysis.use_case is None


async def test_empty_brief_skips_the_llm() -> None:
    client, transport = scripted_client(_analysis_body({"mode": "edit"}))
    parser = OpenRouterBriefParser(client, model=MODEL)

    result = await parser.parse(brief="  ", use_case=None, use_cases=USE_CASES)

    assert transport.requests == []  # no paid call for an empty brief
    assert result.analysis.mode == "edit"


async def test_garbage_response_falls_back_to_deterministic() -> None:
    client, _ = scripted_client(httpx.Response(200, json=text_completion_body("not json")))
    parser = OpenRouterBriefParser(client, model=MODEL)

    result = await parser.parse(brief="a headshot please", use_case=None, use_cases=USE_CASES)
    # Degraded to deterministic overlap, which still finds "headshot".
    assert result.analysis.mode == "generate"
    assert result.analysis.use_case == "headshot"
    assert result.cost == Decimal("0")


async def test_out_of_vocabulary_use_case_falls_back() -> None:
    payload = {
        "mode": "generate",
        "use_case": "underwater_ballet",
        "preserve": [],
        "changes": [],
        "conflicts": [],
    }
    client, _ = scripted_client(_analysis_body(payload))
    parser = OpenRouterBriefParser(client, model=MODEL)

    result = await parser.parse(brief="something odd", use_case=None, use_cases=USE_CASES)
    # The bad token triggers the fallback; deterministic overlap finds nothing
    # curated → an edit carrying the whole brief.
    assert result.analysis.mode == "edit"


async def test_budget_refusal_degrades_without_calling() -> None:
    payload: dict[str, object] = {
        "mode": "edit",
        "use_case": None,
        "preserve": [],
        "changes": [],
        "conflicts": [],
    }
    client, transport = scripted_client(_analysis_body(payload, cost=0.0004))
    parser = OpenRouterBriefParser(client, model=MODEL)
    # A limit below the CLASSIFY flat estimate ($0.002) refuses the reservation.
    _svc, meter = _meter(limit="0.0001")

    result = await parser.parse(brief="a fun edit", use_case=None, use_cases=USE_CASES, meter=meter)

    assert transport.requests == []  # never called — budget refused up front
    assert result.cost == Decimal("0")
