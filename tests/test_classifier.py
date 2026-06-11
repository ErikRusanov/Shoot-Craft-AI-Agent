"""Use-case classifier: deterministic overlap and the budgeted LLM connector.

The LLM path must never fail the session — any misbehavior degrades to the
deterministic token-overlap fallback — and it must reserve through the budget,
degrading the same way when the budget refuses.
"""

from __future__ import annotations

import json
from decimal import Decimal

import httpx
import pytest

from protocols import BudgetMeter
from schemas import PaidCallKind
from services.budget import BudgetService
from services.classifier import FALLBACK_USE_CASE, TokenOverlapClassifier, best_use_case
from services.connectors import InMemoryStateStore, OpenRouterUseCaseClassifier
from services.connectors.openrouter_client import parse_usage
from services.pricing import PricingTable
from tests.openrouter_mock import scripted_client, text_completion_body

USE_CASES = ["avatar", "headshot", "formal_portrait", "dating_profile"]
MODEL = "lite-model"


def _classify_body(use_case: str, *, cost: float | None = None) -> httpx.Response:
    body = text_completion_body(json.dumps({"use_case": use_case}))
    if cost is not None:
        body["usage"] = {"prompt_tokens": 40, "completion_tokens": 3, "cost": cost}
    return httpx.Response(200, json=body)


def _meter(limit: str = "1") -> tuple[BudgetService, BudgetMeter]:
    store = InMemoryStateStore()
    pricing = PricingTable.default(generation_model="gen", lite_model=MODEL)
    svc = BudgetService(store, pricing)
    return svc, svc.meter("sess-1", limit=Decimal(limit), ttl_seconds=60)


# --- deterministic fallback ---


def test_best_use_case_matches_on_words() -> None:
    assert best_use_case("I need a professional headshot", USE_CASES) == "headshot"
    # Multi-word tokens split on the underscore.
    assert best_use_case("a formal portrait for documents", USE_CASES) == "formal_portrait"


def test_best_use_case_no_overlap_is_default() -> None:
    assert best_use_case("подводный балет", USE_CASES) == FALLBACK_USE_CASE


async def test_token_overlap_classifier_is_free() -> None:
    result = await TokenOverlapClassifier().classify(
        brief="a headshot for my resume", use_cases=USE_CASES
    )
    assert result.use_case == "headshot"
    assert result.cost == Decimal("0")


# --- LLM connector ---


async def test_llm_classifies_and_settles_cost() -> None:
    client, _ = scripted_client(_classify_body("avatar", cost=0.0003))
    classifier = OpenRouterUseCaseClassifier(client, model=MODEL)
    _svc, meter = _meter()

    result = await classifier.classify(brief="a fun avatar", use_cases=USE_CASES, meter=meter)

    assert result.use_case == "avatar"
    assert result.cost == Decimal("0.000300")
    assert result.usage is not None and result.usage.cost == Decimal("0.000300")


async def test_empty_brief_skips_the_llm() -> None:
    client, transport = scripted_client(_classify_body("avatar"))
    classifier = OpenRouterUseCaseClassifier(client, model=MODEL)

    result = await classifier.classify(brief="   ", use_cases=USE_CASES)

    assert transport.requests == []  # no paid call for an empty brief
    assert result.use_case == FALLBACK_USE_CASE


async def test_garbage_response_falls_back_to_overlap() -> None:
    client, _ = scripted_client(httpx.Response(200, json=text_completion_body("not json")))
    classifier = OpenRouterUseCaseClassifier(client, model=MODEL)

    result = await classifier.classify(brief="a headshot please", use_cases=USE_CASES)
    # Degraded to deterministic overlap, which still finds "headshot".
    assert result.use_case == "headshot"
    assert result.cost == Decimal("0")


async def test_out_of_vocabulary_value_falls_back() -> None:
    client, _ = scripted_client(_classify_body("underwater_ballet"))
    classifier = OpenRouterUseCaseClassifier(client, model=MODEL)

    result = await classifier.classify(brief="something odd", use_cases=USE_CASES)
    assert result.use_case == FALLBACK_USE_CASE  # no overlap either → fallback token


async def test_budget_refusal_degrades_without_calling() -> None:
    client, transport = scripted_client(_classify_body("avatar", cost=0.0003))
    classifier = OpenRouterUseCaseClassifier(client, model=MODEL)
    # A limit below the CLASSIFY flat estimate ($0.002) refuses the reservation.
    _svc, meter = _meter(limit="0.0001")

    result = await classifier.classify(brief="a fun avatar", use_cases=USE_CASES, meter=meter)

    assert transport.requests == []  # never called — budget refused up front
    assert result.use_case == "avatar"  # deterministic overlap still works
    assert result.cost == Decimal("0")


def test_parse_usage_reads_cost() -> None:
    usage = parse_usage({"usage": {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.0003}})
    assert usage is not None
    assert usage.cost == Decimal("0.000300")
    assert usage.prompt_tokens == 10
    assert parse_usage({"choices": []}) is None  # no usage block → None


@pytest.mark.parametrize("kind", [PaidCallKind.SLOT_FILL, PaidCallKind.CLASSIFY])
def test_paid_call_kinds_exist(kind: PaidCallKind) -> None:
    assert kind.value in {"slot_fill", "classify"}
