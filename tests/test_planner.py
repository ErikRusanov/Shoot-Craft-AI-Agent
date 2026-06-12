"""Step planner: deterministic decomposition, budget trim, and the LLM connector.

The deterministic plan is one change = one step (a single step for generate); the
LLM merges/splits and must never fail the session — any misbehavior degrades to
the deterministic plan. fit_to_budget trims the tail explicitly, never silently.
"""

from __future__ import annotations

import json
from decimal import Decimal

import httpx

from protocols import BudgetMeter
from schemas import BriefAnalysis, Change
from services.budget import BudgetService
from services.connectors import InMemoryStateStore, OpenRouterStepPlanner
from services.planner import DeterministicStepPlanner, deterministic_steps, fit_to_budget
from services.pricing import PricingTable
from tests.openrouter_mock import scripted_client, text_completion_body

MODEL = "lite-model"


def _meter(limit: str = "1") -> tuple[BudgetService, BudgetMeter]:
    store = InMemoryStateStore()
    pricing = PricingTable.default(generation_model="gen", lite_model=MODEL)
    svc = BudgetService(store, pricing)
    return svc, svc.meter("sess-1", limit=Decimal(limit), ttl_seconds=60)


def _edit(*targets: str) -> BriefAnalysis:
    return BriefAnalysis(
        mode="edit",
        use_case=None,
        changes=[Change(target=t, instruction=f"change the {t}") for t in targets],
    )


def _steps_body(steps: list[dict[str, object]], *, cost: float | None = None) -> httpx.Response:
    body = text_completion_body(json.dumps({"steps": steps}))
    if cost is not None:
        body["usage"] = {"prompt_tokens": 50, "completion_tokens": 40, "cost": cost}
    return httpx.Response(200, json=body)


# --- deterministic ---


def test_generate_is_a_single_step() -> None:
    analysis = BriefAnalysis(mode="generate", use_case="headshot")
    steps = deterministic_steps(analysis)
    assert len(steps) == 1
    assert steps[0].title == "headshot"


def test_edit_is_one_step_per_change() -> None:
    steps = deterministic_steps(_edit("background", "lighting", "clothing"))
    assert [s.n for s in steps] == [1, 2, 3]
    assert [s.targets for s in steps] == [["background"], ["lighting"], ["clothing"]]


async def test_deterministic_planner_is_free() -> None:
    result = await DeterministicStepPlanner().plan(analysis=_edit("background"))
    assert len(result.steps) == 1
    assert result.cost == Decimal("0")


# --- fit_to_budget ---


def test_fit_keeps_all_when_affordable() -> None:
    steps = deterministic_steps(_edit("a", "b"))
    kept, note = fit_to_budget(steps, 2)
    assert note is None
    assert all(s.status == "pending" for s in kept)


def test_fit_trims_the_tail_with_a_note() -> None:
    steps = deterministic_steps(_edit("a", "b", "c"))
    kept, note = fit_to_budget(steps, 1)
    assert [s.status for s in kept] == ["pending", "skipped", "skipped"]
    assert note is not None and "1 of 3" in note


def test_fit_zero_skips_everything() -> None:
    steps = deterministic_steps(_edit("a", "b"))
    kept, note = fit_to_budget(steps, 0)
    assert all(s.status == "skipped" for s in kept)
    assert note is not None


# --- LLM connector ---


async def test_llm_merges_and_splits_and_settles_cost() -> None:
    body = _steps_body(
        [
            {
                "title": "scene",
                "instruction": "blue bg, warm light",
                "targets": ["background", "lighting"],
            },
            {"title": "wardrobe", "instruction": "red t-shirt", "targets": ["clothing"]},
        ],
        cost=0.0003,
    )
    client, _ = scripted_client(body)
    planner = OpenRouterStepPlanner(client, model=MODEL)
    _svc, meter = _meter()

    result = await planner.plan(analysis=_edit("background", "lighting", "clothing"), meter=meter)

    assert [s.targets for s in result.steps] == [["background", "lighting"], ["clothing"]]
    assert [s.n for s in result.steps] == [1, 2]
    assert result.cost == Decimal("0.000300")


async def test_single_change_skips_the_llm() -> None:
    client, transport = scripted_client(
        _steps_body([{"title": "x", "instruction": "y", "targets": ["a"]}])
    )
    planner = OpenRouterStepPlanner(client, model=MODEL)

    result = await planner.plan(analysis=_edit("a"))
    assert transport.requests == []  # one change → deterministic, no call
    assert len(result.steps) == 1


async def test_generate_skips_the_llm() -> None:
    client, transport = scripted_client(
        _steps_body([{"title": "x", "instruction": "y", "targets": []}])
    )
    planner = OpenRouterStepPlanner(client, model=MODEL)

    result = await planner.plan(analysis=BriefAnalysis(mode="generate", use_case="avatar"))
    assert transport.requests == []
    assert len(result.steps) == 1


async def test_dropped_target_falls_back_to_deterministic() -> None:
    # The LLM covers only one of two targets → invalid → deterministic plan.
    client, _ = scripted_client(
        _steps_body([{"title": "x", "instruction": "y", "targets": ["background"]}])
    )
    planner = OpenRouterStepPlanner(client, model=MODEL)

    result = await planner.plan(analysis=_edit("background", "lighting"))
    assert [s.targets for s in result.steps] == [["background"], ["lighting"]]  # deterministic


async def test_invented_target_falls_back() -> None:
    client, _ = scripted_client(
        _steps_body(
            [{"title": "x", "instruction": "y", "targets": ["background", "lighting", "halo"]}]
        )
    )
    planner = OpenRouterStepPlanner(client, model=MODEL)

    result = await planner.plan(analysis=_edit("background", "lighting"))
    assert len(result.steps) == 2  # deterministic one-per-change


async def test_budget_refusal_degrades_without_calling() -> None:
    client, transport = scripted_client(
        _steps_body(
            [{"title": "x", "instruction": "y", "targets": ["background", "lighting"]}], cost=0.0003
        )
    )
    planner = OpenRouterStepPlanner(client, model=MODEL)
    _svc, meter = _meter(limit="0.0001")  # below the SLOT_FILL flat estimate

    result = await planner.plan(analysis=_edit("background", "lighting"), meter=meter)
    assert transport.requests == []
    assert result.cost == Decimal("0")
