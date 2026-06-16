"""Step planner: deterministic decomposition and the LLM connector.

The deterministic plan is one change = one step; the LLM merges/splits and must
never fail the session — any misbehavior degrades to the deterministic plan.
The plan is never trimmed to the budget: spending is greedy and a short budget
simply ships a partial chain.
"""

from __future__ import annotations

import json
from decimal import Decimal

import httpx

from protocols import BudgetMeter
from schemas import BriefAnalysis, Change, PhotoInventory
from services.budget import BudgetService
from services.connectors import InMemoryStateStore, OpenRouterStepPlanner
from services.planner import DeterministicStepPlanner, deterministic_steps
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


def test_edit_is_one_step_per_change() -> None:
    steps = deterministic_steps(_edit("background", "lighting", "clothing"))
    assert [s.n for s in steps] == [1, 2, 3]
    assert [s.targets for s in steps] == [["background"], ["lighting"], ["clothing"]]


def test_edit_order_runs_scene_first_face_adjacent_last() -> None:
    # A brief listing the risky changes first still plans scene-level work
    # first and the face-adjacent accessory last; unknown targets sit in the
    # middle, and the brief's order survives within a rank (stable sort).
    steps = deterministic_steps(_edit("accessory", "clothing", "tattoo", "background"))
    assert [s.targets for s in steps] == [["background"], ["clothing"], ["tattoo"], ["accessory"]]
    assert [s.n for s in steps] == [1, 2, 3, 4]


async def test_deterministic_planner_is_free() -> None:
    result = await DeterministicStepPlanner().plan(analysis=_edit("background"))
    assert len(result.steps) == 1
    assert result.cost == Decimal("0")


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


async def test_llm_parses_applied_and_sends_the_inventory() -> None:
    body = _steps_body(
        [
            {
                "title": "bg",
                "instruction": "a dark stage backdrop",
                "targets": ["background"],
                "applied": "the new dark stage backdrop",
            },
            {
                "title": "shirt",
                "instruction": "a white tee",
                "targets": ["clothing"],
                "applied": "",  # tolerated: the loop's generic fallback covers it
            },
        ]
    )
    client, transport = scripted_client(body)
    planner = OpenRouterStepPlanner(client, model=MODEL)

    result = await planner.plan(
        analysis=_edit("background", "clothing"),
        inventory=PhotoInventory(clothing="beige t-shirt"),
    )

    assert [s.applied for s in result.steps] == ["the new dark stage backdrop", ""]
    sent = json.loads(transport.requests[0].content)
    user = json.loads(sent["messages"][1]["content"])
    assert user["photo_inventory"]["clothing"] == "beige t-shirt"


async def test_missing_inventory_is_sent_as_null() -> None:
    body = _steps_body(
        [
            {"title": "bg", "instruction": "x", "targets": ["background"]},
            {"title": "shirt", "instruction": "y", "targets": ["clothing"]},
        ]
    )
    client, transport = scripted_client(body)
    planner = OpenRouterStepPlanner(client, model=MODEL)

    await planner.plan(analysis=_edit("background", "clothing"))

    sent = json.loads(transport.requests[0].content)
    user = json.loads(sent["messages"][1]["content"])
    assert user["photo_inventory"] is None


async def test_single_change_skips_the_llm() -> None:
    client, transport = scripted_client(
        _steps_body([{"title": "x", "instruction": "y", "targets": ["a"]}])
    )
    planner = OpenRouterStepPlanner(client, model=MODEL)

    result = await planner.plan(analysis=_edit("a"))
    assert transport.requests == []  # one change → deterministic, no call
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
