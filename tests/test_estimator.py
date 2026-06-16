"""Estimator — a greedy-spend forecast: a floor of one generation per step plus
the budget ceiling, never a per-step attempt guess.

Spending is pay-as-you-go: the runtime reserves before each generation and stops
when the next would overdraw, so the plan is never trimmed. The forecast reports
``generations`` = one per step (best case) and a ``note`` carrying how many
generations the padded reservation admits in total.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from config import Settings
from schemas import CostEstimate, Preset
from services.estimator import estimate_cost
from services.preset_matcher import PresetLibrary, load_library
from services.pricing import PricingTable

GEN = "gen-model"


@pytest.fixture(scope="module")
def pricing() -> PricingTable:
    return PricingTable.default(generation_model=GEN, lite_model="lite-model")


@pytest.fixture(scope="module")
def library() -> PresetLibrary:
    return load_library(Settings(_env_file=None))


@pytest.fixture(scope="module")
def preset(library: PresetLibrary) -> Preset:
    p = library.get("demo_headshot")
    assert p is not None
    return p


def _estimate(
    preset: Preset,
    pricing: PricingTable,
    budget: str,
    *,
    steps: int = 1,
    enhance_steps: int = 1,
) -> CostEstimate:
    return estimate_cost(
        preset,
        budget_limit=Decimal(budget),
        pricing=pricing,
        generation_model=GEN,
        step_count=steps,
        enhance_step_count=enhance_steps,
    )


def test_floor_is_one_generation_per_step(preset: Preset, pricing: PricingTable) -> None:
    assert _estimate(preset, pricing, "10", steps=1).generations == 1
    assert _estimate(preset, pricing, "10", steps=4).generations == 4


def test_total_cost_is_the_full_budget(preset: Preset, pricing: PricingTable) -> None:
    # Greedy spend: the session may consume the whole limit, so that is the total.
    estimate = _estimate(preset, pricing, "10", steps=2)
    assert estimate.total_cost == Decimal("10")


def test_note_reports_the_budget_ceiling(preset: Preset, pricing: PricingTable) -> None:
    # A generous budget funds many generations; the note states how many.
    estimate = _estimate(preset, pricing, "10", steps=2)
    assert estimate.note is not None and "sufficient" in estimate.note


def test_underfunded_chain_warns_partial(preset: Preset, pricing: PricingTable) -> None:
    # ~$0.08 per padded reservation: $0.20 funds ~2 generations, fewer than the
    # 5-step floor → the note flags a partial chain, but the floor still counts
    # every step (the plan is never trimmed).
    estimate = _estimate(preset, pricing, "0.20", steps=5)
    assert estimate.generations == 5
    assert estimate.note is not None and "ship partial" in estimate.note


def test_minimum_cost_is_floor_times_per_gen_plus_llm(
    preset: Preset, pricing: PricingTable
) -> None:
    # No enhance steps: all steps are 1K intermediates, so the formula is uniform.
    estimate = _estimate(preset, pricing, "10", steps=3, enhance_steps=0)
    expected = estimate.per_generation_cost * 3 + estimate.llm_overhead_cost
    assert estimate.minimum_cost == expected


def test_minimum_cost_single_step(preset: Preset, pricing: PricingTable) -> None:
    # No enhance steps: single 1K intermediate step.
    estimate = _estimate(preset, pricing, "10", steps=1, enhance_steps=0)
    assert estimate.minimum_cost == estimate.per_generation_cost + estimate.llm_overhead_cost


def test_minimum_cost_mixed_pricing(preset: Preset, pricing: PricingTable) -> None:
    # 3 steps: 2 intermediate (1K) + 1 enhance (4K). The enhance costs more than
    # per_generation_cost, so minimum_cost > per_generation_cost * 3 + overhead.
    estimate_mixed = _estimate(preset, pricing, "10", steps=3, enhance_steps=1)
    estimate_all_1k = _estimate(preset, pricing, "10", steps=3, enhance_steps=0)
    assert estimate_mixed.minimum_cost > estimate_all_1k.minimum_cost


def test_informational_unit_prices(preset: Preset, pricing: PricingTable) -> None:
    estimate = _estimate(preset, pricing, "10")
    assert estimate.per_generation_cost > 0
    assert estimate.llm_overhead_cost == Decimal("0.002")


def test_deterministic(preset: Preset, pricing: PricingTable) -> None:
    assert _estimate(preset, pricing, "0.30", steps=3) == _estimate(
        preset, pricing, "0.30", steps=3
    )


def test_negative_budget_raises(preset: Preset, pricing: PricingTable) -> None:
    with pytest.raises(ValueError, match="budget_limit"):
        _estimate(preset, pricing, "-1")
