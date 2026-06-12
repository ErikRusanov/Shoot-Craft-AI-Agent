"""Estimator — a dollar forecast, capped by affordability under the reservation.

demo_avatar ships convergence stats (expected_generations: 2); demo_headshot
does not — together the demo library exercises both sources of the expected
attempt count.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from config import Settings
from schemas import CostEstimate, Preset
from services.estimator import affordable_step_count, estimate_cost, expected_per_step
from services.preset_matcher import PresetLibrary, load_library
from services.pricing import PricingTable

GEN = "gen-model"
DEFAULT_EXPECTED = 3


@pytest.fixture(scope="module")
def pricing() -> PricingTable:
    return PricingTable.default(generation_model=GEN, lite_model="lite-model")


@pytest.fixture(scope="module")
def library() -> PresetLibrary:
    return load_library(Settings(_env_file=None))


@pytest.fixture(scope="module")
def avatar(library: PresetLibrary) -> Preset:
    preset = library.get("demo_avatar")
    assert preset is not None and preset.convergence is not None
    return preset


@pytest.fixture(scope="module")
def headshot(library: PresetLibrary) -> Preset:
    preset = library.get("demo_headshot")
    assert preset is not None and preset.convergence is None
    return preset


def _estimate(preset: Preset, pricing: PricingTable, budget: str) -> CostEstimate:
    return estimate_cost(
        preset,
        budget_limit=Decimal(budget),
        pricing=pricing,
        generation_model=GEN,
        default_expected_generations=DEFAULT_EXPECTED,
    )


def test_uses_preset_convergence_stats(avatar: Preset, pricing: PricingTable) -> None:
    estimate = _estimate(avatar, pricing, "10")
    assert estimate.generations == 2  # the preset's expected_generations, affordable
    assert estimate.note is None


def test_falls_back_to_config_default(headshot: Preset, pricing: PricingTable) -> None:
    estimate = _estimate(headshot, pricing, "10")
    assert estimate.generations == DEFAULT_EXPECTED


def test_capped_by_budget(headshot: Preset, pricing: PricingTable) -> None:
    # ~$0.08 per padded reservation: a $0.10 budget affords just one generation.
    estimate = _estimate(headshot, pricing, "0.10")
    assert estimate.generations == 1
    assert estimate.note is not None and "capped" in estimate.note


def test_zero_budget_estimates_zero(avatar: Preset, pricing: PricingTable) -> None:
    estimate = _estimate(avatar, pricing, "0")
    assert estimate.generations == 0
    assert estimate.total_cost == estimate.llm_overhead_cost  # only the (unspent) overhead line


def test_total_is_per_generation_plus_overhead(avatar: Preset, pricing: PricingTable) -> None:
    estimate = _estimate(avatar, pricing, "10")
    assert estimate.total_cost == (
        estimate.per_generation_cost * estimate.generations + estimate.llm_overhead_cost
    )
    assert estimate.per_generation_cost > 0
    assert estimate.llm_overhead_cost == Decimal("0.002")


def test_monotone_in_budget(headshot: Preset, pricing: PricingTable) -> None:
    budgets = ["0", "0.05", "0.10", "0.20", "0.40", "10"]
    generations = [_estimate(headshot, pricing, b).generations for b in budgets]
    assert generations == sorted(generations)


def test_deterministic(headshot: Preset, pricing: PricingTable) -> None:
    assert _estimate(headshot, pricing, "0.30") == _estimate(headshot, pricing, "0.30")


def test_multi_step_forecast_sums_over_steps(headshot: Preset, pricing: PricingTable) -> None:
    # default expected = 3 per step; 2 steps under a generous budget → 6.
    estimate = estimate_cost(
        headshot,
        budget_limit=Decimal("10"),
        pricing=pricing,
        generation_model=GEN,
        default_expected_generations=DEFAULT_EXPECTED,
        step_count=2,
    )
    assert estimate.generations == 6
    assert estimate.note is None


def test_multi_step_capped_by_budget(headshot: Preset, pricing: PricingTable) -> None:
    estimate = estimate_cost(
        headshot,
        budget_limit=Decimal("0.20"),
        pricing=pricing,
        generation_model=GEN,
        default_expected_generations=DEFAULT_EXPECTED,
        step_count=3,
    )
    assert estimate.generations < 9
    assert estimate.note is not None and "capped" in estimate.note


def test_affordable_step_count(headshot: Preset, pricing: PricingTable) -> None:
    per_step = expected_per_step(headshot, DEFAULT_EXPECTED)
    assert per_step == DEFAULT_EXPECTED
    # ~$0.08 per padded reservation: $0.30 affords ~3 generations → 1 full step.
    steps = affordable_step_count(
        headshot,
        budget_limit=Decimal("0.30"),
        pricing=pricing,
        generation_model=GEN,
        default_expected_generations=DEFAULT_EXPECTED,
    )
    assert steps == 1


def test_invalid_inputs_raise(avatar: Preset, pricing: PricingTable) -> None:
    with pytest.raises(ValueError, match="budget_limit"):
        _estimate(avatar, pricing, "-1")
    with pytest.raises(ValueError, match="default_expected_generations"):
        estimate_cost(
            avatar,
            budget_limit=Decimal("1"),
            pricing=pricing,
            generation_model=GEN,
            default_expected_generations=0,
        )
