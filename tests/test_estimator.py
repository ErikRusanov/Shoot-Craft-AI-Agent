"""Estimator — pure, capped by budget, monotone in it.

demo_avatar ships convergence stats (expected_generations: 2); demo_headshot
does not — together the demo library exercises both sources of the expected
attempt count.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from config import Settings
from schemas import CostEstimate, Preset
from services.estimator import estimate_cost
from services.preset_matcher import PresetLibrary, load_library

UNIT_PRICE = Decimal("2.5")
DEFAULT_EXPECTED = 3


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


def _estimate(preset: Preset, budget_limit: int) -> CostEstimate:
    return estimate_cost(
        preset,
        budget_limit=budget_limit,
        unit_price=UNIT_PRICE,
        default_expected_generations=DEFAULT_EXPECTED,
    )


def test_uses_preset_convergence_stats(avatar: Preset) -> None:
    estimate = _estimate(avatar, budget_limit=10)
    assert estimate.generations == 2
    assert estimate.total_cost == Decimal("5.0")
    assert estimate.note is None


def test_falls_back_to_config_default(headshot: Preset) -> None:
    estimate = _estimate(headshot, budget_limit=10)
    assert estimate.generations == DEFAULT_EXPECTED


def test_capped_by_budget_limit(headshot: Preset) -> None:
    estimate = _estimate(headshot, budget_limit=1)
    assert estimate.generations == 1
    assert estimate.budget_limit == 1
    assert estimate.note is not None and "capped" in estimate.note


def test_zero_budget_estimates_zero(avatar: Preset) -> None:
    estimate = _estimate(avatar, budget_limit=0)
    assert estimate.generations == 0
    assert estimate.total_cost == Decimal("0")


def test_total_is_generations_times_unit_price(avatar: Preset, headshot: Preset) -> None:
    for preset in (avatar, headshot):
        for budget in (0, 1, 2, 5):
            estimate = _estimate(preset, budget_limit=budget)
            assert estimate.total_cost == estimate.generations * UNIT_PRICE
            assert estimate.unit_price == UNIT_PRICE


def test_monotone_in_budget_limit(avatar: Preset, headshot: Preset) -> None:
    for preset in (avatar, headshot):
        estimates = [_estimate(preset, budget_limit=b) for b in range(9)]
        generations = [e.generations for e in estimates]
        totals = [e.total_cost for e in estimates]
        assert generations == sorted(generations)
        assert totals == sorted(totals)


def test_deterministic(headshot: Preset) -> None:
    assert _estimate(headshot, budget_limit=4) == _estimate(headshot, budget_limit=4)


def test_invalid_inputs_raise(avatar: Preset) -> None:
    with pytest.raises(ValueError, match="budget_limit"):
        _estimate(avatar, budget_limit=-1)
    with pytest.raises(ValueError, match="unit_price"):
        estimate_cost(
            avatar, budget_limit=1, unit_price=Decimal("-0.1"), default_expected_generations=3
        )
    with pytest.raises(ValueError, match="default_expected_generations"):
        estimate_cost(
            avatar, budget_limit=1, unit_price=Decimal("1"), default_expected_generations=0
        )
