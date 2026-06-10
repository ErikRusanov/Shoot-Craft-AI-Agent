"""Cost estimation — a pure forecast of paid generations for the plan.

The expected attempt count comes from the preset's offline-measured
convergence stats (:class:`~schemas.presets.ConvergenceProfile`), falling back
to the config-wide default when a preset ships none, and is capped by the
session's ``budget_limit`` — the business service's ceiling is authoritative,
so the plan never promises more than it may spend. Price per generation is a
config value in abstract units; mapping units to user-facing money is the
business service's job.

Pure function: config values arrive as explicit arguments, no I/O, no clock.
"""

from __future__ import annotations

from decimal import Decimal

from schemas import CostEstimate, Preset


def estimate_cost(
    preset: Preset,
    *,
    budget_limit: int,
    unit_price: Decimal,
    default_expected_generations: int,
) -> CostEstimate:
    """Forecast paid generations and their cost for a session on ``preset``."""
    if budget_limit < 0:
        raise ValueError(f"budget_limit {budget_limit} must be >= 0")
    if unit_price < 0:
        raise ValueError(f"unit_price {unit_price} must be >= 0")
    if default_expected_generations < 1:
        raise ValueError(
            f"default_expected_generations {default_expected_generations} must be >= 1"
        )

    expected = (
        preset.convergence.expected_generations
        if preset.convergence is not None
        else default_expected_generations
    )
    generations = min(expected, budget_limit)
    note = f"capped by budget_limit; {expected} expected" if expected > budget_limit else None

    return CostEstimate(
        generations=generations,
        budget_limit=budget_limit,
        unit_price=unit_price,
        total_cost=generations * unit_price,
        note=note,
    )
