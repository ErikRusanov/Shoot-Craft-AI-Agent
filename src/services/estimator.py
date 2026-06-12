"""Cost estimation — a dollar forecast of paid spend for the plan.

The expected attempt count comes from the preset's offline-measured convergence
stats (:class:`~schemas.presets.ConvergenceProfile`), falling back to the
config-wide default when a preset ships none. It is then capped by **affordability
under the padded reservation**: the runtime reserves a conservative estimate
before each generation and refuses one it cannot cover, so the plan must not
promise a generation the loop would decline. Pricing is real USD from the
:class:`~services.pricing.PricingTable`.

Pure function: pricing and config arrive as explicit arguments, no I/O, no clock.
"""

from __future__ import annotations

from decimal import Decimal
from typing import NamedTuple

from schemas import CostEstimate, PaidCallKind, Preset
from services.pricing import PricingTable

# Padding over the preset's frozen text to approximate the filled prompt's size
# (slot values + the sanctioned addendum) for the forecast.
_SLOT_PADDING_CHARS = 120


def _nominal_prompt_chars(preset: Preset) -> int:
    """A representative filled-prompt length from the preset's frozen blocks."""
    return (
        len(preset.identity_instruction)
        + len(preset.prompt_structure)
        + len(preset.negative_prompt)
        + _SLOT_PADDING_CHARS
    )


def expected_per_step(preset: Preset, default_expected_generations: int) -> int:
    """The preset's measured expected generations per step, else the config default."""
    if preset.convergence is not None:
        return preset.convergence.expected_generations
    return default_expected_generations


class _Rates(NamedTuple):
    per_generation_cost: Decimal
    per_generation_reserve: Decimal
    llm_overhead_cost: Decimal
    affordable: int  # generations the budget admits under the padded reservation


def _rates(
    preset: Preset, *, budget_limit: Decimal, pricing: PricingTable, generation_model: str
) -> _Rates:
    """The per-generation prices and how many generations the budget affords."""
    prompt_chars = _nominal_prompt_chars(preset)
    face_detail = preset.generation.face_media_resolution  # the crop is sent every attempt
    per_generation_cost = pricing.predict_generation_cost(
        generation_model, prompt_chars=prompt_chars, reference_count=1, face_detail=face_detail
    )
    per_generation_reserve = pricing.generation_reserve(
        generation_model, prompt_chars=prompt_chars, reference_count=1, face_detail=face_detail
    )
    llm_overhead_cost = pricing.flat_estimate(PaidCallKind.SLOT_FILL)
    # Budget left for generations after the auxiliary LLM overhead, divided by the
    # padded per-generation reservation — the runtime's own admission rule.
    spendable = budget_limit - llm_overhead_cost
    affordable = int(spendable / per_generation_reserve) if spendable > 0 else 0
    return _Rates(per_generation_cost, per_generation_reserve, llm_overhead_cost, affordable)


def affordable_step_count(
    preset: Preset,
    *,
    budget_limit: Decimal,
    pricing: PricingTable,
    generation_model: str,
    default_expected_generations: int,
) -> int:
    """How many full plan steps the budget admits, at the per-step expectation.

    The planner trims its tail to this so the plan never promises a step the
    runtime cannot afford to generate.
    """
    rates = _rates(
        preset, budget_limit=budget_limit, pricing=pricing, generation_model=generation_model
    )
    return rates.affordable // expected_per_step(preset, default_expected_generations)


def estimate_cost(
    preset: Preset,
    *,
    budget_limit: Decimal,
    pricing: PricingTable,
    generation_model: str,
    default_expected_generations: int,
    step_count: int = 1,
) -> CostEstimate:
    """Forecast paid spend (USD) for a session on ``preset`` under ``budget_limit``.

    ``step_count`` is the number of plan steps; the forecast is their sum,
    ``step_count * expected_per_step``, capped by affordability. A single step
    (``step_count=1``) is the generate-mode / legacy case.
    """
    if budget_limit < 0:
        raise ValueError(f"budget_limit {budget_limit} must be >= 0")
    if default_expected_generations < 1:
        raise ValueError(
            f"default_expected_generations {default_expected_generations} must be >= 1"
        )

    rates = _rates(
        preset, budget_limit=budget_limit, pricing=pricing, generation_model=generation_model
    )
    expected = expected_per_step(preset, default_expected_generations)
    requested = expected * step_count
    generations = min(requested, rates.affordable)
    note = (
        f"capped by budget; {requested} expected, {rates.affordable} affordable"
        if requested > rates.affordable
        else None
    )

    return CostEstimate(
        generations=generations,
        budget_limit=budget_limit,
        per_generation_cost=rates.per_generation_cost,
        llm_overhead_cost=rates.llm_overhead_cost,
        total_cost=rates.per_generation_cost * generations + rates.llm_overhead_cost,
        note=note,
    )
