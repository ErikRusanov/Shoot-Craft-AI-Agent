"""Cost estimation — a dollar forecast of paid spend for the plan.

Spending is **greedy pay-as-you-go**: the runtime reserves a padded estimate
before each generation and refuses one it cannot cover, so the budget drains down
to its last affordable generation regardless of how the steps split. The forecast
therefore reports a **floor** (one generation per step, the best case with zero
retries) and the budget ceiling — how many generations the reservation admits in
total, retries included — never a per-step attempt guess. Pricing is real USD from
the :class:`~services.pricing.PricingTable`.

Intermediate steps generate at 1K to keep per-attempt cost low; the final
quality-enhancement step runs at 4K. The estimate accounts for this split.

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


class _Rates(NamedTuple):
    per_intermediate_cost: Decimal  # realistic cost for one 1K generation
    per_enhance_cost: Decimal  # realistic cost for one 4K enhance generation
    per_intermediate_reserve: Decimal
    per_enhance_reserve: Decimal
    llm_overhead_cost: Decimal
    affordable: int  # total generations the budget admits under padded reservations


def _rates(
    preset: Preset,
    *,
    budget_limit: Decimal,
    pricing: PricingTable,
    generation_model: str,
    enhance_step_count: int,
) -> _Rates:
    """Per-generation prices (1K intermediate, 4K enhance) and budget headroom."""
    prompt_chars = _nominal_prompt_chars(preset)
    face_detail = preset.generation.face_media_resolution  # crop is sent every attempt

    def _pair(output_size: str) -> tuple[Decimal, Decimal]:
        cost = pricing.predict_generation_cost(
            generation_model,
            prompt_chars=prompt_chars,
            reference_count=1,
            output_size=output_size,
            face_detail=face_detail,
        )
        reserve = pricing.generation_reserve(
            generation_model,
            prompt_chars=prompt_chars,
            reference_count=1,
            output_size=output_size,
            face_detail=face_detail,
        )
        return cost, reserve

    int_cost, int_reserve = _pair("1K")
    enh_cost, enh_reserve = _pair("4K")
    llm_overhead_cost = pricing.flat_estimate(PaidCallKind.SLOT_FILL)

    # Set aside padded budget for all enhance steps; the rest goes to 1K generations.
    spendable = budget_limit - llm_overhead_cost - enhance_step_count * enh_reserve
    affordable_intermediate = int(spendable / int_reserve) if spendable > 0 else 0
    affordable = affordable_intermediate + enhance_step_count

    return _Rates(int_cost, enh_cost, int_reserve, enh_reserve, llm_overhead_cost, affordable)


def estimate_cost(
    preset: Preset,
    *,
    budget_limit: Decimal,
    pricing: PricingTable,
    generation_model: str,
    step_count: int = 1,
    enhance_step_count: int = 1,
) -> CostEstimate:
    """Forecast paid spend (USD) for a session on ``preset`` under ``budget_limit``.

    ``step_count`` is the total number of plan steps (including enhance steps).
    ``enhance_step_count`` is how many of those are quality-enhancement steps that
    generate at 4K; the rest generate at 1K. The forecast is a **floor** of one
    generation per step (the best case, zero retries); ``total_cost`` is the full
    ``budget_limit`` because greedy spending may consume all of it. ``note`` carries
    the budget ceiling — how many generations the padded reservation admits in
    total — so a budget that cannot even fund one generation per step surfaces as a
    partial-chain warning rather than a silent trim.
    """
    if budget_limit < 0:
        raise ValueError(f"budget_limit {budget_limit} must be >= 0")

    rates = _rates(
        preset,
        budget_limit=budget_limit,
        pricing=pricing,
        generation_model=generation_model,
        enhance_step_count=enhance_step_count,
    )
    floor = max(step_count, 1)
    intermediate_count = max(step_count - enhance_step_count, 0)
    minimum_cost = (
        rates.per_intermediate_cost * intermediate_count
        + rates.per_enhance_cost * enhance_step_count
        + rates.llm_overhead_cost
    )
    note = (
        f"sufficient — up to {rates.affordable} generations with retries funded"
        if rates.affordable >= floor
        else f"⚠ budget covers ~{rates.affordable} of {floor} steps — the rest ship partial"
    )

    return CostEstimate(
        generations=floor,
        budget_limit=budget_limit,
        per_generation_cost=rates.per_intermediate_cost,
        llm_overhead_cost=rates.llm_overhead_cost,
        total_cost=budget_limit,
        minimum_cost=minimum_cost,
        note=note,
    )
