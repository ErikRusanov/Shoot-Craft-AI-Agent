"""Pricing — the June-2026 OpenRouter numbers and the reserve/predict relation.

Pins the dominant arithmetic (the output image is ~98% of a generation's cost),
so a silent rate drift or a unit slip (per-token vs per-million) is caught.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from schemas import PaidCallKind
from services.pricing import PricingTable

GEN = "google/gemini-3.1-flash-image-preview"
LITE = "google/gemini-3.1-flash-lite"


def _table() -> PricingTable:
    return PricingTable.default(generation_model=GEN, lite_model=LITE)


def test_default_table_has_the_two_models() -> None:
    table = _table()
    assert table.rate_for(GEN).image_output_per_mtok == Decimal("60")
    assert table.rate_for(GEN).input_per_mtok == Decimal("0.50")
    assert table.rate_for(LITE).input_per_mtok == Decimal("0.25")
    # The lite model produces no images, so it carries no image-output rate.
    assert table.rate_for(LITE).image_output_per_mtok == Decimal("0")


def test_one_generation_is_about_seven_cents() -> None:
    # ~1000-char prompt, one reference, 1K output: the output image dominates.
    cost = _table().predict_generation_cost(GEN, prompt_chars=1000, reference_count=1)
    assert cost == pytest.approx(Decimal("0.069"), abs=Decimal("0.0015"))


def test_output_resolution_dominates_cost() -> None:
    table = _table()
    small = table.predict_generation_cost(
        GEN, prompt_chars=1000, reference_count=1, output_size="0.5K"
    )
    large = table.predict_generation_cost(
        GEN, prompt_chars=1000, reference_count=1, output_size="4K"
    )
    assert large > small


def test_face_crop_adds_input_cost() -> None:
    table = _table()
    without = table.predict_generation_cost(GEN, prompt_chars=1000, reference_count=1)
    with_crop = table.predict_generation_cost(
        GEN, prompt_chars=1000, reference_count=1, face_detail="high"
    )
    assert with_crop > without


def test_reserve_pads_the_forecast() -> None:
    table = _table()
    predicted = table.predict_generation_cost(GEN, prompt_chars=1000, reference_count=1)
    reserve = table.generation_reserve(GEN, prompt_chars=1000, reference_count=1)
    assert reserve > predicted
    assert reserve == pytest.approx(predicted * Decimal("1.15"), abs=Decimal("0.0001"))


def test_flat_estimates_for_auxiliary_calls() -> None:
    table = _table()
    assert table.flat_estimate(PaidCallKind.SLOT_FILL) == Decimal("0.002")
    assert table.flat_estimate(PaidCallKind.CLASSIFY) == Decimal("0.002")


def test_unknown_model_raises() -> None:
    with pytest.raises(ValueError, match="no pricing for model"):
        _table().predict_generation_cost("mystery/model", prompt_chars=100, reference_count=1)
