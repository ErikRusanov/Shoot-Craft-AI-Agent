"""Money conversions — USD ↔ micro-USD, with reservation-safe rounding."""

from __future__ import annotations

from decimal import ROUND_FLOOR, Decimal

import pytest

from utils.money import from_micro, parse_usd, to_micro


def test_round_trip_exact_on_the_micro_grid() -> None:
    for usd in ("0", "0.069", "0.50", "1.234567"):
        assert from_micro(to_micro(Decimal(usd))) == Decimal(usd).quantize(Decimal("0.000001"))


def test_to_micro_rounds_up_by_default() -> None:
    # A sub-micro crumb rounds up, so a reservation never under-reserves.
    assert to_micro(Decimal("0.0000001")) == 1
    assert to_micro(Decimal("0.0695001")) == 69501


def test_to_micro_floor_for_limits() -> None:
    # A limit must not over-grant, so it floors.
    assert to_micro(Decimal("0.0695009"), rounding=ROUND_FLOOR) == 69500


def test_parse_usd_avoids_float_noise() -> None:
    # A float cost from the provider must not smuggle binary noise into the grid.
    assert parse_usd(0.1) == Decimal("0.100000")
    assert parse_usd("0.0671") == Decimal("0.067100")
    assert parse_usd(0) == Decimal("0")


def test_from_micro_is_six_dp() -> None:
    assert from_micro(69501) == Decimal("0.069501")
    assert str(from_micro(500000)) == "0.500000"


@pytest.mark.parametrize("micro", [0, 1, 69500, 1_000_000])
def test_from_micro_round_trips_through_to_micro(micro: int) -> None:
    assert to_micro(from_micro(micro)) == micro
