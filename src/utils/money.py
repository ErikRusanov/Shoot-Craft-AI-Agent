"""Money — the one place USD becomes integer micro-dollars and back.

Money is :class:`~decimal.Decimal` USD everywhere in Python and on the wire
(pydantic serializes Decimal as a string, so there is never a float in the
arithmetic). Redis cannot store a Decimal atomically, and the budget counter
must be incremented under a Lua script, so *inside the store* money is integer
**micro-dollars** (1 USD = 1_000_000 micro). This module is the only boundary
that converts between the two representations.

Rounding is asymmetric on purpose: a reservation rounds **up** (``ROUND_CEILING``)
so the conservative estimate can never under-reserve by a sub-micro crumb, while
reading a counter back is exact. ``parse_usd`` quantizes provider-supplied costs
to 6 decimal places — micro resolution — so ``usage.cost`` lands on the same grid
the counter uses.
"""

from __future__ import annotations

from decimal import ROUND_CEILING, ROUND_HALF_UP, Decimal

_MICRO = Decimal(1_000_000)
# 6 dp == micro-dollar resolution; all USD values live on this grid.
_USD_QUANT = Decimal("0.000001")


def to_micro(usd: Decimal, *, rounding: str = ROUND_CEILING) -> int:
    """USD → integer micro-dollars. Rounds up by default (reservations never
    under-reserve); pass ``ROUND_FLOOR`` for a limit that must not over-grant."""
    return int((Decimal(usd) * _MICRO).to_integral_value(rounding=rounding))


def from_micro(micro: int) -> Decimal:
    """Integer micro-dollars → exact USD on the 6-dp grid."""
    return (Decimal(micro) / _MICRO).quantize(_USD_QUANT)


def parse_usd(value: object) -> Decimal:
    """Coerce a provider-supplied cost (float/str/Decimal) to 6-dp USD.

    Goes through ``str`` so a float's binary noise never enters the Decimal.
    """
    return Decimal(str(value)).quantize(_USD_QUANT, rounding=ROUND_HALF_UP)
