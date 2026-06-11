"""Port: budget meter — reserve before a paid call, settle to its real cost.

Pay-as-you-go: the dollar budget must cover every upstream call and a slot has
to be claimed *before* the call returns its real price. The meter is that seam.
:meth:`BudgetMeter.reserve` atomically holds a conservative estimate (``None``
when the budget can't cover it — the caller must not make the call); the returned
:class:`BudgetReservation` is then either **settled** to the real cost (a signed
delta against the held estimate) or **cancelled** (full refund) when the provider
never charged. Every future paid call — the image generator today, the slot
filler and the use-case classifier — flows through this one surface, so adding a
paid call never touches the budget arithmetic.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Protocol, runtime_checkable

from schemas.enums import PaidCallKind


@runtime_checkable
class BudgetReservation(Protocol):
    """A held reservation; settle it to the real cost or cancel for a refund."""

    async def settle(self, actual_cost: Decimal | None = None) -> Decimal:
        """Settle to ``actual_cost`` (or the reserved estimate when ``None``).

        Applies the signed delta against the held estimate and returns the final
        cost recorded for this call. ``None`` means the provider charged but
        reported no ``usage.cost`` — keep the reserved estimate as the spend
        rather than guessing. Settling twice is a programming error and raises.
        """
        ...

    async def cancel(self) -> None:
        """Refund the whole reservation — the provider never charged (transport
        failure / 4xx). Cancelling a settled reservation raises."""
        ...


@runtime_checkable
class BudgetMeter(Protocol):
    """Reserves dollar slots for one session, never past the limit."""

    async def reserve(
        self, kind: PaidCallKind, *, estimate: Decimal | None = None
    ) -> BudgetReservation | None:
        """Reserve a slot for a ``kind`` call; ``None`` when the budget can't cover it.

        ``estimate`` overrides the per-kind default (the generation loop passes
        the price of the specific prompt it is about to send); omitted, the meter
        uses the flat padded estimate for that kind. A non-``None`` return is
        already reserved — the call may proceed and must be settled or cancelled.
        """
        ...
