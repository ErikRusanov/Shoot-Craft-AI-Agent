"""Port: use-case classifier — map the user's free-text brief to a preset token.

The business service often has no ``use_case`` (the user just described what they
want), so the core derives one from the ``brief``. The classifier picks one token
from the library's curated vocabulary, or the reserved ``default`` when none fits
— that token then drives preset matching exactly as a caller-supplied one would.

Like the slot filler, an implementation reserves a dollar slot through the budget
:class:`~protocols.budget.BudgetMeter` before its paid call and degrades to a free
deterministic guess when the budget refuses or the call misbehaves — classifying
must never fail the session.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from typing import NamedTuple, Protocol, runtime_checkable

from protocols.budget import BudgetMeter
from schemas import ProviderUsage


class ClassifyResult(NamedTuple):
    """The chosen ``use_case`` token plus what the call billed.

    ``cost`` is the dollars an LLM-backed classifier settled (0 for the
    deterministic fallback); ``usage`` is the provider's billing detail. The
    orchestration records these so the dollar budget accounts for classification.
    """

    use_case: str
    usage: ProviderUsage | None = None
    cost: Decimal = Decimal("0")


@runtime_checkable
class UseCaseClassifier(Protocol):
    """Resolve a ``use_case`` token from a free-text brief."""

    async def classify(
        self, *, brief: str, use_cases: Sequence[str], meter: BudgetMeter | None = None
    ) -> ClassifyResult:
        """Pick one of ``use_cases`` (or ``default``) that best fits ``brief``.

        ``use_cases`` is the library's curated vocabulary (never the reserved
        ``default`` token — that is the implicit fall-through). ``meter`` is the
        session budget for the paid path; a refused budget degrades to a free
        deterministic guess rather than failing.
        """
        ...
