"""Port: brief parser — read a free-text request into a :class:`BriefAnalysis`.

One call returns the structured reading the whole pipeline runs on —
preserve-list, changes, conflicts. The pipeline is always edit-mode; the
parser's job is to extract what must change and what must be kept.

Same failure policy as the other LLM ports: an implementation reserves a dollar
slot through the :class:`~protocols.budget.BudgetMeter` before its paid call and
degrades to a free deterministic parse when the budget refuses or the call
misbehaves — parsing must never fail the session.
"""

from __future__ import annotations

from decimal import Decimal
from typing import NamedTuple, Protocol, runtime_checkable

from protocols.budget import BudgetMeter
from schemas import BriefAnalysis, ProviderUsage


class ParseResult(NamedTuple):
    """The structured reading plus what the call billed.

    ``cost`` is the dollars an LLM-backed parser settled (0 for the deterministic
    fallback); ``usage`` is the provider's billing detail. The orchestration
    records these so the dollar budget accounts for brief parsing.
    """

    analysis: BriefAnalysis
    usage: ProviderUsage | None = None
    cost: Decimal = Decimal("0")


@runtime_checkable
class BriefParser(Protocol):
    """Read a free-text brief into a :class:`BriefAnalysis`."""

    async def parse(
        self,
        *,
        brief: str,
        meter: BudgetMeter | None = None,
    ) -> ParseResult:
        """Analyze ``brief`` into preserve-list, changes and conflicts.

        ``meter`` is the session budget for the paid path; a refused budget
        degrades to a free deterministic parse.
        """
        ...
