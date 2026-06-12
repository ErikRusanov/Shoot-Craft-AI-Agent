"""Port: brief parser — read a free-text request into a :class:`BriefAnalysis`.

Replaces the use-case classifier. The classifier collapsed the brief to one
token and threw away the constraints ("keep X") and deltas ("change Y to blue");
the parser keeps them. One call returns the structured reading the whole
pipeline runs on — mode, preserve-list, changes, conflicts — and the chosen
``use_case`` (for a generate-mode curated preset) when there is one.

Same failure policy as the other LLM ports: an implementation reserves a dollar
slot through the :class:`~protocols.budget.BudgetMeter` before its paid call and
degrades to a free deterministic parse when the budget refuses or the call
misbehaves — parsing must never fail the session.
"""

from __future__ import annotations

from collections.abc import Sequence
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
        use_case: str | None,
        use_cases: Sequence[str],
        meter: BudgetMeter | None = None,
    ) -> ParseResult:
        """Analyze ``brief`` into mode, preserve-list, changes and conflicts.

        ``use_case`` is the caller-supplied token when the business service
        already knows the target (then the reading is target-driven); ``None``
        leaves the parser to decide mode and pick a ``use_case`` from
        ``use_cases`` (the library's curated vocabulary, never the reserved
        ``default``). ``meter`` is the session budget for the paid path; a
        refused budget degrades to a free deterministic parse.
        """
        ...
