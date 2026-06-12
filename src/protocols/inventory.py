"""Port: inventory extractor — catalogue what is visible in the reference photo.

One VLM call per reference photo turning pixels into a
:class:`~schemas.inventory.PhotoInventory`: the concrete visible attributes
(pose, hands, accessories, clothing, hair, lighting, background) the edit-mode
prompt builder enumerates as untouchables.

Same failure policy as the other LLM ports: an implementation reserves a dollar
slot through the :class:`~protocols.budget.BudgetMeter` before its paid call and
degrades to an empty inventory when the budget refuses or the call misbehaves —
extraction must never fail the session, it only makes prompts less specific.
"""

from __future__ import annotations

from decimal import Decimal
from typing import NamedTuple, Protocol, runtime_checkable

from protocols.budget import BudgetMeter
from schemas import PhotoInventory, ProviderUsage


class InventoryResult(NamedTuple):
    """The extracted inventory plus what the call billed.

    ``cost`` is the dollars an LLM-backed extractor settled (0 for the empty
    fallback); ``usage`` is the provider's billing detail. The orchestration
    records these so the dollar budget accounts for the extraction.
    """

    inventory: PhotoInventory
    usage: ProviderUsage | None = None
    cost: Decimal = Decimal("0")


@runtime_checkable
class InventoryExtractor(Protocol):
    """Extract a :class:`PhotoInventory` from one reference photo."""

    async def extract(
        self,
        image: bytes,
        *,
        meter: BudgetMeter | None = None,
    ) -> InventoryResult:
        """Catalogue the visible attributes of ``image``.

        ``meter`` is the session budget for the paid path; a refused budget
        degrades to an empty inventory rather than failing.
        """
        ...
