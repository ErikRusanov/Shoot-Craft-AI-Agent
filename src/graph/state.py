"""Graph state — the mutable working set the LangGraph FSM threads through nodes.

A thin TypedDict over the contract in `schemas/state`: the graph carries only
what nodes pass between each other; the durable session record stays in
:class:`~schemas.state.SessionState` behind the StateStore port. LangGraph
checkpoints this dict, so everything here must survive its serde round-trip.
"""

from __future__ import annotations

from typing import TypedDict


class GraphState(TypedDict):
    """Working state for one session run, keyed by ``thread_id == session_key``."""

    session_key: str
    face_key: str
    # Resolved slot values (the answer to the `ask` interrupt lands here).
    slots: dict[str, str]
