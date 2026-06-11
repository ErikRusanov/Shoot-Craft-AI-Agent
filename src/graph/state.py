"""Graph state — the working set the LangGraph FSM threads through nodes.

A thin TypedDict over the contract in `schemas/state`: the graph carries only
what nodes pass between each other; the durable session record stays in
:class:`~schemas.state.SessionState` behind the StateStore port. LangGraph
checkpoints this dict, so everything here must survive its serde round-trip —
plain strings, ints, dicts and ``None`` only, never pydantic models.
"""

from __future__ import annotations

from typing import NotRequired, TypedDict


class GraphState(TypedDict):
    """Working state for one session run, keyed by ``thread_id == session_key``.

    The required keys are the run's input (seeded by :func:`initial_state`);
    the ``NotRequired`` ones accumulate as nodes execute and exist only from
    the node that sets them onward — read them with ``.get``.
    """

    session_key: str
    face_key: str
    use_case: str
    gender: str
    budget_limit: int

    # -- set by analyze --
    gate_verdict: NotRequired[str]  # Verdict value of the input-photo gate
    gate_reason: NotRequired[str]  # GateReason value behind the verdict
    has_identity: NotRequired[bool]  # the profile carries an embedding

    # -- set by ask / match_fill --
    answer: NotRequired[str | None]  # the user's reply to the one ask-slot question
    preset_id: NotRequired[str]
    slots: NotRequired[dict[str, str]]  # resolved (and later composition-merged) values
    addendum: NotRequired[str]  # the filler's sanctioned free-text extension
    reasks: NotRequired[int]  # free-form rejections so far, capped
    reask_reason: NotRequired[str | None]  # non-None routes match_fill back to ask

    # -- terminal routing --
    failure: NotRequired[dict[str, str | None] | None]  # reason/code (+ gate_reason) → fail node
    delivered: NotRequired[bool]  # the generation loop ended in a result, not a failure


def initial_state(
    *, session_key: str, face_key: str, use_case: str, gender: str, budget_limit: int
) -> GraphState:
    """The seed state for a fresh run — exactly the contract's start payload."""
    return {
        "session_key": session_key,
        "face_key": face_key,
        "use_case": use_case,
        "gender": gender,
        "budget_limit": budget_limit,
    }
