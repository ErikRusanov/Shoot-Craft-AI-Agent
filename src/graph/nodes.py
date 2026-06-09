"""Graph nodes — walking-skeleton stubs.

Real nodes will only call services; these stubs prove the plumbing instead:
each one narrates its stage to the EventBus, and `ask` exercises the
interrupt/resume mechanic. Nodes are built by a factory so the bus arrives via
DI, never as a global.

Interrupt semantics that shape the design: on resume LangGraph re-executes the
interrupted node *from the top*, so `ask` must not publish anything before
``interrupt()`` — a pre-interrupt publish would duplicate on every resume.
The runner (api/deps.py) emits the ``need_input`` event from the surfaced
interrupt payload exactly once instead.
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any, Protocol

from langgraph.types import interrupt

from graph.state import GraphState
from protocols.event_bus import EventBus
from schemas import DoneEvent, FsmState, StageEvent


class NodeFn(Protocol):
    """A graph node. A Protocol (not a Callable alias) because LangGraph's
    `add_node` overloads require the ``state`` parameter to be passable by
    keyword, which a bare Callable type erases."""

    def __call__(self, state: GraphState) -> Awaitable[dict[str, Any]]: ...


# The one clarifying question of the skeleton; real slots come from the preset.
ASK_SLOT = "style"
ASK_QUESTION = "Which style should the shoot use?"


def make_nodes(bus: EventBus) -> dict[str, NodeFn]:
    """Bind the stub nodes to a bus; returns them keyed by graph node name."""

    async def face_check(state: GraphState) -> dict[str, Any]:
        await bus.publish(state["session_key"], StageEvent(stage=FsmState.FACE_CHECK))
        return {}

    async def ask(state: GraphState) -> dict[str, Any]:
        # Payload is a plain dict (not a model) so the checkpointer's serde
        # never depends on pydantic; the runner rebuilds NeedInputEvent from it.
        answer = interrupt(
            {"slot": ASK_SLOT, "question": ASK_QUESTION, "options": None, "default": None}
        )
        return {"slots": {**state["slots"], ASK_SLOT: str(answer)}}

    async def plan(state: GraphState) -> dict[str, Any]:
        await bus.publish(state["session_key"], StageEvent(stage=FsmState.PLANNING))
        return {}

    async def generate(state: GraphState) -> dict[str, Any]:
        await bus.publish(state["session_key"], StageEvent(stage=FsmState.GENERATING))
        return {}

    async def finalize(state: GraphState) -> dict[str, Any]:
        # Echo the answered slot so an e2e test can assert the resume value
        # made it through the whole pipeline, not just past the interrupt.
        await bus.publish(state["session_key"], DoneEvent(detail=state["slots"].get(ASK_SLOT)))
        return {}

    return {
        "face_check": face_check,
        "ask": ask,
        "plan": plan,
        "generate": generate,
        "finalize": finalize,
    }
