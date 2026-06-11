"""Graph assembly — the session FSM.

::

    analyze → quality_gate → classify → ask ⇄ match_fill → plan → approve → generate → done
                    │                    │        │                  │          │
                    └────────────────────┴────────┴──── fail ────────┘         END

``classify`` derives ``use_case`` from the user's brief when the caller gave
none (a budgeted LLM call with a deterministic fallback — it never fails).
``ask`` and ``approve`` pause on ``interrupt()``; ``match_fill`` loops back to
``ask`` when a free-form answer is rejected. ``generate`` routes to ``done``
on a delivered result and straight to END on a loop failure (the loop already
published its own terminal event).

The checkpointer is injected so a resume can happen in a *different* graph
instance than the one that interrupted — the crash-recovery property the
whole design hangs on. In-memory in fake/dev wiring, Redis in prod (see
api/deps.py).
"""

from __future__ import annotations

from collections.abc import Callable

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from graph.nodes import GraphServices, make_nodes
from graph.state import GraphState


def _route_failure(next_node: str) -> Callable[[GraphState], str]:
    def route(state: GraphState) -> str:
        return "fail" if state.get("failure") else next_node

    return route


def _route_match_fill(state: GraphState) -> str:
    if state.get("failure"):
        return "fail"
    if state.get("reask_reason"):
        return "ask"
    return "plan"


def _route_generate(state: GraphState) -> str:
    # On a loop failure the terminal `failed` event is already on the stream —
    # ending the run here keeps exactly one terminal event per session.
    return "done" if state.get("delivered") else END


def build_graph(
    services: GraphServices, *, checkpointer: BaseCheckpointSaver[str]
) -> CompiledStateGraph[GraphState]:
    """Compile the session FSM against the services and a checkpointer."""
    graph: StateGraph[GraphState] = StateGraph(GraphState)
    for name, fn in make_nodes(services).items():
        graph.add_node(name, fn)

    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "quality_gate")
    graph.add_conditional_edges("quality_gate", _route_failure("classify"), ["fail", "classify"])
    graph.add_edge("classify", "ask")
    graph.add_conditional_edges("ask", _route_failure("match_fill"), ["fail", "match_fill"])
    graph.add_conditional_edges("match_fill", _route_match_fill, ["fail", "ask", "plan"])
    graph.add_edge("plan", "approve")
    graph.add_conditional_edges("approve", _route_failure("generate"), ["fail", "generate"])
    graph.add_conditional_edges("generate", _route_generate, ["done", END])
    graph.add_edge("done", END)
    graph.add_edge("fail", END)

    return graph.compile(checkpointer=checkpointer)
