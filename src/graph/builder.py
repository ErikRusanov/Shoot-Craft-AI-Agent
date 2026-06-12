"""Graph assembly — the session FSM.

::

    analyze → quality_gate → parse_brief → ask ⇄ resolve_constraints
        → plan_steps → approve → generate → done
    (quality_gate / ask / resolve_constraints / approve can route to fail;
     generate routes to done or straight to END)

``parse_brief`` reads the brief into a BriefAnalysis (mode, preserve, changes,
conflicts) — a budgeted LLM call with a deterministic fallback that never fails.
``ask`` and ``approve`` pause on ``interrupt()``; ``resolve_constraints`` loops
back to ``ask`` when a free-form answer is rejected. ``plan_steps`` decomposes
the changes into the step plan the user approves. ``generate`` routes to ``done``
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


def _route_resolve(state: GraphState) -> str:
    if state.get("failure"):
        return "fail"
    if state.get("reask_reason"):
        return "ask"
    return "plan_steps"


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
    graph.add_conditional_edges(
        "quality_gate", _route_failure("parse_brief"), ["fail", "parse_brief"]
    )
    graph.add_edge("parse_brief", "ask")
    graph.add_conditional_edges(
        "ask", _route_failure("resolve_constraints"), ["fail", "resolve_constraints"]
    )
    graph.add_conditional_edges(
        "resolve_constraints", _route_resolve, ["fail", "ask", "plan_steps"]
    )
    graph.add_edge("plan_steps", "approve")
    graph.add_conditional_edges("approve", _route_failure("generate"), ["fail", "generate"])
    graph.add_conditional_edges("generate", _route_generate, ["done", END])
    graph.add_edge("done", END)
    graph.add_edge("fail", END)

    return graph.compile(checkpointer=checkpointer)
