"""Graph assembly — the linear walking-skeleton pipeline.

face_check → ask (interrupt) → plan → generate → finalize

The checkpointer is injected so a resume can happen in a *different* graph
instance than the one that interrupted — that is exactly the crash-recovery
property the skeleton must prove. In-memory for now; Redis lands later behind
the same `BaseCheckpointSaver` seam.
"""

from __future__ import annotations

from itertools import pairwise

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from graph.nodes import make_nodes
from graph.state import GraphState
from protocols.event_bus import EventBus

PIPELINE = ("face_check", "ask", "plan", "generate", "finalize")


def build_graph(
    bus: EventBus, *, checkpointer: BaseCheckpointSaver[str]
) -> CompiledStateGraph[GraphState]:
    """Compile the skeleton pipeline against a bus and a checkpointer."""
    graph: StateGraph[GraphState] = StateGraph(GraphState)
    for name, fn in make_nodes(bus).items():
        graph.add_node(name, fn)

    graph.add_edge(START, PIPELINE[0])
    for prev, nxt in pairwise(PIPELINE):
        graph.add_edge(prev, nxt)
    graph.add_edge(PIPELINE[-1], END)

    return graph.compile(checkpointer=checkpointer)
