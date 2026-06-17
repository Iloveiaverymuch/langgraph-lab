"""
Graph topology for the blackboard pattern.

Topology:
    START → classifier → supervisor ⇢ researcher → supervisor
                                   ⇢ coder      → supervisor
                                   ⇢ critic     → supervisor
                                   ⇢ END

classifier is a pre-flight node with one unconditional edge.
It writes task_type once. Supervisor still owns all routing after that.
"""

from langgraph.graph import StateGraph, START, END

from .state import AgentState
from .nodes import (
    classifier_node,
    supervisor_node,
    researcher,
    coder,
    critic,
    WORKERS,
)


def route_supervisor(state: AgentState) -> str:
    """Edge function — reads state["next"], returns target node name."""
    if state["next"] == "FINISH":
        return END
    return state["next"]


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # nodes
    graph.add_node("classifier", classifier_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher)
    graph.add_node("coder",      coder)
    graph.add_node("critic",     critic)

    # pre-flight: START → classifier → supervisor (both unconditional)
    graph.add_edge(START,        "classifier")
    graph.add_edge("classifier", "supervisor")

    # supervisor routes conditionally
    graph.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "researcher": "researcher",
            "coder":      "coder",
            "critic":     "critic",
            END:          END,
        }
    )

    # all workers return to supervisor — unconditional back-edges form the loop
    for worker in WORKERS:
        graph.add_edge(worker, "supervisor")

    return graph


app = build_graph().compile()
