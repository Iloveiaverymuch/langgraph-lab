"""
Graph construction and compilation.

Topology:
    START → supervisor ⇢ search_worker  → supervisor
                      ⇢ analyst_worker  → supervisor
                      ⇢ writer_worker   → supervisor
                      ⇢ END

Solid edges  = unconditional (workers always return to supervisor)
Dashed edges = conditional   (supervisor routes based on state["next"])
"""

from langgraph.graph import StateGraph, START, END

from .state import AgentState
from .nodes import (
    supervisor_node,
    search_worker,
    analyst_worker,
    writer_worker,
    WORKERS,
)


def route_supervisor(state: AgentState) -> str:
    """
    Routing function for the conditional edge after supervisor.
    Reads state["next"] — written by supervisor_node — and returns the target node name.

    Kept separate from supervisor_node because:
    - nodes write facts into state
    - edges make decisions based on those facts
    """
    if state["next"] == "FINISH":
        return END
    return state["next"]


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("search_worker", search_worker)
    graph.add_node("analyst_worker", analyst_worker)
    graph.add_node("writer_worker", writer_worker)

    # entry point
    graph.add_edge(START, "supervisor")

    # supervisor routes conditionally to workers or END
    graph.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "search_worker": "search_worker",
            "analyst_worker": "analyst_worker",
            "writer_worker": "writer_worker",
            END: END,
        }
    )

    # all workers return to supervisor — this edge IS the loop
    for worker in WORKERS:
        graph.add_edge(worker, "supervisor")

    return graph


# compiled app — import this to run the graph
app = build_graph().compile()
