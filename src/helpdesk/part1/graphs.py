from functools import partial
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from .nodes import (
    GUIDE_PATH, clarify_problem, instruction_help, other_help,
    read_request_main, read_request_original,
)
from .state import HelpdeskState

ROUTES = {"wifi": "wifi_help", "password": "password_help", "other": "other_help", "both": "clarify_problem"}


def choose_route(state: HelpdeskState) -> str:
    """Select a conditional edge; do not mutate graph state."""
    return ROUTES[state["category"]]


def _build(*, improved: bool, guide_path: Path):
    graph = StateGraph(HelpdeskState)
    graph.add_node("read_request", read_request_main if improved else read_request_original)
    graph.add_node("wifi_help", partial(instruction_help, guide_path=guide_path))
    graph.add_node("password_help", partial(instruction_help, guide_path=guide_path))
    graph.add_node("other_help", other_help)
    destinations = ["wifi_help", "password_help", "other_help"]
    if improved:
        graph.add_node("clarify_problem", clarify_problem)
        destinations.append("clarify_problem")
    graph.add_edge(START, "read_request")
    graph.add_conditional_edges("read_request", choose_route, {name: name for name in destinations})
    for name in destinations:
        graph.add_edge(name, END)
    return graph.compile()


def build_original_graph(guide_path: Path = GUIDE_PATH):
    """Compile the independently executable Original / Initial Version."""
    return _build(improved=False, guide_path=guide_path)


def build_main_graph(guide_path: Path = GUIDE_PATH):
    """Compile the default Improved / Main Version."""
    return _build(improved=True, guide_path=guide_path)
