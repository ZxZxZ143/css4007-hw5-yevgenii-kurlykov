from copy import deepcopy


def execute(graph, message: str, *, version: str, predicted_route: list[str], metadata: dict) -> dict:
    """Pair real node updates with LangGraph's complete values snapshots."""
    initial = {"message": message, "category": "", "reply": ""}
    steps = []
    pending = None
    final = initial
    for event in graph.stream(initial, stream_mode=["updates", "values"], version="v2"):
        if event["type"] == "updates":
            for node, update in event["data"].items():
                if pending is not None:
                    raise RuntimeError("Expected a full state snapshot after each sequential node")
                pending = {"node": node, "update": deepcopy(update)}
        elif event["type"] == "values":
            final = deepcopy(event["data"])
            if pending is not None:
                steps.append({**pending, "state_after": final})
                pending = None
    if pending is not None or not steps:
        raise RuntimeError("Incomplete LangGraph trace")
    nodes = [step["node"] for step in steps]
    return {
        **metadata,
        "graph_version": version,
        "graph_label": "Original / Initial Version" if version == "original" else "Improved / Main Version",
        "message": message,
        "predicted_route": predicted_route,
        "actual_execution_path": nodes,
        "nodes_executed": nodes,
        "boundary_note": "START and END are graph sentinels, not executed Python nodes. Stream exhausted normally.",
        "initial_state": initial,
        "steps": steps,
        "final_state": final,
        "final_response": final["reply"],
        "prediction_matched": predicted_route == nodes,
        "reference_source": "data/helpdesk.json" if nodes[-1] in ("wifi_help", "password_help") else None,
        "memory_used": False,
        "model_calls": [],
        "tool_calls": [],
    }
