"""Part 1 predictions, scenario replay and deterministic graph execution."""
import json
import shlex

from ..paths import ROOT
from .graphs import build_main_graph, build_original_graph
from .trace import execute

# Fixed predictions recorded in outputs/part1/PREDICTIONS.md before any execution.
PREDICTIONS = {
    "S01": ("original", "wifi_help"),
    "S02": ("original", "password_help"),
    "S03": ("original", "other_help"),
    "S04": ("original", "wifi_help"),
    "S05": ("main", "clarify_problem"),
}


def predict(message: str, version: str) -> list[str]:
    """Pre-execution prediction for an ad-hoc message from the written rules."""
    text = message.casefold()
    wifi = "wifi" in text or "wi-fi" in text
    password = "password" in text
    node = "other_help"
    if version == "main" and wifi and password:
        node = "clarify_problem"
    elif wifi:
        node = "wifi_help"
    elif password:
        node = "password_help"
    return ["read_request", node]


def run_part1_cli(args, arguments, parser) -> int:
    metadata = {"scenario_id": None, "status_profile": None, "user_id": None, "thread_id": None}
    version = args.graph or "main"
    output = args.output
    if args.scenario:
        scenarios = json.loads((ROOT / "data/scenarios.json").read_text(encoding="utf-8"))["scenarios"]
        scenario = next(s for s in scenarios if s["id"] == args.scenario)
        version, target = PREDICTIONS[args.scenario]
        if args.graph and args.graph != version:
            parser.error(f"{args.scenario} requires --graph {version}")
        event = scenario["events"][0]
        message = event["message"]
        metadata = {"scenario_id": scenario["id"], "status_profile": scenario["status_profile"],
                    "user_id": event["user_id"], "thread_id": event["thread_id"],
                    "fixture_graph_variant": scenario.get("graph_variant"),
                    "reset_note": "Fresh stateless execution; Part 1 has no conversation or user storage."}
        predicted = ["read_request", target]
        output = output or ROOT / "outputs/part1" / ("original" if version == "original" else "improved") / f"{args.scenario}.json"
    else:
        message = args.message
        predicted = predict(message, version)
    metadata["command"] = "python -m helpdesk.cli " + shlex.join(arguments)
    graph = build_original_graph() if version == "original" else build_main_graph()
    result = execute(graph, message, version=version, predicted_route=predicted, metadata=metadata)
    serialized = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if result["prediction_matched"] else 1
