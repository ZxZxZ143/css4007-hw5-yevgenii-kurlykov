import argparse
import json
import shlex
import sys
from pathlib import Path

from .graphs import build_main_graph, build_original_graph
from .nodes import ROOT
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="HW5 helpdesk: original/main (Part 1) or agent (Part 2)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--message")
    source.add_argument("--scenario", choices=(*PREDICTIONS, "S06", "S07", "S08", "S09", "S14"))
    source.add_argument("--interactive", action="store_true")
    parser.add_argument("--graph", choices=("original", "main", "agent"), default=None,
                        help="Default: main for messages; required fixture variant for scenarios")
    parser.add_argument("--output", type=Path, help="Save the complete JSON trace")
    parser.add_argument("--user-id", default="demo-user-a")
    parser.add_argument("--thread-id", default="demo-thread")
    parser.add_argument("--status-profile", choices=("normal", "wifi_outage", "recovery_outage", "status_unavailable"), default="normal")
    parser.add_argument("--storage-dir", type=Path)
    args = parser.parse_args(argv)
    if args.interactive and args.graph != "agent":
        parser.error("--interactive requires --graph agent")
    if args.graph == "agent" or (args.scenario and args.scenario not in PREDICTIONS):
        if args.scenario and args.graph and args.graph != "agent":
            parser.error(f"{args.scenario} requires --graph agent")
        if args.scenario in PREDICTIONS:
            parser.error("Part 1 scenarios require their original/main graph")
        from .agent_cli import run_agent_cli
        return run_agent_cli(args)
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
    arguments = sys.argv[1:] if argv is None else argv
    metadata["command"] = "python -m helpdesk.cli " + shlex.join(arguments)
    graph = build_original_graph() if version == "original" else build_main_graph()
    result = execute(graph, message, version=version, predicted_route=predicted, metadata=metadata)
    serialized = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if result["prediction_matched"] else 1


if __name__ == "__main__":
    # Model replies may contain characters outside Windows' legacy code page.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
