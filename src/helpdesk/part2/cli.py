"""Part 2 imports are lazy so existing Part 1 needs no model/provider setup."""
import json
import os
from pathlib import Path
import sys

from .agent import ConfigurationError, build_agent_graph
from .runner import AgentSession, run_scenario
from .state import SessionContext
from ..paths import ROOT


def run_agent_cli(args) -> int:
    try:
        if args.scenario:
            log = run_scenario(args.scenario, output=args.output, storage_dir=args.storage_dir)
            print(json.dumps(log, indent=2, ensure_ascii=False))
            return 1 if any(t["control"].get("stop_reason") not in ("final_response",) for t in log["turns"]) else 0
        graph = build_agent_graph()
        context = SessionContext(args.user_id, args.thread_id, args.status_profile,
                                 args.storage_dir or Path(os.getenv("HELPDESK_STORAGE_DIR", ROOT / "local-storage")))
        session = AgentSession(graph, context, model_name=os.getenv("MODEL_NAME", "unconfigured"))
        if args.interactive:
            print("Fictional helpdesk; local drafts only. Enter /quit to end. Context lasts for this process.")
            while True:
                try:
                    message = input("You: ")
                except EOFError:
                    break
                if message.strip() == "/quit":
                    break
                if message.strip():
                    print(json.dumps(session.turn(message)["final_response"], ensure_ascii=False))
                    if args.output:
                        session.save(args.output)
        else:
            session.turn(args.message)
            print(json.dumps(session.log, indent=2, ensure_ascii=False))
        if args.output:
            session.save(args.output)
        return int(any(t["control"].get("stop_reason") != "final_response" for t in session.log["turns"]))
    except ConfigurationError as error:
        print(str(error), file=sys.stderr)
        return 2
