"""Fresh process segment worker. Its input never contains an earlier transcript."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys

from ..part2.runner import serializable
from ..part2.state import SessionContext
from ..part2.tools import read_fixture
from .runner import PersistentSession


def main():
    request = json.load(sys.stdin)
    if request["model_mode"] not in ("openai", "fake"):
        raise ValueError("Unsupported model mode")
    scenario = next(s for s in read_fixture("scenarios.json")["scenarios"] if s["id"] == request["scenario_id"] and s["part"] == 3)
    events = []
    with ExitStack() as stack:
        sessions = {}
        for index in request["event_indices"]:
            event = scenario["events"][index]
            if event["action"] != "message":
                raise ValueError("Only new message events may be submitted to a worker")
            identity = (event["user_id"], event["thread_id"])
            if identity not in sessions:
                model, name, kind = None, None, "live_openai"
                if request["model_mode"] == "fake":
                    from tests.part3.fake_model import FakeMemoryModel
                    model, name, kind = FakeMemoryModel(), "FakeMemoryModel (offline; not OpenAI)", "deterministic_fake"
                context = SessionContext(*identity, scenario["status_profile"], Path(request["storage_dir"]))
                sessions[identity] = stack.enter_context(PersistentSession(context, model=model, model_name=name, evidence_kind=kind))
            session = sessions[identity]
            trace = session.turn(event["message"])
            events.append({"fixture_event_index": index, "user_id": identity[0], "thread_id": identity[1],
                           "process_id": os.getpid(), "status_profile": scenario["status_profile"], **trace})
    return {"process_id": os.getpid(), "events": events, "model": session.log["model"]}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        result = main()
    except Exception as error:
        print(json.dumps({"error": type(error).__name__}))
        raise SystemExit(1)
    print(json.dumps(serializable(result), ensure_ascii=False))
