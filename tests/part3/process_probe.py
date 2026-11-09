"""Test-only fresh-process operations; never calls a real model."""
import json
import os
from pathlib import Path
import sys

from helpdesk.part2.runner import serializable
from helpdesk.part2.state import SessionContext
from helpdesk.part3.memory_store import UserMemoryStore
from helpdesk.part3.runner import PersistentSession
from tests.part3.fake_model import FakeMemoryModel


def main(request):
    directory = Path(request["storage_dir"])
    if request.get("operation") == "save_preference":
        return UserMemoryStore(directory).save(request["user_id"], request["key"], request["value"])
    context = SessionContext(request.get("user_id", "demo-user-a"), request.get("thread_id", "shared"),
                             request.get("status_profile", "normal"), directory)
    with PersistentSession(context, model=FakeMemoryModel(), model_name="offline", evidence_kind="deterministic_fake") as run:
        trace = run.turn(request["message"])
        return {"pid": os.getpid(), "trace": trace, "helpdesk": run.state["helpdesk"], "control": run.state["control"],
                "humans": [m.content for m in run.state["messages"] if m.type == "human"],
                "tool_ids": [m.tool_call_id for m in run.state["messages"] if m.type == "tool"],
                "drafts": sorted(p.name for p in directory.rglob("DRAFT-*.json"))}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(serializable(main(json.load(sys.stdin))), ensure_ascii=False))
