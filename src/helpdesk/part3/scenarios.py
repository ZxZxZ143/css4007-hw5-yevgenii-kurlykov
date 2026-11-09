"""Fixture replay with actual process boundaries and isolated scenario storage."""
import json
import os
from pathlib import Path
import subprocess
import sys

from ..paths import ROOT, portable_path, repository_path
from ..part2.runner import serializable
from ..part2.tools import read_fixture
from .memory_store import UserMemoryStore
from .persistence import reset_thread

PART3_SCENARIOS = ("S10", "S11", "S12", "S13")


def run_memory_scenario(scenario_id: str, *, storage_dir=None, output=None, model_mode="openai") -> dict:
    if scenario_id not in PART3_SCENARIOS or model_mode not in ("openai", "fake"):
        raise ValueError("Unsupported Part 3 scenario or model mode")
    scenario = next(s for s in read_fixture("scenarios.json")["scenarios"] if s["id"] == scenario_id)
    base = repository_path(storage_dir or ROOT / "local-storage")
    directory = base / "part3-scenarios" / scenario_id
    messages = [e for e in scenario["events"] if e["action"] == "message"]
    if scenario["reset_before"]:
        for user, thread in {(e["user_id"], e["thread_id"]) for e in messages}:
            reset_thread(directory, user, thread)
        store = UserMemoryStore(directory)
        for user in {e["user_id"] for e in messages}:
            store.reset(user)
    segments, current = [], []
    for index, event in enumerate(scenario["events"]):
        if event["action"] == "restart_process":
            if not event["preserve_local_storage"]:
                raise ValueError("This runner supports restarts preserving scenario storage")
            segments.append(current)
            current = []
        else:
            current.append(index)
    segments.append(current)
    log = {"scenario_id": scenario_id, "evidence_kind": "live_openai" if model_mode == "openai" else "deterministic_fake",
           "provider": "OpenAI" if model_mode == "openai" else "deterministic fake (not OpenAI)",
           "command": f"python -m helpdesk.cli --scenario {scenario_id}" if model_mode == "openai" else "python -m scripts.part3.record_offline",
           "reset_before": scenario["reset_before"], "scenario_storage": portable_path(directory),
           "restart_occurred": len(segments) > 1, "processes": [], "events": []}
    for segment in segments:
        request = {"scenario_id": scenario_id, "event_indices": segment,
                   "storage_dir": str(directory), "model_mode": model_mode}
        command = [sys.executable, "-m", "helpdesk.part3.worker"]
        run = subprocess.run(command, input=json.dumps(request), text=True, encoding="utf-8", capture_output=True,
                             cwd=ROOT, env={**os.environ, "PYTHONIOENCODING": "utf-8"}, timeout=360)
        try:
            result = json.loads(run.stdout)
        except ValueError:
            raise RuntimeError("Memory scenario worker did not return valid sanitized evidence") from None
        if run.returncode != 0:
            raise RuntimeError("Memory scenario worker failed: " + result.get("error", "unknown error"))
        log["processes"].append({"pid": result["process_id"], "return_code": run.returncode,
                                 "exited_before_next_process": True, "event_indices": segment,
                                 "input_note": "Only these new fixture events were submitted; previous conversation was not copied."})
        log["events"].extend(result["events"])
        log["model"] = result["model"]
    if log["restart_occurred"]:
        assert len({process["pid"] for process in log["processes"]}) == len(segments)
    destination = repository_path(output or ROOT / "outputs/part3" / ("fake" if model_mode == "fake" else "") / f"{scenario_id}.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(serializable(log), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return log
