"""In-process sessions and genuine streamed Part 2 evidence; no persistent memory."""
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import shlex

from langchain_core.messages import BaseMessage, HumanMessage

from .agent import build_agent_graph, initial_agent_state
from .state import SessionContext
from ..paths import ROOT
from .ticket_policy import derive_helpdesk, redact

PART2_SCENARIOS = ("S06", "S07", "S08", "S09", "S14")


def serializable(value):
    if isinstance(value, BaseMessage):
        return serializable(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {k: "[REDACTED SECRET]" if re.fullmatch(r"(?i)(?:.*api[_-]?key|password|recovery[_-]?code)", k)
                else serializable(v) for k, v in value.items() if k != "encrypted_content"}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value if not (isinstance(v, dict) and v.get("type") == "reasoning")]
    if isinstance(value, str):
        return redact(value)
    return value


def response_text(content) -> str:
    """Responses API content may include non-display protocol blocks."""
    if isinstance(content, str):
        return content
    return "\n".join(block["text"] for block in content
                     if isinstance(block, dict) and block.get("type") == "text")


class AgentSession:
    def __init__(self, graph, context: SessionContext, *, model_name: str, evidence_kind="live_openai"):
        self.graph, self.context = graph, context
        self.state = initial_agent_state()
        self.log = {"evidence_kind": evidence_kind, "provider": "OpenAI" if evidence_kind == "live_openai" else "deterministic fake (not OpenAI)",
                    "model": model_name, "user_id": context.user_id, "thread_id": context.thread_id,
                    "status_profile": context.status_profile, "turns": []}

    def turn(self, message: str) -> dict:
        # Secrets are removed before sending to a provider or capturing any state/log.
        self.state["messages"] = [*self.state["messages"], HumanMessage(content=redact(message))]
        self.state["helpdesk"] = derive_helpdesk(self.state["messages"], self.state["helpdesk"])
        self.state["control"] = {"turn": self.state["control"]["turn"] + 1, "model_calls": 0,
                                 "tool_calls": 0, "tool_attempts": 0, "limit_reached": False, "stop_reason": None, "errors": []}
        trace = {"user_message": redact(message), "steps": []}
        pending = []
        for event in self.graph.stream(self.state, context=self.context,
                                       stream_mode=["updates", "values"], version="v2"):
            if event["type"] == "updates":
                for node, update in event["data"].items():
                    pending.append({"node": node, "update": serializable(update)})
            elif event["type"] == "values":
                self.state = deepcopy(event["data"])
                if pending:
                    snapshot = serializable({key: value for key, value in self.state.items() if key != "messages"})
                    snapshot["message_index"] = [{"type": m.type, "id": m.id,
                                                  **({"tool_call_id": m.tool_call_id} if m.type == "tool" else {})}
                                                 for m in self.state["messages"]]
                    trace["steps"].append({"node_updates": pending, "state_after": snapshot})
                    pending = []
        trace["nodes_executed"] = [item["node"] for step in trace["steps"] for item in step["node_updates"]]
        trace["control"] = dict(self.state["control"])
        last = self.state["messages"][-1]
        trace["final_response"] = redact(response_text(last.content))
        # Usage is read from actual AIMessage metadata, never estimated for fake models.
        trace["token_usage"] = [item.get("usage_metadata") for step in trace["steps"]
                                for update in step["node_updates"] if update["node"] == "agent"
                                for item in update["update"].get("messages", []) if item.get("usage_metadata")]
        self.log["turns"].append(trace)
        return trace

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(serializable(self.log), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_scenario(scenario_id: str, *, model=None, model_name=None, output=None,
                 storage_dir=None, evidence_kind="live_openai") -> dict:
    if scenario_id not in PART2_SCENARIOS:
        raise ValueError("Only Part 2 scenarios are supported by this runner")
    scenarios = json.loads((ROOT / "data/scenarios.json").read_text(encoding="utf-8"))["scenarios"]
    scenario = next(s for s in scenarios if s["id"] == scenario_id)
    first = scenario["events"][0]
    graph = build_agent_graph(model)
    context = SessionContext(first["user_id"], first["thread_id"], scenario["status_profile"],
                             Path(storage_dir or os.getenv("HELPDESK_STORAGE_DIR", ROOT / "local-storage")))
    session = AgentSession(graph, context, model_name=model_name or os.getenv("MODEL_NAME", "unconfigured"), evidence_kind=evidence_kind)
    command = ["python", "-m", "helpdesk.cli", "--scenario", scenario_id]
    if storage_dir is not None:
        command.extend(["--storage-dir", str(storage_dir)])
    if output is not None:
        command.extend(["--output", str(output)])
    session.log.update(scenario_id=scenario_id, reset_before=scenario["reset_before"],
                       reset_note="New isolated in-process state; no persistent conversation or user store.",
                       command=shlex.join(command))
    for event in scenario["events"]:
        if event["action"] != "message":
            raise ValueError("Process restart scenarios belong to Part 3")
        session.turn(event["message"])
    destination = Path(output) if output else ROOT / "outputs/part2" / f"{scenario_id}.json"
    session.save(destination)
    return session.log
