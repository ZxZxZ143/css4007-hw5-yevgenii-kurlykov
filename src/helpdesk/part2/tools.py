"""Three direct local functions plus thin model-callable adapters."""
import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from .state import HelpdeskAgentState, HelpdeskContext, SessionContext, ToolResultsState
from ..paths import ROOT
from .ticket_policy import redact


def read_fixture(name: str) -> dict:
    return json.loads((ROOT / "data" / name).read_text(encoding="utf-8"))


def find_instructions(topic: str) -> dict:
    if topic not in ("wifi", "password"):
        raise ValueError("Supported topics: wifi, password")
    guide = read_fixture("helpdesk.json")["instructions"][topic]
    return {"topic": topic, **guide, "source": "data/helpdesk.json", "fictional": True}


def check_service_status(service: str, status_profile: str) -> dict:
    services = read_fixture("helpdesk.json")["topic_services"]
    service = services.get(service, service)
    if service not in ("wifi", "account_recovery"):
        raise ValueError("Supported services: wifi, account_recovery")
    profiles = read_fixture("service_status.json")["profiles"]
    if status_profile not in profiles:
        raise ValueError("Unsupported status profile")
    record = profiles[status_profile].get(service, {})
    return {"service": service, "status_profile": status_profile,
            "status": record.get("status", "unknown"),
            "message": record.get("message", "No status data is available."),
            "source": "data/service_status.json", "fictional": True}


def prepare_ticket(context: SessionContext, helpdesk: HelpdeskContext,
                   evidence: ToolResultsState, *, tool_call_id: str, turn: int) -> dict:
    """Validate trusted user-derived context at the actual file-writing boundary."""
    if helpdesk.get("secret_detected") or redact(json.dumps(helpdesk)) != json.dumps(helpdesk):
        return {"created": False, "reason": "Remove credentials and describe the problem without secrets."}
    if not helpdesk.get("ticket_requested"):
        return {"created": False, "reason": "Ask the user whether a local ticket draft is wanted."}
    if helpdesk.get("ticket_condition") == "unclear":
        return {"created": False, "reason": "Clarify the user's ticket condition before creating a draft."}
    if (helpdesk.get("category") not in ("wifi", "password")
            or len(helpdesk.get("problem_description", "").strip()) < 10
            or not helpdesk.get("steps_known")):
        return {"created": False, "reason": "Ask for the issue (Wi-Fi or password), a problem description, and steps already tried (or none)."}
    service = read_fixture("helpdesk.json")["topic_services"][helpdesk["category"]]
    statuses = [entry["data"] for entry in evidence.values()
                if entry["tool"] == "check_service_status" and entry["executed"]
                and entry["turn"] == turn and entry["data"].get("service") == service
                and entry["data"].get("status_profile") == context.status_profile
                and entry["data"].get("status") in ("operational", "outage", "unknown")]
    known = statuses[-1]["status"] if statuses else "not_checked"
    if helpdesk.get("ticket_condition") == "operational" and known != "operational":
        return {"created": False, "reason": "The user's condition requires a completed operational status check in this turn.",
                "service": service, "service_status": known}
    refs = sorted({entry["data"]["instruction_id"] for entry in evidence.values()
                   if entry["tool"] == "find_instructions" and entry["data"].get("topic") == helpdesk["category"]
                   and "instruction_id" in entry["data"]})
    key = json.dumps([context.run_id, context.user_id, context.thread_id, tool_call_id])
    ticket_id = "DRAFT-" + uuid5(NAMESPACE_URL, key).hex
    ticket = read_fixture("ticket_template.json")["ticket"]
    ticket.update(ticket_id=ticket_id, user_id=context.user_id, thread_id=context.thread_id,
                  category=helpdesk["category"], summary=helpdesk["problem_description"],
                  steps_tried=helpdesk["steps_tried"], service_status=known,
                  status_profile=context.status_profile, reference_ids=refs)
    directory = Path(context.storage_dir) / "tickets"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{ticket_id}.json"
    # Exclusive creation and stable call identity make same-call retries idempotent.
    try:
        with path.open("x", encoding="utf-8") as file:
            json.dump(ticket, file, indent=2, ensure_ascii=False)
            file.write("\n")
    except FileExistsError:
        if json.loads(path.read_text(encoding="utf-8")) != ticket:
            return {"created": False, "reason": "An existing draft conflicts with this retry."}
    return {"created": True, "ticket_id": ticket_id, "path": str(path), "delivery": "local draft only"}


def agent_tools():
    @tool("find_instructions")
    def instructions_tool(topic: str) -> dict:
        """Retrieve the complete fictional local guide for topic wifi or password."""
        return find_instructions(topic)

    @tool("check_service_status")
    def status_tool(service: str, runtime: ToolRuntime[SessionContext, HelpdeskAgentState]) -> dict:
        """Check fictional wifi or account_recovery status in the application's active snapshot."""
        return check_service_status(service, runtime.context.status_profile)

    @tool("prepare_ticket")
    def ticket_tool(runtime: ToolRuntime[SessionContext, HelpdeskAgentState]) -> dict:
        """Prepare a local draft using reported user details. Call after dependent status checks finish.

        Missing details or unfulfilled user conditions return a clarification reason; nothing is submitted.
        """
        return prepare_ticket(runtime.context, runtime.state["helpdesk"], runtime.state["tool_results"],
                              tool_call_id=runtime.tool_call_id, turn=runtime.state["control"]["turn"])

    return [instructions_tool, status_tool, ticket_tool]
