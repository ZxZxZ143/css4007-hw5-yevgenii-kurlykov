"""Generate explicitly OFFLINE evidence. Run: python -m scripts.part2.record_offline."""
import json
from pathlib import Path

from langchain_core.messages import HumanMessage

from helpdesk.part2.runner import PART2_SCENARIOS, run_scenario
from helpdesk.part2.state import SessionContext
from helpdesk.part2.ticket_policy import derive_helpdesk
from helpdesk.part2.tools import check_service_status, find_instructions, prepare_ticket
from helpdesk.paths import ROOT
from tests.part2.fake_model import DemoFakeModel


def main():
    directory = ROOT / "outputs/part2/fake"
    directory.mkdir(parents=True, exist_ok=True)
    for scenario in PART2_SCENARIOS:
        log = run_scenario(scenario, model=DemoFakeModel(), model_name="DemoFakeModel (offline; not OpenAI)",
                           evidence_kind="deterministic_fake", output=directory / f"{scenario}.json",
                           storage_dir=directory / "drafts" / scenario)
        # The recorded command describes the actual fake-evidence generator.
        log["command"] = "python -m scripts.part2.record_offline"
        (directory / f"{scenario}.json").write_text(json.dumps(log, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(scenario, [(t["control"]["model_calls"], t["control"]["tool_calls"]) for t in log["turns"]])
    context = SessionContext("demo-user-a", "direct-tool-demo", "normal", directory / "drafts/direct")
    text = "My issue is Wi-Fi. I tried reconnecting, but it still fails. Prepare a ticket."
    result = {
        "evidence_kind": "direct_python_functions_no_model",
        "command": "python -m scripts.part2.record_offline",
        "instructions": {topic: find_instructions(topic) for topic in ("wifi", "password")},
        "statuses": {profile: {service: check_service_status(service, profile) for service in ("wifi", "account_recovery")}
                     for profile in ("normal", "wifi_outage", "recovery_outage", "status_unavailable")},
        "ticket_input": text,
        "ticket_result": prepare_ticket(context, derive_helpdesk([HumanMessage(content=text)]), {},
                                        tool_call_id="direct-demo", turn=1),
    }
    (directory / "direct_tools.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
