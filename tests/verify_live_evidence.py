"""Verify saved live artifacts without making API calls: python -m tests.verify_live_evidence."""
import json
from pathlib import Path

from helpdesk.agent_runner import PART2_SCENARIOS
from helpdesk.tools import read_fixture


def main():
    summaries = []
    actual_models = set()
    fixtures = {s["id"]: s for s in read_fixture("scenarios.json")["scenarios"]}
    for scenario in PART2_SCENARIOS:
        path = Path("outputs/part2") / f"{scenario}.json"
        log = json.loads(path.read_text(encoding="utf-8"))
        fixture = fixtures[scenario]
        assert log["evidence_kind"] == "live_openai" and log["provider"] == "OpenAI"
        assert log["status_profile"] == fixture["status_profile"]
        assert [t["user_message"] for t in log["turns"]] == [e["message"] for e in fixture["events"]]
        all_evidence = {}
        tokens = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        batches = []
        drafts = []
        for turn in log["turns"]:
            assert turn["control"]["stop_reason"] == "final_response"
            assert not turn["control"]["limit_reached"] and not turn["control"]["errors"]
            assert len(turn["token_usage"]) == turn["control"]["model_calls"]
            pending = set()
            prior_step_evidence = {}
            for step in turn["steps"]:
                batch_evidence = {}
                for update in step["node_updates"]:
                    chunks = update["update"] if isinstance(update["update"], list) else [update["update"]]
                    for chunk in chunks:
                        batch_evidence.update(chunk.get("tool_results", {}))
                        for message in chunk.get("messages", []):
                            if message["type"] == "ai":
                                assert not pending
                                calls = message.get("tool_calls", [])
                                pending = {c["id"] for c in calls}
                                if calls:
                                    batches.append([c["name"] for c in calls])
                                actual_models.add(message["response_metadata"]["model_name"])
                            elif message["type"] == "tool":
                                assert message["tool_call_id"] in pending
                                pending.remove(message["tool_call_id"])
                                assert message["status"] == "success"
                for entry in batch_evidence.values():
                    data = entry["data"]
                    if data.get("created"):
                        ticket_path = Path(data["path"])
                        assert ticket_path.is_file()
                        ticket = json.loads(ticket_path.read_text(encoding="utf-8"))
                        assert set(ticket) == set(read_fixture("ticket_template.json")["ticket"])
                        assert ticket["status"] == "draft" and ticket["user_id"] == log["user_id"]
                        assert ticket["thread_id"] == log["thread_id"] and ticket["status_profile"] == log["status_profile"]
                        assert ticket["steps_tried"] == step["state_after"]["helpdesk"]["steps_tried"]
                        assert all(span in turn["user_message"] for span in ticket["steps_tried"])
                        if scenario == "S06":
                            assert any(e["data"].get("service") == "wifi" and e["data"].get("status") == "operational"
                                       for e in prior_step_evidence.values())
                            assert ticket["service_status"] == "operational"
                        assert ticket["ticket_id"] in turn["final_response"]
                        drafts.append(str(ticket_path))
                prior_step_evidence.update(batch_evidence)
                all_evidence.update(batch_evidence)
            assert not pending
            for usage in turn["token_usage"]:
                for key in tokens:
                    tokens[key] += usage[key]
        statuses = [e["data"] for e in all_evidence.values() if e["tool"] == "check_service_status"]
        expected = {"S06": "operational", "S07": "outage", "S08": "unknown", "S09": "operational", "S14": "outage"}[scenario]
        assert any(s["status"] == expected for s in statuses)
        assert len(drafts) == (1 if scenario in ("S06", "S09") else 0)
        if scenario == "S09":
            assert log["turns"][0]["control"]["tool_calls"] == 0
            assert "steps" in log["turns"][0]["final_response"].lower()
        if scenario == "S14":
            assert all(s["service"] == "account_recovery" for s in statuses)
        assert "encrypted_content" not in path.read_text(encoding="utf-8")
        summaries.append({"scenario_id": scenario, "verification_status": "passed_live_openai",
                          "log": str(path), "command": log["command"], "ticket_paths": drafts,
                          "model_calls": sum(t["control"]["model_calls"] for t in log["turns"]),
                          "tool_calls": sum(t["control"]["tool_calls"] for t in log["turns"]),
                          "token_usage": tokens, "tool_batches": batches})
    report = {"evidence_kind": "verified_saved_live_openai_executions", "provider": "OpenAI",
              "requested_model": "gpt-5-mini", "verified_openai_models": sorted(actual_models),
              "scenarios": summaries,
              "totals": {key: sum(s[key] for s in summaries) for key in ("model_calls", "tool_calls")},
              "token_usage": {key: sum(s["token_usage"][key] for s in summaries) for key in tokens},
              "note": "Totals cover the five final accepted traces. Initial S08/S09 runs are retained separately and excluded."}
    Path("outputs/part2/LIVE_STATUS.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"models": report["verified_openai_models"], **report["totals"], **report["token_usage"]}))


if __name__ == "__main__":
    main()
