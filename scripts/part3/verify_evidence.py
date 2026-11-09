"""Validate sanitized Part 3 evidence without API calls or private database files."""
import json
from pathlib import Path

from helpdesk.paths import ROOT, portable_path, repository_path
from helpdesk.part2.tools import read_fixture
from helpdesk.part3.persistence import checkpoint_key
from helpdesk.part3.scenarios import PART3_SCENARIOS


def verify_part3(directory=ROOT / "outputs/part3", *, evidence_kind="live_openai") -> dict:
    directory = repository_path(directory)
    fixtures = {s["id"]: s for s in read_fixture("scenarios.json")["scenarios"]}
    summaries = []
    models = set()
    for scenario in PART3_SCENARIOS:
        path = directory / f"{scenario}.json"
        log = json.loads(path.read_text(encoding="utf-8"))
        assert log["scenario_id"] == scenario and log["evidence_kind"] == evidence_kind
        fixture = fixtures[scenario]
        expected = [(i, e) for i, e in enumerate(fixture["events"]) if e["action"] == "message"]
        assert len(log["events"]) == len(expected) == 2
        tokens = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        calls = tools = memory_calls = 0
        for actual, (index, event) in zip(log["events"], expected, strict=True):
            assert actual["fixture_event_index"] == index and actual["user_message"] == event["message"]
            assert actual["user_id"] == event["user_id"] and actual["thread_id"] == event["thread_id"]
            assert actual["memory"]["checkpoint_key"] == checkpoint_key(event["user_id"], event["thread_id"])
            assert actual["memory"]["new_human_messages_submitted"] == 1
            assert actual["memory"]["message_counts_after"]["system"] == 1
            assert actual["control"]["stop_reason"] == "final_response"
            assert not actual["control"]["errors"] and not actual["control"]["limit_reached"]
            pending = set()
            for step in actual["steps"]:
                for update in step["node_updates"]:
                    chunks = update["update"] if isinstance(update["update"], list) else [update["update"]]
                    for chunk in chunks:
                        for message in chunk.get("messages", []):
                            if message["type"] == "ai":
                                assert not pending
                                pending = {c["id"] for c in message.get("tool_calls", [])}
                                if evidence_kind == "live_openai":
                                    models.add(message["response_metadata"]["model_name"])
                            elif message["type"] == "tool":
                                assert message["tool_call_id"] in pending
                                pending.remove(message["tool_call_id"])
                                assert message["status"] == "success"
                                if message["name"] == "save_user_memory":
                                    result = json.loads(message["content"])
                                    assert result["success"] and result["key"] == "preferred_language" and result["value"] == "kk"
                                    memory_calls += 1
            assert not pending
            if evidence_kind == "live_openai":
                assert len(actual["token_usage"]) == actual["control"]["model_calls"]
            else:
                assert not actual["token_usage"]
            for usage in actual["token_usage"]:
                for key in tokens:
                    tokens[key] += usage[key]
            calls += actual["control"]["model_calls"]
            tools += actual["control"]["tool_calls"]
        first, second = log["events"]
        assert not first["memory"]["checkpoint_restored"]
        if scenario in ("S10", "S11"):
            assert second["memory"]["checkpoint_restored"]
            assert second["memory"]["checkpoint_human_messages_before"] == 1
            assert second["memory"]["message_counts_after"]["human"] == 2
            assert second["memory"]["available_fact_sources"]["conversation_checkpoint"]["category"] == "wifi"
            assert second["memory"]["available_fact_sources"]["conversation_checkpoint"]["steps_tried"]
            assert first["control"]["conversation_id"] == second["control"]["conversation_id"]
        else:
            assert not second["memory"]["checkpoint_restored"]
            assert second["memory"]["checkpoint_human_messages_before"] == 0
            assert second["memory"]["message_counts_after"]["human"] == 1
            assert second["memory"]["available_fact_sources"]["conversation_checkpoint"] == {}
        if scenario in ("S11", "S12"):
            assert log["restart_occurred"] and len(log["processes"]) == 2
            assert first["process_id"] != second["process_id"]
            assert all(p["exited_before_next_process"] and p["return_code"] == 0 for p in log["processes"])
            assert first["memory"]["user_preferences_after"] == {"preferred_language": "kk"}
            assert any(r["preferences"].get("preferred_language") == "kk" for r in first["memory"]["model_context_reads"])
            assert second["memory"]["available_fact_sources"]["user_json_memory"] == {"preferred_language": "kk"}
            assert memory_calls >= 1
        if scenario == "S13":
            assert first["memory"]["user_preferences_after"] == {"preferred_language": "kk"}
            assert second["memory"]["user_preferences_after"] == {}
            assert second["memory"]["available_fact_sources"]["user_json_memory"] == {}
            assert memory_calls >= 1
            assert any(r["preferences"].get("preferred_language") == "kk" for r in first["memory"]["model_context_reads"])
        for process in log["processes"]:
            assert [e["fixture_event_index"] for e in log["events"] if e["process_id"] == process["pid"]] == process["event_indices"]
        summaries.append({"scenario_id": scenario, "log": portable_path(path), "verification_status": "passed",
                          "process_ids": [p["pid"] for p in log["processes"]], "restart_occurred": log["restart_occurred"],
                          "model_calls": calls, "tool_calls": tools, "memory_tool_calls": memory_calls, "token_usage": tokens})
    return {"evidence_kind": evidence_kind, "verified_models": sorted(models), "scenarios": summaries,
            "totals": {key: sum(s[key] for s in summaries) for key in ("model_calls", "tool_calls", "memory_tool_calls")},
            "token_usage": {key: sum(s["token_usage"][key] for s in summaries) for key in tokens}}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate without rewriting the saved summary")
    args = parser.parse_args()
    report = verify_part3()
    if not args.check:
        (ROOT / "outputs/part3/LIVE_STATUS.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"models": report["verified_models"], **report["totals"], **report["token_usage"]}))


if __name__ == "__main__":
    main()
