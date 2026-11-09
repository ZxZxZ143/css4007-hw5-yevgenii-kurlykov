import json
from pathlib import Path
from threading import Barrier

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
import pytest

from helpdesk.part2.agent import ConfigurationError, build_agent_graph, openai_model
from helpdesk.part2.runner import AgentSession, PART2_SCENARIOS, run_scenario
from helpdesk.part2.state import SessionContext, merge_results
from helpdesk.cli import main
from helpdesk.paths import ROOT
from helpdesk.part2.ticket_policy import derive_helpdesk, redact
from helpdesk.part2.tools import agent_tools, check_service_status, find_instructions, prepare_ticket, read_fixture
from tests.part2.fake_model import DemoFakeModel, ScriptedModel, call

CONDITIONAL = "I enabled Wi-Fi and forgot the saved network, but Campus-Demo still will not connect. Check whether Wi-Fi is down. If it is operating, prepare a ticket about this problem."
UNCONDITIONAL = "My issue is Wi-Fi. I enabled it and tried reconnecting, but it still fails. Prepare a ticket."


def session(tmp_path, model, profile="normal", **budgets):
    return AgentSession(build_agent_graph(model, **budgets),
                        SessionContext("demo-user-a", "test", profile, tmp_path),
                        model_name="deterministic-test", evidence_kind="deterministic_fake")


def assert_protocol(messages):
    pending = set()
    for message in messages:
        if isinstance(message, AIMessage):
            assert not pending, "Model received unresolved calls"
            pending = {c["id"] for c in message.tool_calls}
        if isinstance(message, ToolMessage):
            assert message.tool_call_id in pending
            pending.remove(message.tool_call_id)
    assert not pending


@pytest.mark.parametrize("topic", ["wifi", "password"])
def test_direct_instructions(topic):
    actual = find_instructions(topic)
    guide = read_fixture("helpdesk.json")["instructions"][topic]
    assert {key: actual[key] for key in guide} == guide
    assert actual["source"] == "data/helpdesk.json"


@pytest.mark.parametrize("profile,service,status", [
    ("normal", "wifi", "operational"), ("normal", "account_recovery", "operational"),
    ("wifi_outage", "wifi", "outage"), ("wifi_outage", "account_recovery", "operational"),
    ("recovery_outage", "password", "outage"), ("recovery_outage", "wifi", "operational"),
    ("status_unavailable", "wifi", "unknown"), ("status_unavailable", "account_recovery", "unknown"),
])
def test_direct_status_profiles(profile, service, status):
    actual = check_service_status(service, profile)
    assert actual["status"] == status
    assert actual["status_profile"] == profile
    assert actual["service"] == ("account_recovery" if service == "password" else service)
    assert actual["fictional"] is True


@pytest.mark.parametrize("function,args", [
    (find_instructions, ("screen",)), (check_service_status, ("screen", "normal")),
    (check_service_status, ("wifi", "invented")),
])
def test_invalid_direct_input(function, args):
    with pytest.raises(ValueError):
        function(*args)


def test_direct_ticket_template_and_retry(tmp_path):
    context = SessionContext("demo-user-a", "direct", "normal", tmp_path)
    helpdesk = derive_helpdesk([HumanMessage(content=UNCONDITIONAL)])
    result = prepare_ticket(context, helpdesk, {}, tool_call_id="repeat", turn=1)
    assert result["created"]
    ticket = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
    assert set(ticket) == set(read_fixture("ticket_template.json")["ticket"])
    assert ticket["summary"] == "I enabled it and tried reconnecting, but it still fails."
    assert ticket["steps_tried"] == ["I enabled it and tried reconnecting, but it still fails."]
    assert ticket["service_status"] == "not_checked"
    assert ticket["user_id"] == "demo-user-a" and ticket["thread_id"] == "direct"
    assert ticket["status"] == "draft"
    assert prepare_ticket(context, helpdesk, {}, tool_call_id="repeat", turn=1) == result
    assert len(list(tmp_path.rglob("*.json"))) == 1


@pytest.mark.parametrize("text", ["Prepare a support ticket.", "My issue is Wi-Fi. Prepare a ticket.",
                                    "My issue is Wi-Fi. I tried reconnecting.",
                                    "My issue is Wi-Fi. I tried reconnecting. Prepare a ticket only if the administrator agrees."])
def test_ticket_missing_details_or_intent(tmp_path, text):
    context = SessionContext("a", "b", "normal", tmp_path)
    result = prepare_ticket(context, derive_helpdesk([HumanMessage(content=text)]), {}, tool_call_id="x", turn=1)
    assert result["created"] is False
    assert not list(tmp_path.rglob("*.json"))


@pytest.mark.parametrize("status,turn,profile,created", [
    ("operational", 1, "normal", True), ("outage", 1, "normal", False),
    ("unknown", 1, "normal", False), ("operational", 0, "normal", False),
    ("operational", 1, "wifi_outage", False),
])
def test_ticket_requires_verified_current_workflow(tmp_path, status, turn, profile, created):
    context = SessionContext("a", "b", "normal", tmp_path)
    evidence = {"status": {"tool": "check_service_status", "turn": turn, "executed": True,
                            "data": {"service": "wifi", "status": status, "status_profile": profile}}}
    result = prepare_ticket(context, derive_helpdesk([HumanMessage(content=CONDITIONAL)]), evidence, tool_call_id="x", turn=1)
    assert result["created"] is created


def test_missing_status_and_wrong_service_do_not_satisfy_condition(tmp_path):
    ctx = SessionContext("a", "b", "normal", tmp_path)
    helpdesk = derive_helpdesk([HumanMessage(content=CONDITIONAL)])
    for results in ({}, {"x": {"tool": "check_service_status", "turn": 1, "executed": True,
                               "data": {"service": "account_recovery", "status": "operational", "status_profile": "normal"}}}):
        assert not prepare_ticket(ctx, helpdesk, results, tool_call_id="x", turn=1)["created"]


@pytest.mark.parametrize("secret", ["My password is Secret123", "API key: abcdefghi", "recovery code=123456", "sk-testabcdefghijk"])
def test_secrets_not_written_or_sent(tmp_path, secret):
    model = ScriptedModel([AIMessage(content="", tool_calls=[call("prepare_ticket")]), AIMessage(content="Please omit secrets.")])
    run = session(tmp_path, model)
    text = UNCONDITIONAL + " " + secret
    run.turn(text)
    run.save(tmp_path / "trace.json")
    assert not list((tmp_path / "tickets").glob("*"))
    assert secret not in (tmp_path / "trace.json").read_text(encoding="utf-8")
    assert secret not in str(model.received)
    assert not prepare_ticket(run.context, derive_helpdesk([HumanMessage(content=text)]), {}, tool_call_id="x", turn=1)["created"]


def test_parallel_tools_really_overlap_and_reach_model(tmp_path, monkeypatch):
    import helpdesk.part2.tools as module
    barrier = Barrier(2)
    instructions, status = module.find_instructions, module.check_service_status
    def first(topic):
        barrier.wait(timeout=5)
        return instructions(topic)
    def second(service, profile):
        barrier.wait(timeout=5)
        return status(service, profile)
    monkeypatch.setattr(module, "find_instructions", first)
    monkeypatch.setattr(module, "check_service_status", second)
    model = ScriptedModel([AIMessage(content="", tool_calls=[call("find_instructions", {"topic": "wifi"}, "guide"),
                                                            call("check_service_status", {"service": "wifi"}, "status")]),
                           AIMessage(content="Both local results received.")])
    run = session(tmp_path, model)
    result = run.turn("Show Wi-Fi instructions and check status.")
    assert result["nodes_executed"] == ["agent", "tools", "agent"]
    assert set(run.state["tool_results"]) == {"guide", "status"}
    assert run.state["tool_results"]["guide"]["data"]["instruction_id"] == "GUIDE-WIFI"
    assert "steps" not in run.state["tool_results"]["guide"]["data"]
    tool_messages = [m for m in model.received[-1] if isinstance(m, ToolMessage)]
    assert {m.tool_call_id for m in tool_messages} == {"guide", "status"}
    assert len(tool_messages) == 2
    assert result["control"]["tool_calls"] == 2
    assert_protocol(run.state["messages"])


def test_same_batch_status_cannot_authorize_ticket(tmp_path):
    model = ScriptedModel([AIMessage(content="", tool_calls=[call("check_service_status", {"service": "wifi"}, "status"),
                                                            call("prepare_ticket", call_id="premature")]),
                           AIMessage(content="", tool_calls=[call("prepare_ticket", call_id="later")]),
                           AIMessage(content="Read the resulting local draft.")])
    run = session(tmp_path, model)
    run.turn(CONDITIONAL)
    data = run.state["tool_results"]
    assert data["premature"]["data"]["created"] is False
    assert data["premature"]["data"]["service_status"] == "not_checked"
    assert data["later"]["data"]["created"] is True
    assert len(list(tmp_path.rglob("DRAFT-*.json"))) == 1
    assert_protocol(run.state["messages"])


@pytest.mark.parametrize("bad_call", [call("find_instructions", {"topic": "screen"}),
                                       call("check_service_status", {"unexpected": True}),
                                       call("not_a_tool")])
def test_tool_errors_close_protocol(tmp_path, bad_call):
    model = ScriptedModel([AIMessage(content="", tool_calls=[bad_call]), AIMessage(content="Please clarify.")])
    run = session(tmp_path, model)
    run.turn("Help with Wi-Fi")
    result = run.state["tool_results"][bad_call["id"]]
    assert "error" in result["data"]
    assert_protocol(run.state["messages"])


def test_tool_exception_is_safe(tmp_path, monkeypatch):
    import helpdesk.part2.tools as module
    def fail(topic):
        raise OSError("sensitive internal detail")
    monkeypatch.setattr(module, "find_instructions", fail)
    run = session(tmp_path, ScriptedModel([AIMessage(content="", tool_calls=[call("find_instructions", {"topic": "wifi"})]), AIMessage(content="Could not read the guide.")]))
    run.turn("wifi")
    assert "sensitive internal detail" not in str(run.log)
    assert_protocol(run.state["messages"])


@pytest.mark.parametrize("model_budget,tool_budget,batch,reason", [
    (1, 5, 1, "invocation_budget"), (6, 1, 2, "tool_batch_budget"), (6, 1, 1, "invocation_budget"),
])
def test_budgets_close_protocol(tmp_path, model_budget, tool_budget, batch, reason):
    model = ScriptedModel([AIMessage(content="", tool_calls=[call("find_instructions", {"topic": "wifi"}, str(i)) for i in range(batch)])])
    run = session(tmp_path, model, max_model_calls=model_budget, max_tool_calls=tool_budget)
    result = run.turn("wifi")
    assert result["control"]["limit_reached"]
    assert result["control"]["stop_reason"] == reason
    assert result["control"]["model_calls"] <= model_budget
    assert result["control"]["tool_calls"] <= tool_budget
    assert_protocol(run.state["messages"])


def test_repeated_tool_loop_terminates(tmp_path):
    model = ScriptedModel([AIMessage(content="", tool_calls=[call("find_instructions", {"topic": "wifi"})]) for _ in range(12)])
    run = session(tmp_path, model)
    result = run.turn("wifi")
    assert result["control"]["tool_calls"] == 5
    assert result["control"]["model_calls"] == 5
    assert result["control"]["limit_reached"]
    assert_protocol(run.state["messages"])


def test_model_error_safe_and_counted(tmp_path):
    run = session(tmp_path, ScriptedModel([RuntimeError("credential secret must not appear")]))
    result = run.turn("wifi")
    assert result["control"]["stop_reason"] == "model_error"
    assert result["control"]["model_calls"] == 1
    assert "credential secret" not in str(run.log)
    assert_protocol(run.state["messages"])


def test_runtime_config_not_in_model_schema():
    tools = agent_tools()
    schemas = {tool.name: tool.tool_call_schema.model_json_schema().get("properties", {}) for tool in tools}
    assert set(schemas["find_instructions"]) == {"topic"}
    assert set(schemas["check_service_status"]) == {"service"}
    assert schemas["prepare_ticket"] == {}


def test_model_cannot_switch_snapshot_or_fabricate_ticket_details(tmp_path):
    model = ScriptedModel([AIMessage(content="", tool_calls=[call("check_service_status", {"service": "wifi", "status_profile": "normal"})]),
                           AIMessage(content="", tool_calls=[call("prepare_ticket", {"ticket_condition": "unconditional", "steps_tried": ["Invented"]})]),
                           AIMessage(content="No ticket created.")])
    run = session(tmp_path, model, profile="wifi_outage")
    run.turn(CONDITIONAL)
    assert any(e["data"].get("status") == "outage" for e in run.state["tool_results"].values())
    assert not list(tmp_path.rglob("DRAFT-*.json"))
    assert_protocol(run.state["messages"])


def test_reducer_preserves_and_rejects_conflicting_ids():
    first = {"a": {"tool": "x", "data": {}, "executed": True, "turn": 1}}
    second = {"b": {"tool": "y", "data": {}, "executed": True, "turn": 1}}
    assert set(merge_results(first, second)) == {"a", "b"}
    assert set(first) == {"a"}
    with pytest.raises(ValueError):
        merge_results(first, {"a": second["b"]})


@pytest.mark.parametrize("scenario", PART2_SCENARIOS)
def test_scenarios(tmp_path, scenario):
    output = tmp_path / f"{scenario}.json"
    log = run_scenario(scenario, model=DemoFakeModel(), model_name="DemoFakeModel (offline)",
                       evidence_kind="deterministic_fake", output=output, storage_dir=tmp_path)
    assert log["evidence_kind"] == "deterministic_fake"
    fixture = next(s for s in read_fixture("scenarios.json")["scenarios"] if s["id"] == scenario)
    assert [t["user_message"] for t in log["turns"]] == [e["message"] for e in fixture["events"]]
    assert all(t["control"]["stop_reason"] == "final_response" for t in log["turns"])
    tickets = list(tmp_path.rglob("DRAFT-*.json"))
    assert len(tickets) == (1 if scenario in ("S06", "S09") else 0)
    for turn in log["turns"]:
        assert "agent" in turn["nodes_executed"]
        assert turn["token_usage"] == []
    if scenario == "S06":
        ticket = json.loads(tickets[0].read_text(encoding="utf-8"))
        assert ticket["service_status"] == "operational"
        assert ticket["reference_ids"] == ["GUIDE-WIFI"]
        assert ticket["steps_tried"] == [fixture["events"][1]["message"].split(". Check")[0] + "."]
        assert ticket["summary"] == ticket["steps_tried"][0]
    if scenario == "S09":
        assert log["turns"][0]["nodes_executed"] == ["agent"]
        assert "steps" in log["turns"][0]["final_response"]
        assert json.loads(tickets[0].read_text(encoding="utf-8"))["service_status"] == "not_checked"
    if scenario == "S14":
        assert "account_recovery" in log["turns"][0]["final_response"]
        assert "outage" in log["turns"][0]["final_response"]
    assert json.loads(output.read_text(encoding="utf-8")) == log


def test_sessions_do_not_leak_context(tmp_path):
    first = session(tmp_path / "first", DemoFakeModel())
    second = session(tmp_path / "second", DemoFakeModel())
    first.turn(UNCONDITIONAL)
    assert not second.state["helpdesk"]
    assert second.state["tool_results"] == {}
    assert not any(isinstance(m, HumanMessage) for m in second.state["messages"])


def test_missing_configuration_is_safe(monkeypatch, tmp_path, capsys):
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)
    with pytest.raises(ConfigurationError):
        openai_model()
    assert main(["--graph", "agent", "--message", "wifi"]) == 2
    assert "Live verification is pending" in capsys.readouterr().err


@pytest.mark.parametrize("condition", [
    "If the administrator agrees. Prepare a ticket.",
    "Prepare a ticket when the service is operating.",
    "If it is operating and someone approves, prepare a ticket.",
    "Do not prepare a ticket.",
    "What is a support ticket?",
])
def test_ambiguous_or_absent_authorization_never_creates(tmp_path, condition):
    text = "My issue is Wi-Fi. I tried reconnecting but it failed. " + condition
    context = derive_helpdesk([HumanMessage(content=text)])
    evidence = {"s": {"tool": "check_service_status", "turn": 1, "executed": True,
                      "data": {"service": "wifi", "status": "operational", "status_profile": "normal"}}}
    assert not prepare_ticket(SessionContext("a", "b", "normal", tmp_path), context, evidence,
                              tool_call_id="no", turn=1)["created"]


def test_separate_sentence_condition_remains_binding(tmp_path):
    text = "My issue is Wi-Fi. I tried reconnecting but it failed. If it is operating. Prepare a ticket."
    context = derive_helpdesk([HumanMessage(content=text)])
    assert context["ticket_condition"] == "operational"
    assert not prepare_ticket(SessionContext("a", "b", "normal", tmp_path), context, {}, tool_call_id="no", turn=1)["created"]


def test_followup_condition_and_prior_context_preserved():
    messages = [HumanMessage(content=CONDITIONAL), HumanMessage(content="Prepare the ticket please.")]
    context = derive_helpdesk(messages)
    assert context["ticket_condition"] == "operational"
    assert context["steps_tried"] == [CONDITIONAL.split(". Check")[0] + "."]
    context = derive_helpdesk([*messages, HumanMessage(content="Only if I approve first.")])
    assert context["ticket_condition"] == "unclear"


def test_unconditional_outage_ticket_is_allowed(tmp_path):
    context = derive_helpdesk([HumanMessage(content=UNCONDITIONAL)])
    evidence = {"s": {"tool": "check_service_status", "turn": 1, "executed": True,
                      "data": {"service": "wifi", "status": "outage", "status_profile": "wifi_outage"}}}
    result = prepare_ticket(SessionContext("a", "b", "wifi_outage", tmp_path), context, evidence, tool_call_id="yes", turn=1)
    assert result["created"]
    assert json.loads(Path(result["path"]).read_text())["service_status"] == "outage"


def test_missing_snapshot_record_stays_unknown(monkeypatch):
    import helpdesk.part2.tools as module
    read = module.read_fixture
    def missing(name):
        data = read(name)
        if name == "service_status.json":
            data["profiles"]["normal"].pop("wifi")
        return data
    monkeypatch.setattr(module, "read_fixture", missing)
    assert module.check_service_status("wifi", "normal")["status"] == "unknown"


def test_real_openai_adapter_configuration_offline(monkeypatch):
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setenv("OPENAI_API_KEY", "offline-placeholder-not-a-real-key")
    monkeypatch.setenv("MODEL_NAME", "configured-tool-model")
    model = openai_model()
    assert model.model_name == "configured-tool-model"
    assert model.use_responses_api is True
    assert model.max_retries == 0
    bound = model.bind_tools(agent_tools(), parallel_tool_calls=True)
    assert bound.kwargs["parallel_tool_calls"] is True
    assert len(bound.kwargs["tools"]) == 3


def test_usage_metadata_is_actual_message_usage(tmp_path):
    model = ScriptedModel([AIMessage(content="No tool needed.", usage_metadata={"input_tokens": 11, "output_tokens": 7, "total_tokens": 18})])
    trace = session(tmp_path, model).turn("hello")
    assert trace["token_usage"] == [{"input_tokens": 11, "output_tokens": 7, "total_tokens": 18}]


def test_cli_part2_interactive_and_scenario_dispatch(monkeypatch, tmp_path, capsys):
    import helpdesk.part2.agent as module
    monkeypatch.setattr(module, "openai_model", DemoFakeModel)
    messages = iter(["Prepare a support ticket.", "The issue is Wi-Fi. I enabled it and tried reconnecting, but it still fails.", "/quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(messages))
    output = tmp_path / "interactive.json"
    assert main(["--graph", "agent", "--interactive", "--output", str(output), "--storage-dir", str(tmp_path)]) == 0
    assert len(json.loads(output.read_text())["turns"]) == 2
    capsys.readouterr()
    assert main(["--scenario", "S14", "--output", str(tmp_path / "scenario.json"), "--storage-dir", str(tmp_path)]) == 0
    log = json.loads(capsys.readouterr().out)
    assert log["status_profile"] == "recovery_outage"


@pytest.mark.parametrize("arguments", [
    ["--scenario", "S06", "--graph", "main"],
    ["--scenario", "S01", "--graph", "agent"],
    ["--interactive"],
])
def test_cli_rejects_incompatible_modes(arguments):
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2


def test_changing_topic_cannot_reuse_steps_or_ticket_consent():
    result = derive_helpdesk([HumanMessage(content=UNCONDITIONAL), HumanMessage(content="My password recovery page is unavailable.")])
    assert result["category"] == "password"
    assert not result["steps_tried"] and not result["steps_known"]
    assert result["ticket_requested"] is False


def test_hypothetical_steps_are_not_known():
    result = derive_helpdesk([HumanMessage(content="My issue is Wi-Fi. If I tried reconnecting, would it help? Nothing works. Prepare a ticket.")])
    assert not result["steps_tried"]
    assert not result["steps_known"]


@pytest.mark.parametrize("response", [
    AIMessage(content="", tool_calls=[call("find_instructions", {"topic": "wifi"}, "same"), call("find_instructions", {"topic": "wifi"}, "same")]),
    AIMessage(content="", invalid_tool_calls=[{"name": "find_instructions", "args": "{malformed", "id": "bad", "error": "Invalid JSON", "type": "invalid_tool_call"}]),
])
def test_malformed_protocol_is_kept_out_of_model_history(tmp_path, response):
    run = session(tmp_path, ScriptedModel([response]))
    trace = run.turn("wifi")
    assert trace["control"]["stop_reason"] == "invalid_tool_calls"
    assert trace["control"]["rejected_model_output"]["type"] == "ai"
    assert trace["control"]["tool_attempts"] == 0
    assert_protocol(run.state["messages"])


def test_status_from_earlier_user_turn_cannot_authorize_ticket(tmp_path):
    model = ScriptedModel([AIMessage(content="", tool_calls=[call("check_service_status", {"service": "wifi"})]), AIMessage(content="Checked fictional status."),
                           AIMessage(content="", tool_calls=[call("prepare_ticket")]), AIMessage(content="Please check status again before the conditional draft.")])
    run = session(tmp_path, model)
    run.turn("Check Wi-Fi status.")
    run.turn(CONDITIONAL)
    assert not list(tmp_path.rglob("DRAFT-*.json"))
    assert any(e["data"].get("service_status") == "not_checked" for e in run.state["tool_results"].values())
    assert_protocol(run.state["messages"])


def test_responses_blocks_are_not_displayed_or_saved_as_final_text(tmp_path):
    model = ScriptedModel([AIMessage(content=[
        {"type": "reasoning", "id": "rs_test", "encrypted_content": "opaque-provider-protocol"},
        {"type": "text", "text": "Visible reply only."},
    ])])
    run = session(tmp_path, model)
    trace = run.turn("hello")
    run.save(tmp_path / "trace.json")
    assert trace["final_response"] == "Visible reply only."
    assert "opaque-provider-protocol" not in (tmp_path / "trace.json").read_text()
    assert run.state["messages"][-1].content[0]["encrypted_content"] == "opaque-provider-protocol"


def test_cli_unicode_output_on_legacy_windows_encoding():
    import os
    import subprocess
    import sys
    env = {**os.environ, "PYTHONIOENCODING": "ascii"}
    result = subprocess.run([sys.executable, "-m", "helpdesk.cli", "--message", "Wi-Fi \u2011 test"],
                            env=env, capture_output=True, check=True)
    assert json.loads(result.stdout.decode("utf-8"))["message"] == "Wi-Fi \u2011 test"
