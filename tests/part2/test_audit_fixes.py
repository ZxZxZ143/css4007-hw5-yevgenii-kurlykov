from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path, PureWindowsPath
import shutil
from threading import Barrier

from langchain_core.messages import AIMessage, HumanMessage
import pytest

from helpdesk.paths import ROOT, evidence_path, portable_path, repository_path
from helpdesk.part2.agent import build_agent_graph
from helpdesk.part2.runner import AgentSession
from helpdesk.part2.state import SessionContext
from helpdesk.part2.ticket_policy import derive_helpdesk
from helpdesk.part2.tools import prepare_ticket
from scripts.part2.verify_live import verify_evidence
from tests.part2.fake_model import ScriptedModel, call

REQUEST = "My issue is Wi-Fi. I tried reconnecting but it failed. Prepare a ticket."
CONDITIONAL = "My issue is Wi-Fi. I tried reconnecting but it failed. If it is operating, prepare a ticket."


def helpdesk(text=REQUEST):
    return derive_helpdesk([HumanMessage(content=text)])


def status():
    return {"checked": {"tool": "check_service_status", "turn": 1, "executed": True,
                         "data": {"service": "wifi", "status": "operational", "status_profile": "normal"}}}


@pytest.mark.parametrize("second_call", ["first", "different"])
def test_request_idempotency_not_tool_call_id(tmp_path, second_call):
    ctx = SessionContext("demo-user-a", "one", "normal", tmp_path)
    issue = helpdesk()
    first = prepare_ticket(ctx, issue, {}, tool_call_id="first", turn=1)
    second = prepare_ticket(ctx, issue, {}, tool_call_id=second_call, turn=2)
    assert first == second
    assert evidence_path(first["path"]).is_file()
    assert len(list(tmp_path.rglob("DRAFT-*.json"))) == 1


def test_two_concurrent_publishers_return_one_complete_draft(tmp_path, monkeypatch):
    import helpdesk.part2.tools as tools
    ctx, issue = SessionContext("demo-user-a", "parallel", "normal", tmp_path), helpdesk()
    publish = tools.os.link
    barrier = Barrier(2)
    def simultaneous(source, target):
        barrier.wait(timeout=5)  # Both have fully written private files before either publishes.
        publish(source, target)
    monkeypatch.setattr(tools.os, "link", simultaneous)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda key: prepare_ticket(ctx, issue, {}, tool_call_id=key, turn=1), ("a", "b")))
    assert results[0] == results[1]
    files = list(tmp_path.rglob("DRAFT-*.json"))
    assert len(files) == 1 and not list(tmp_path.rglob("*.tmp"))
    assert json.loads(files[0].read_text())["ticket_id"] == results[0]["ticket_id"]


def test_parallel_toolnode_duplicate_calls_preserve_both_responses(tmp_path):
    model = ScriptedModel([
        AIMessage(content="", tool_calls=[call("prepare_ticket", {"ticket_request_id": "model-a"}, "a"),
                                          call("prepare_ticket", {"ticket_request_id": "model-b"}, "b")]),
        AIMessage(content="I invented a different saved problem."),
    ])
    run = AgentSession(build_agent_graph(model), SessionContext("demo-user-a", "parallel-graph", "normal", tmp_path),
                       model_name="offline", evidence_kind="deterministic_fake")
    trace = run.turn(REQUEST)
    results = run.state["tool_results"]
    assert set(results) == {"a", "b"}
    assert results["a"]["data"] == results["b"]["data"]
    assert len(list(tmp_path.rglob("DRAFT-*.json"))) == 1
    assert {m.tool_call_id for m in model.received[-1] if m.type == "tool"} == {"a", "b"}
    assert trace["response_source"] == "saved_ticket_fields"
    assert "invented" not in trace["final_response"]
    assert "saved_fields" not in results["a"]["data"]  # No duplicate saved facts in the compact State channel.


def test_followup_reuses_request_but_explicit_new_request_creates_another(tmp_path):
    responses = []
    for identifier in ("first", "followup", "new"):
        responses.extend([AIMessage(content="", tool_calls=[call("prepare_ticket", call_id=identifier)]),
                          AIMessage(content="Actual model reply retained.")])
    run = AgentSession(build_agent_graph(ScriptedModel(responses)), SessionContext("demo-user-a", "followups", "normal", tmp_path),
                       model_name="offline", evidence_kind="deterministic_fake")
    run.turn(REQUEST)
    original_request = run.state["helpdesk"]["ticket_request_id"]
    original_ticket = run.state["helpdesk"]["ticket_id"]
    run.turn("Please show the existing draft again.")
    assert run.state["helpdesk"]["ticket_request_id"] == original_request
    assert run.state["helpdesk"]["ticket_id"] == original_ticket
    assert len(list(tmp_path.rglob("DRAFT-*.json"))) == 1
    run.turn("Prepare a new ticket for this issue.")
    assert run.state["helpdesk"]["ticket_request_id"] != original_request
    assert run.state["helpdesk"]["ticket_id"] != original_ticket
    assert len(list(tmp_path.rglob("DRAFT-*.json"))) == 2


def test_existing_draft_is_not_overwritten_by_followup_details(tmp_path):
    ctx, issue = SessionContext("demo-user-a", "immutable", "normal", tmp_path), helpdesk(CONDITIONAL)
    first = prepare_ticket(ctx, issue, status(), tool_call_id="first", turn=1)
    path = evidence_path(first["path"])
    saved = path.read_bytes()
    changed = {**issue, "problem_description": "Changed follow-up description", "steps_tried": ["Additional later step"]}
    retry = prepare_ticket(ctx, changed, {}, tool_call_id="different", turn=2)
    assert retry == first and path.read_bytes() == saved
    assert retry["saved_fields"]["summary"] != changed["problem_description"]


@pytest.mark.parametrize("user,thread", [("a", "one"), ("b", "one"), ("a", "two")])
def test_logical_identity_scoped_by_user_and_thread(tmp_path, user, thread):
    issue = helpdesk()
    baseline = prepare_ticket(SessionContext("baseline", "baseline", "normal", tmp_path, run_id="same-run"),
                              issue, {}, tool_call_id="a", turn=1)
    result = prepare_ticket(SessionContext(user, thread, "normal", tmp_path, run_id="same-run"),
                            issue, {}, tool_call_id="a", turn=1)
    assert result["ticket_id"] != baseline["ticket_id"]
    ticket = json.loads(evidence_path(result["path"]).read_text())
    assert ticket["user_id"] == user and ticket["thread_id"] == thread


def test_rejected_conditional_request_can_retry_with_completed_status(tmp_path):
    ctx, issue = SessionContext("a", "b", "normal", tmp_path), helpdesk(CONDITIONAL)
    assert not prepare_ticket(ctx, issue, {}, tool_call_id="rejected", turn=1)["created"]
    assert not list(tmp_path.rglob("*"))
    result = prepare_ticket(ctx, issue, status(), tool_call_id="retry", turn=1)
    assert result["created"] and evidence_path(result["path"]).is_file()


@pytest.mark.parametrize("failure_point", ["dump", "link"])
def test_failed_private_write_or_publish_does_not_block_retry(tmp_path, monkeypatch, failure_point):
    import helpdesk.part2.tools as tools
    ctx, issue = SessionContext("a", "failure", "normal", tmp_path), helpdesk()
    def fail(*args, **kwargs):
        if failure_point == "dump":
            args[1].write('{"incomplete":')
        raise OSError("Simulated write failure")
    with monkeypatch.context() as patch:
        patch.setattr(tools.json if failure_point == "dump" else tools.os, failure_point, fail)
        with pytest.raises(OSError):
            prepare_ticket(ctx, issue, {}, tool_call_id="failure", turn=1)
    assert not list(tmp_path.rglob("DRAFT-*.json")) and not list(tmp_path.rglob("*.tmp"))
    result = prepare_ticket(ctx, issue, {}, tool_call_id="retry", turn=1)
    assert result["created"] and evidence_path(result["path"]).is_file()


@pytest.mark.parametrize("topic", ["Wi-Fi", "password recovery"])
def test_ticket_confirmation_reports_only_saved_fields_and_keeps_raw_model_output(tmp_path, topic):
    invented = "Saved: Campus-Demo, error E999, factory reset and a new laptop."
    model = ScriptedModel([AIMessage(content="", tool_calls=[call("prepare_ticket")]), AIMessage(content=invented)])
    run = AgentSession(build_agent_graph(model), SessionContext("a", "facts", "normal", tmp_path),
                       model_name="offline", evidence_kind="deterministic_fake")
    trace = run.turn(f"My issue is {topic}. I tried reconnecting but it failed. Prepare a ticket.")
    fields = json.loads(trace["final_response"].split("\n", 1)[1])
    ticket = json.loads(evidence_path(fields["path"]).read_text())
    assert fields == {key: ticket[key] for key in ("ticket_id", "category", "summary", "steps_tried")} | {"path": fields["path"]}
    assert not any(word in trace["final_response"] for word in ("Campus-Demo", "E999", "factory reset", "laptop"))
    assert trace["model_response"] == invented
    assert run.state["messages"][-1].content == invented
    result = next(m for m in run.state["messages"] if m.type == "tool")
    assert json.loads(result.content)["saved_fields"] == {key: ticket[key] for key in ("category", "summary", "steps_tried")}


def test_request_identity_stable_when_history_is_rederived():
    messages = [HumanMessage(content="Prepare a support ticket."), HumanMessage(content="My issue is Wi-Fi. I tried reconnecting but it failed.")]
    first = derive_helpdesk(messages)
    second = derive_helpdesk([*messages, HumanMessage(content="Please prepare the ticket again.")], first)
    assert first["ticket_request_id"] == second["ticket_request_id"]
    fresh = derive_helpdesk([*messages, HumanMessage(content="Prepare another ticket.")], first)
    assert fresh["ticket_request_id"] != first["ticket_request_id"]
    assert fresh["ticket_id"] is None


@pytest.mark.parametrize("question", ["Should I create a new ticket?", "Can I prepare another ticket?"])
def test_informational_question_is_not_consent_for_an_additional_ticket(question):
    messages = [HumanMessage(content=REQUEST)]
    original = derive_helpdesk(messages)
    followup = derive_helpdesk([*messages, HumanMessage(content=question)], original)
    assert followup["ticket_request_id"] == original["ticket_request_id"]


def test_explicit_new_request_has_its_own_condition(tmp_path):
    messages = [HumanMessage(content=CONDITIONAL)]
    first = derive_helpdesk(messages)
    unconditional = derive_helpdesk([*messages, HumanMessage(content="Prepare another ticket unconditionally.")], first)
    conditional = derive_helpdesk([*messages, HumanMessage(content="If it is operating, prepare another ticket.")], first)
    assert unconditional["ticket_condition"] == "unconditional"
    assert conditional["ticket_condition"] == "operational"
    ctx = SessionContext("a", "new-condition", "normal", tmp_path)
    assert not prepare_ticket(ctx, conditional, {}, tool_call_id="blocked", turn=1)["created"]
    assert prepare_ticket(ctx, unconditional, {}, tool_call_id="allowed", turn=1)["created"]


def test_readiness_uses_the_same_required_fields_without_exposing_runtime_context(tmp_path):
    captured = []
    def inspect(messages):
        prompt = messages[0].content
        marker = "Application validation for this turn: "
        readiness = json.JSONDecoder().raw_decode(prompt.split(marker, 1)[1])[0]
        captured.append(readiness)
        assert str(tmp_path) not in prompt
        assert "private-user" not in prompt and "private-thread" not in prompt
        return AIMessage(content="No fabricated tool result.")
    model = ScriptedModel([inspect, inspect])
    run = AgentSession(build_agent_graph(model), SessionContext("private-user", "private-thread", "normal", tmp_path),
                       model_name="offline", evidence_kind="deterministic_fake")
    run.turn("Prepare a support ticket.")
    run.turn("My issue is Wi-Fi. I tried reconnecting but it failed.")
    assert captured[0]["missing_required_details"]
    assert captured[1]["missing_required_details"] == []
    assert captured[1]["ticket_requested"] is True
    assert "Application validation" not in run.state["messages"][0].content


def test_explicit_unconditional_clarification_preserves_logical_request():
    first = derive_helpdesk([HumanMessage(content=CONDITIONAL)])
    next_context = derive_helpdesk([HumanMessage(content=CONDITIONAL), HumanMessage(content="Prepare an unconditional draft.")], first)
    assert next_context["ticket_request_id"] == first["ticket_request_id"]
    assert next_context["ticket_condition"] == "unconditional"


@pytest.mark.parametrize("description", ["a different Wi-Fi issue", "another problem", "a new issue"])
def test_separate_issue_in_same_category_is_a_new_request(tmp_path, description):
    messages = [HumanMessage(content=REQUEST)]
    first = derive_helpdesk(messages)
    new = derive_helpdesk([*messages, HumanMessage(content=f"I have {description}. I tried restarting but it failed. Prepare a ticket.")], first)
    assert new["ticket_request_id"] != first["ticket_request_id"]
    assert new["steps_tried"] == ["I tried restarting but it failed."]
    ctx = SessionContext("a", "new-issue", "normal", tmp_path)
    old_result = prepare_ticket(ctx, first, {}, tool_call_id="old", turn=1)
    new_result = prepare_ticket(ctx, new, {}, tool_call_id="new", turn=2)
    assert old_result["ticket_id"] != new_result["ticket_id"]


def test_new_issue_cannot_reuse_troubleshooting_steps_from_previous_issue():
    messages = [HumanMessage(content=REQUEST), HumanMessage(content="I have a different Wi-Fi issue. Prepare a ticket.")]
    context = derive_helpdesk(messages)
    assert context["steps_tried"] == [] and not context["steps_known"]


@pytest.mark.parametrize("spelling", ["folder/example.json", r"folder\example.json"])
def test_existing_evidence_paths_in_both_formats(tmp_path, spelling):
    file = tmp_path / "folder/example.json"
    file.parent.mkdir()
    file.write_text('{"actual": true}')
    loaded = evidence_path(spelling, base=tmp_path)
    assert loaded == file.resolve() and json.loads(loaded.read_text())["actual"]


def test_portable_paths_preserve_repository_and_external_file_identity(tmp_path):
    repo_file = ROOT / "outputs/part2/S06.json"
    external = tmp_path / "external.json"
    external.write_text("{}")
    assert portable_path(repo_file) == "outputs/part2/S06.json"
    assert evidence_path(portable_path(repo_file)) == repo_file
    assert portable_path(external) == external.resolve().as_posix()
    assert evidence_path(portable_path(external)) == external.resolve()


def test_foreign_absolute_windows_path_is_not_rebased_as_repository_relative():
    value = r"Z:\external\example.json"
    if os.name == "nt":
        assert evidence_path(value) == Path(value).resolve()
    else:
        with pytest.raises(ValueError, match="explicit local mapping"):
            evidence_path(value)
    assert PureWindowsPath(value).is_absolute()


def test_paths_use_explicit_repository_base_not_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ctx = SessionContext("a", "b", "normal", Path("local-storage/example"))
    assert ctx.storage_dir == ROOT / "local-storage/example"
    assert repository_path("outputs/part2/S06.json") == ROOT / "outputs/part2/S06.json"
    assert evidence_path(r"outputs\part2\S06.json").is_file()


def test_legacy_verifier_loads_saved_examples_from_another_checkout_and_cwd(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    shutil.copytree(ROOT / "outputs/part2", checkout / "outputs/part2")
    monkeypatch.chdir(tmp_path)
    report = verify_evidence(repository_root=checkout, accepted_runs={})
    assert len(report["scenarios"]) == 5
    assert report["totals"]["model_calls"] == 16  # Actual original committed evidence, not new model calls.
    assert all("\\" not in s["log"] for s in report["scenarios"])
    assert all("\\" not in path for s in report["scenarios"] for path in s["ticket_paths"])
