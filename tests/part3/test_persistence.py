import json
from pathlib import Path
import subprocess
import sys

from langchain_core.messages import AIMessage
import pytest

from helpdesk.paths import ROOT
from helpdesk.part2.state import SessionContext
from helpdesk.part3.memory_store import MemoryCorruptionError, UserMemoryStore
from helpdesk.part3.persistence import checkpoint_key, inspect_thread, reset_thread
from helpdesk.part3.runner import PersistentSession
from tests.part2.fake_model import ScriptedModel, call
from tests.part3.fake_model import FakeMemoryModel

REPORTED = "I cannot connect to Wi-Fi. I enabled it and forgot the saved network, but reconnecting failed."
TICKET = "My issue is Wi-Fi. I tried reconnecting but it failed. Prepare a ticket."


def probe(directory, message, **identity):
    request = {"storage_dir": str(directory), "message": message, **identity}
    result = subprocess.run([sys.executable, "-m", "tests.part3.process_probe"], input=json.dumps(request),
                            capture_output=True, text=True, encoding="utf-8", cwd=ROOT, timeout=40, check=True)
    return json.loads(result.stdout)


def test_two_turns_use_sqlite_without_duplicated_messages_or_lost_context(tmp_path):
    model = FakeMemoryModel()
    context = SessionContext("a", "same", "normal", tmp_path)
    with PersistentSession(context, model=model, model_name="offline") as run:
        first = run.turn(REPORTED)
        issue = run.state["helpdesk"]["problem_description"]
        old_results = set(run.state["tool_results"])
        second = run.turn("Which problem and steps did we discuss?")
        assert second["memory"]["checkpoint_restored"]
        assert second["memory"]["checkpoint_human_messages_before"] == 1
        assert second["memory"]["new_human_messages_submitted"] == 1
        assert run.state["helpdesk"]["problem_description"] == issue
        assert run.state["helpdesk"]["steps_tried"] == [REPORTED.split(". ", 1)[1]]
        assert old_results <= set(run.state["tool_results"])
        assert second["memory"]["message_counts_after"]["system"] == 1
        assert second["memory"]["message_counts_after"]["human"] == 2
        assert second["control"]["turn"] == 2 and second["control"]["model_calls"] == 1
        assert second["control"]["tool_attempts"] == 0
        assert first["control"]["conversation_id"] == second["control"]["conversation_id"]
        assert "forgot the saved network" in second["final_response"]
        assert "airplane mode" not in second["final_response"]
    assert (tmp_path / "conversations.sqlite").is_file()
    summary = inspect_thread(tmp_path, "a", "same")
    assert summary["exists"] and summary["control"]["turn"] == 2


def test_real_process_restart_restores_messages_context_and_turn(tmp_path):
    first = probe(tmp_path, REPORTED)
    second = probe(tmp_path, "Which steps did I already try?")
    assert first["pid"] != second["pid"]
    assert second["trace"]["memory"]["checkpoint_restored"]
    assert second["humans"] == [REPORTED, "Which steps did I already try?"]
    assert second["trace"]["memory"]["message_counts_after"]["system"] == 1
    assert first["control"]["conversation_id"] == second["control"]["conversation_id"]
    assert second["control"]["turn"] == 2 and second["control"]["model_calls"] == 1
    assert first["tool_ids"] == second["tool_ids"]  # No tool side effect was replayed for the recall question.
    assert "reconnecting failed" in second["trace"]["final_response"]


def test_ticket_identity_survives_real_restart_and_new_request_is_distinct(tmp_path):
    first = probe(tmp_path, TICKET)
    second = probe(tmp_path, "Please show the draft again.")
    third = probe(tmp_path, "Prepare another ticket for this issue.")
    assert len({r["pid"] for r in (first, second, third)}) == 3
    assert first["helpdesk"]["ticket_id"] == second["helpdesk"]["ticket_id"]
    assert first["helpdesk"]["ticket_request_id"] == second["helpdesk"]["ticket_request_id"]
    assert second["tool_ids"][-1] != first["tool_ids"][-1]
    assert len(second["drafts"]) == 1
    assert third["helpdesk"]["ticket_id"] != second["helpdesk"]["ticket_id"]
    assert len(third["drafts"]) == 2


def test_new_conditional_write_after_restart_still_requires_current_status(tmp_path):
    first = probe(tmp_path, TICKET)
    second = probe(tmp_path, "If it is operating, prepare another ticket for this Wi-Fi issue.", status_profile="wifi_outage")
    assert len(first["drafts"]) == len(second["drafts"]) == 1
    assert second["helpdesk"]["ticket_request_id"] != first["helpdesk"]["ticket_request_id"]
    assert second["helpdesk"]["ticket_id"] is None


def test_same_user_new_thread_shares_preference_but_not_conversation(tmp_path):
    first = probe(tmp_path, "Please remember that I prefer replies in Kazakh. I have a Wi-Fi problem.", thread_id="first")
    second = probe(tmp_path, "Which language did I ask you to use? What can you help with?", thread_id="second")
    assert first["pid"] != second["pid"]
    assert second["trace"]["memory"]["checkpoint_restored"] is False
    assert second["trace"]["memory"]["available_fact_sources"]["user_json_memory"] == {"preferred_language": "kk"}
    assert len(second["humans"]) == 1 and second["helpdesk"]["problem_description"] == ""
    assert second["trace"]["final_response"].startswith("FAKE kk")


def test_different_users_with_same_raw_thread_are_isolated(tmp_path):
    probe(tmp_path, "Please remember that I prefer replies in Kazakh. I have a Wi-Fi problem.", user_id="user-a", thread_id="same")
    second = probe(tmp_path, "Have I told you a preferred language or a previous problem?", user_id="user-b", thread_id="same")
    assert checkpoint_key("user-a", "same") != checkpoint_key("user-b", "same")
    assert second["trace"]["memory"]["checkpoint_restored"] is False
    assert second["trace"]["memory"]["user_preferences_after"] == {}
    assert len(second["humans"]) == 1 and second["helpdesk"]["problem_description"] == ""
    assert "No preference or previous problem" in second["trace"]["final_response"]


@pytest.mark.parametrize("value", ["", "../other", "a/b", r"a\b", "C:other", "..", "a" * 129])
def test_path_like_identifiers_rejected_before_storage_access(tmp_path, value):
    for identity in ((value, "thread"), ("user", value)):
        with pytest.raises(ValueError):
            PersistentSession(SessionContext(*identity, "normal", tmp_path), model=FakeMemoryModel())
    assert not list(tmp_path.iterdir())


def test_checkpoint_namespacing_uses_unambiguous_pair_encoding():
    assert checkpoint_key("a", "bc") != checkpoint_key("ab", "c")
    assert checkpoint_key("a", "bc") == checkpoint_key("a", "bc")
    assert len(checkpoint_key("a", "bc")) == 64


def test_reset_thread_changes_ticket_namespace_without_touching_other_users(tmp_path):
    first = probe(tmp_path, TICKET, user_id="a", thread_id="one")
    probe(tmp_path, REPORTED, user_id="b", thread_id="one")
    UserMemoryStore(tmp_path).save("b", "preferred_language", "ru")
    reset_thread(tmp_path, "a", "one")
    assert not inspect_thread(tmp_path, "a", "one")["exists"]
    assert inspect_thread(tmp_path, "b", "one")["exists"]
    assert UserMemoryStore(tmp_path).load("b") == {"preferred_language": "ru"}
    repeated = probe(tmp_path, TICKET, user_id="a", thread_id="one")
    assert repeated["helpdesk"]["ticket_id"] != first["helpdesk"]["ticket_id"]
    assert repeated["control"]["conversation_id"] != first["control"]["conversation_id"]


def test_session_owns_connection_lifetime_and_does_not_reinvoke_on_open(tmp_path):
    ctx = SessionContext("a", "b", "normal", tmp_path)
    with PersistentSession(ctx, model=ScriptedModel([AIMessage(content="saved")]), model_name="offline") as run:
        run.turn("Hello")
    unopened_model = ScriptedModel([])
    with PersistentSession(ctx, model=unopened_model, model_name="offline") as reopened:
        assert reopened.graph.get_state(reopened.config).values["control"]["turn"] == 1
        assert not unopened_model.received
    with pytest.raises(RuntimeError, match="with block"):
        reopened.turn("After close")


def test_corrupted_json_fails_before_new_input_or_model_call(tmp_path):
    ctx = SessionContext("a", "b", "normal", tmp_path)
    store = UserMemoryStore(tmp_path)
    store.path_for("a").parent.mkdir()
    store.path_for("a").write_text("{broken")
    model = ScriptedModel([])
    with PersistentSession(ctx, model=model, model_name="offline") as run:
        with pytest.raises(MemoryCorruptionError):
            run.turn("Hello")
        assert not model.received and not run.graph.get_state(run.config).values


def test_sensitive_user_input_is_redacted_before_checkpointing(tmp_path):
    secret = "My password is TestSecret123"
    with PersistentSession(SessionContext("a", "b", "normal", tmp_path), model=ScriptedModel([AIMessage(content="Do not share secrets.")]), model_name="offline") as run:
        run.turn(secret)
        state = run.graph.get_state(run.config).values
        assert secret not in str(state)
        assert not run.store.path_for("a").exists()


def test_parallel_duplicate_ticket_calls_after_restoration_still_publish_once(tmp_path):
    first = probe(tmp_path, TICKET)
    model = ScriptedModel([AIMessage(content="", tool_calls=[call("prepare_ticket", call_id="parallel-a"), call("prepare_ticket", call_id="parallel-b")]),
                           AIMessage(content="Read confirmed saved fields.")])
    with PersistentSession(SessionContext("demo-user-a", "shared", "normal", tmp_path), model=model, model_name="offline") as run:
        run.turn("Please show the draft again.")
        assert run.state["helpdesk"]["ticket_id"] == first["helpdesk"]["ticket_id"]
        assert run.state["tool_results"]["parallel-a"]["data"] == run.state["tool_results"]["parallel-b"]["data"]
    assert len(list(tmp_path.rglob("DRAFT-*.json"))) == 1


def test_reset_user_preferences_keeps_checkpoint_history(tmp_path):
    probe(tmp_path, "Please remember that I prefer replies in Kazakh. I have a Wi-Fi problem.")
    before = inspect_thread(tmp_path, "demo-user-a", "shared")
    UserMemoryStore(tmp_path).reset("demo-user-a")
    after = inspect_thread(tmp_path, "demo-user-a", "shared")
    assert before == after
    assert UserMemoryStore(tmp_path).load("demo-user-a") == {}
