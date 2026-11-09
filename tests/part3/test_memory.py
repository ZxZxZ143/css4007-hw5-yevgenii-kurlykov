from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

from langchain_core.messages import AIMessage
import pytest

from helpdesk.paths import ROOT
from helpdesk.part2.state import SessionContext
from helpdesk.part3.memory_store import MemoryCorruptionError, MemoryValidationError, UserMemoryStore
from helpdesk.part3.memory_tool import authorized_preferences, memory_tool
from helpdesk.part3.runner import PersistentSession
from tests.part2.fake_model import ScriptedModel, call


def test_missing_user_and_normalized_idempotent_save(tmp_path):
    store = UserMemoryStore(tmp_path)
    assert store.load("demo-user-a") == {}
    assert not store.path_for("demo-user-a").exists()
    first = store.save("demo-user-a", "preferred_language", "Kazakh")
    path = store.path_for("demo-user-a")
    before = path.stat().st_mtime_ns
    second = store.save("demo-user-a", "preferred_language", "kz")
    assert first["value"] == "kk" and first["changed"] is True
    assert second["changed"] is False and path.stat().st_mtime_ns == before
    assert json.loads(path.read_text()) == {"schema_version": 1, "preferences": {"preferred_language": "kk"}}
    assert store.load("demo-user-b") == {}
    assert "demo-user" not in path.name


@pytest.mark.parametrize("key,value", [
    ("password", "anything"), ("api_key", "sk-test123456789"), ("recovery_code", "123456"),
    ("problem", "Wi-Fi is broken"), ("preferred_language", "de"), ("preferred_language", "x" * 100),
    ("preferred_language", {"value": "kk"}), ("response_style", "unknown"), ({"key": "preferred_language"}, "kk"),
])
def test_invalid_preference_never_written(tmp_path, key, value):
    store = UserMemoryStore(tmp_path)
    with pytest.raises(MemoryValidationError):
        store.save("a", key, value)
    assert not list(tmp_path.rglob("*.json"))


@pytest.mark.parametrize("document", [
    "{truncated", '{"schema_version":true,"preferences":{}}',
    '{"schema_version":1,"preferences":{"password":"secret"}}',
    '{"schema_version":1,"preferences":{"preferred_language":"Kazakh"}}',
    '{"schema_version":1,"preferences":{},"extra":"data"}',
])
def test_corrupted_document_is_reported_without_overwrite(tmp_path, document):
    store = UserMemoryStore(tmp_path)
    path = store.path_for("a")
    path.parent.mkdir()
    path.write_text(document)
    for operation in (lambda: store.load("a"), lambda: store.save("a", "preferred_language", "kk")):
        with pytest.raises(MemoryCorruptionError):
            operation()
    assert path.read_text() == document
    store.reset("a")
    assert store.load("a") == {}


def test_atomic_update_failure_preserves_old_memory(tmp_path, monkeypatch):
    import helpdesk.part3.memory_store as module
    store = UserMemoryStore(tmp_path)
    store.save("a", "preferred_language", "kk")
    before = store.path_for("a").read_bytes()
    def fail(*args):
        raise OSError("Simulated atomic publication failure")
    with monkeypatch.context() as patch:
        patch.setattr(module.os, "replace", fail)
        with pytest.raises(OSError):
            store.save("a", "response_style", "concise")
    assert store.path_for("a").read_bytes() == before
    assert not list(tmp_path.rglob("*.tmp"))
    assert store.save("a", "response_style", "concise")["success"]


def test_parallel_thread_updates_preserve_both_preferences(tmp_path):
    store = UserMemoryStore(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda item: store.save("a", *item), [("preferred_language", "kk"), ("response_style", "concise")]))
    assert all(result["success"] for result in results)
    assert store.load("a") == {"preferred_language": "kk", "response_style": "concise"}


def test_parallel_process_updates_preserve_both_preferences(tmp_path):
    processes = []
    for key, value in [("preferred_language", "kk"), ("response_style", "concise")]:
        process = subprocess.Popen([sys.executable, "-m", "tests.part3.process_probe"], cwd=ROOT,
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
        process.stdin.write(json.dumps({"operation": "save_preference", "storage_dir": str(tmp_path),
                                       "user_id": "a", "key": key, "value": value}))
        process.stdin.close()
        processes.append(process)
    for process in processes:
        process.wait(timeout=30)
        assert process.returncode == 0
        assert json.loads(process.stdout.read())["success"]
    assert UserMemoryStore(tmp_path).load("a") == {"preferred_language": "kk", "response_style": "concise"}


def test_parallel_memory_and_read_tools_refresh_next_model_context(tmp_path):
    def final(messages):
        assert "preferred_language=kk" in messages[0].content and "response_style=concise" in messages[0].content
        assert len([m for m in messages if m.type == "tool"]) == 4
        return AIMessage(content="Offline confirmation.")
    model = ScriptedModel([
        AIMessage(content="", tool_calls=[call("save_user_memory", {"key": "preferred_language", "value": "kk"}, "language"),
                                          call("save_user_memory", {"key": "response_style", "value": "concise"}, "style"),
                                          call("find_instructions", {"topic": "wifi"}, "guide"),
                                          call("check_service_status", {"service": "wifi"}, "status")]), final])
    with PersistentSession(SessionContext("a", "b", "normal", tmp_path), model=model, model_name="offline") as run:
        trace = run.turn("Please remember that I prefer replies in Kazakh and prefer concise replies. Check Wi-Fi.")
        assert set(run.state["tool_results"]) == {"language", "style", "guide", "status"}
        assert trace["control"]["tool_calls"] == 4
        assert trace["memory"]["model_context_reads"][0]["preferences"] == {}
        assert trace["memory"]["model_context_reads"][1]["preferences"] == {"preferred_language": "kk", "response_style": "concise"}
        assert len([m for m in run.state["messages"] if m.type == "system"]) == 1
        assert "Saved user preferences" not in run.state["messages"][0].content
        assert messages_are_paired(run.state["messages"])


def messages_are_paired(messages):
    pending = set()
    for message in messages:
        if message.type == "ai":
            assert not pending
            pending = {call["id"] for call in message.tool_calls}
        elif message.type == "tool":
            assert message.tool_call_id in pending
            pending.remove(message.tool_call_id)
    return not pending


@pytest.mark.parametrize("arguments", [
    {"key": "preferred_language", "value": "kk", "user_id": "other-user"},
    {"key": "preferred_language", "value": "kk", "path": "../other.json"},
    {"key": "preferred_language", "value": "kk", "operation": "delete"},
    {"key": "problem", "value": "wifi"},
    {"key": "preferred_language", "value": "sk-secret123456789"},
    {"key": "preferred_language", "value": {"nested": "kk"}},
])
def test_model_cannot_choose_identity_path_operation_or_unsupported_memory(tmp_path, arguments):
    model = ScriptedModel([AIMessage(content="", tool_calls=[call("save_user_memory", arguments)]), AIMessage(content="Rejected.")])
    with PersistentSession(SessionContext("a", "b", "normal", tmp_path), model=model, model_name="offline") as run:
        trace = run.turn("Please remember that I prefer replies in Kazakh.")
        assert trace["memory"]["user_preferences_after"] == {}
        assert not run.store.path_for("a").exists()
        assert not run.store.path_for("other-user").exists()
        assert messages_are_paired(run.state["messages"])


@pytest.mark.parametrize("message", ["My Wi-Fi is broken today.", "Do not remember my language preference.", "Which language did I request?",
                                     "I prefer Kazakh food.", "Do I prefer replies in Kazakh?", "I do not prefer replies in Kazakh."])
def test_model_proposal_requires_a_current_durable_preference(tmp_path, message):
    model = ScriptedModel([AIMessage(content="", tool_calls=[call("save_user_memory", {"key": "preferred_language", "value": "kk"})]), AIMessage(content="No save.")])
    with PersistentSession(SessionContext("a", "b", "normal", tmp_path), model=model, model_name="offline") as run:
        trace = run.turn(message)
        assert trace["memory"]["user_preferences_after"] == {}


def test_schema_has_only_preference_fields(tmp_path):
    schema = memory_tool(UserMemoryStore(tmp_path)).tool_call_schema.model_json_schema()
    assert set(schema["properties"]) == {"key", "value"}
    assert authorized_preferences("Always reply to me in Kazakh.") == {"preferred_language": "kk"}
    assert authorized_preferences("My Wi-Fi problem is not a language preference.") == {}


def test_preference_overwrite_applies_in_the_next_model_invocation(tmp_path):
    seen = []
    def reply(messages):
        seen.append(messages[0].content)
        assert messages[0].content.count("Saved user preferences:") == 1
        return AIMessage(content="Preference updated.")
    model = ScriptedModel([
        AIMessage(content="", tool_calls=[call("save_user_memory", {"key": "preferred_language", "value": "kk"})]), reply,
        AIMessage(content="", tool_calls=[call("save_user_memory", {"key": "preferred_language", "value": "ru"})]), reply,
    ])
    with PersistentSession(SessionContext("a", "b", "normal", tmp_path), model=model, model_name="offline") as run:
        run.turn("Always reply to me in Kazakh.")
        run.turn("Please remember that I prefer replies in Russian.")
        assert "preferred_language=kk" in seen[0]
        assert "preferred_language=ru" in seen[1] and "preferred_language=kk" not in seen[1]
        assert run.store.load("a") == {"preferred_language": "ru"}
        assert run.state["messages"][0].content.count("Saved user preferences:") == 0


def test_secret_in_preference_request_is_not_written_to_json_or_checkpoint(tmp_path):
    secret = "My password is TestSecret123"
    model = ScriptedModel([AIMessage(content="", tool_calls=[call("save_user_memory", {"key": "preferred_language", "value": "kk"})]), AIMessage(content="Remove secrets.")])
    with PersistentSession(SessionContext("a", "b", "normal", tmp_path), model=model, model_name="offline") as run:
        run.turn("Please remember that I prefer replies in Kazakh. " + secret)
        assert not run.store.path_for("a").exists()
        assert secret not in str(run.graph.get_state(run.config).values)


def test_incidental_detail_word_is_not_a_response_style_preference(tmp_path):
    model = ScriptedModel([AIMessage(content="", tool_calls=[call("save_user_memory", {"key": "response_style", "value": "detailed"})]), AIMessage(content="No style preference saved.")])
    with PersistentSession(SessionContext("a", "b", "normal", tmp_path), model=model, model_name="offline") as run:
        run.turn("I prefer detailed maps.")
        assert not run.store.path_for("a").exists()


def test_clearly_stated_preferred_language_is_authorized():
    assert authorized_preferences("My preferred language is Kazakh.") == {"preferred_language": "kk"}
