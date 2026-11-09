import json

import pytest

from helpdesk.cli import main
from helpdesk.part2.tools import read_fixture
from helpdesk.part3.scenarios import PART3_SCENARIOS, run_memory_scenario


@pytest.mark.parametrize("scenario", PART3_SCENARIOS)
def test_actual_fixtures_with_real_worker_processes(tmp_path, scenario):
    output = tmp_path / f"{scenario}.json"
    log = run_memory_scenario(scenario, storage_dir=tmp_path, output=output, model_mode="fake")
    fixture = next(s for s in read_fixture("scenarios.json")["scenarios"] if s["id"] == scenario)
    assert [e["user_message"] for e in log["events"]] == [e["message"] for e in fixture["events"] if e["action"] == "message"]
    assert log["evidence_kind"] == "deterministic_fake"
    assert all(e["control"]["stop_reason"] == "final_response" for e in log["events"])
    assert all(e["memory"]["message_counts_after"]["system"] == 1 for e in log["events"])
    assert json.loads(output.read_text(encoding="utf-8")) == log
    if scenario in ("S11", "S12"):
        assert log["restart_occurred"]
        assert len({p["pid"] for p in log["processes"]}) == 2
        assert all(p["return_code"] == 0 and p["exited_before_next_process"] for p in log["processes"])
        assert log["events"][1]["memory"]["user_preferences_after"] == {"preferred_language": "kk"}
    if scenario in ("S10", "S11"):
        assert log["events"][1]["memory"]["checkpoint_restored"]
        assert "tried" in log["events"][1]["final_response"] or "enabled" in log["events"][1]["final_response"]
    if scenario == "S12":
        assert not log["events"][1]["memory"]["checkpoint_restored"]
        assert log["events"][1]["memory"]["available_fact_sources"]["conversation_checkpoint"] == {}
    if scenario == "S13":
        assert log["events"][1]["memory"]["user_preferences_after"] == {}
        assert not log["events"][1]["memory"]["checkpoint_restored"]


def test_reset_before_replay_does_not_accumulate_user_messages(tmp_path):
    for _ in range(2):
        log = run_memory_scenario("S11", storage_dir=tmp_path, output=tmp_path / "S11.json", model_mode="fake")
        assert log["events"][0]["memory"]["checkpoint_restored"] is False
        assert log["events"][1]["memory"]["message_counts_after"]["human"] == 2


def test_scenario_reset_does_not_touch_manual_session_or_preferences(tmp_path):
    from helpdesk.part3.memory_store import UserMemoryStore
    from tests.part3.test_persistence import probe
    from helpdesk.part3.persistence import inspect_thread
    probe(tmp_path, "Please remember that I prefer replies in Kazakh. I have a Wi-Fi problem.")
    before = inspect_thread(tmp_path, "demo-user-a", "shared")
    run_memory_scenario("S13", storage_dir=tmp_path, output=tmp_path / "S13.json", model_mode="fake")
    assert inspect_thread(tmp_path, "demo-user-a", "shared") == before
    assert UserMemoryStore(tmp_path).load("demo-user-a") == {"preferred_language": "kk"}


def test_persistent_cli_and_inspection_reset_do_not_change_part2_mode(tmp_path, monkeypatch, capsys):
    from tests.part3.fake_model import FakeMemoryModel
    import helpdesk.part2.agent as module
    monkeypatch.setattr(module, "openai_model", FakeMemoryModel)
    common = ["--graph", "agent", "--persistent", "--user-id", "a", "--thread-id", "b", "--storage-dir", str(tmp_path)]
    assert main([*common, "--message", "I have a Wi-Fi problem."]) == 0
    capsys.readouterr()
    assert main([*common, "--inspect-thread"]) == 0
    assert json.loads(capsys.readouterr().out)["exists"]
    assert main([*common, "--reset-thread"]) == 0
    capsys.readouterr()
    assert main([*common, "--inspect-thread"]) == 0
    assert not json.loads(capsys.readouterr().out)["exists"]
    assert main([*common, "--reset-user-memory"]) == 0


@pytest.mark.parametrize("args", [["--persistent", "--message", "Hi"], ["--inspect-thread"],
                                   ["--scenario", "S11", "--graph", "main"],
                                   ["--graph", "agent", "--persistent", "--scenario", "S06"]])
def test_incompatible_cli_combinations_rejected(args):
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2


@pytest.mark.parametrize("scenario", PART3_SCENARIOS)
def test_documented_cli_scenarios_select_persistent_runner(tmp_path, monkeypatch, capsys, scenario):
    import helpdesk.part3.cli as cli
    actual_runner = cli.run_memory_scenario
    def offline_run(identifier, **kwargs):
        return actual_runner(identifier, model_mode="fake", **kwargs)
    monkeypatch.setattr(cli, "run_memory_scenario", offline_run)
    assert main(["--scenario", scenario, "--storage-dir", str(tmp_path), "--output", str(tmp_path / f"{scenario}.json")]) == 0
    log = json.loads(capsys.readouterr().out)
    assert log["scenario_id"] == scenario and log["evidence_kind"] == "deterministic_fake"
    assert all(e["memory"]["message_counts_after"]["system"] == 1 for e in log["events"])


def test_documented_interactive_persistent_cli(tmp_path, monkeypatch, capsys):
    import helpdesk.part2.agent as module
    from tests.part3.fake_model import FakeMemoryModel
    monkeypatch.setattr(module, "openai_model", FakeMemoryModel)
    messages = iter(["I cannot connect to Wi-Fi. I tried reconnecting but it failed.", "Which steps did I already try?", "/quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(messages))
    output = tmp_path / "session.json"
    assert main(["--graph", "agent", "--persistent", "--interactive", "--user-id", "a", "--thread-id", "b",
                 "--storage-dir", str(tmp_path), "--output", str(output)]) == 0
    log = json.loads(output.read_text(encoding="utf-8"))
    assert len(log["turns"]) == 2 and log["turns"][1]["memory"]["checkpoint_restored"]
