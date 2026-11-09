import json
import os
import subprocess
import sys

import pytest

from helpdesk.cli import main
from helpdesk.part1.graphs import build_main_graph, build_original_graph, choose_route
from helpdesk.paths import GUIDE_PATH, ROOT
from helpdesk.part1.trace import execute


@pytest.mark.parametrize("builder,version", [(build_original_graph, "original"), (build_main_graph, "main")])
@pytest.mark.parametrize("message,category,node", [
    ("I cannot connect to Wi-Fi.", "wifi", "wifi_help"),
    ("WIFI is unavailable", "wifi", "wifi_help"),
    ("wI-fI", "wifi", "wifi_help"),
    ("I forgot my PASSWORD.", "password", "password_help"),
    ("password", "password", "password_help"),
    ("My laptop screen is broken.", "other", "other_help"),
    ("", "other", "other_help"),
])
def test_real_routes_and_state(builder, version, message, category, node):
    graph = builder()
    result = execute(graph, message, version=version,
                     predicted_route=["read_request", node], metadata={})
    assert result["prediction_matched"]
    assert result["nodes_executed"] == ["read_request", node]
    assert result["steps"][0]["update"] == {"category": category}
    assert result["steps"][0]["state_after"] == {"message": message, "category": category, "reply": ""}
    assert set(result["steps"][1]["update"]) == {"reply"}
    assert result["final_state"]["message"] == message
    assert result["final_state"]["category"] == category
    assert result["final_state"] == graph.invoke(result["initial_state"])
    if category in ("wifi", "password"):
        guide = json.loads(GUIDE_PATH.read_text(encoding="utf-8"))["instructions"][category]
        assert guide["title"] in result["final_response"]
        assert guide["instruction_id"] in result["final_response"]
        assert all(step in result["final_response"] for step in guide["steps"])
    else:
        assert "Wi-Fi or a password" in result["final_response"]


@pytest.mark.parametrize("builder,category,node", [
    (build_original_graph, "wifi", "wifi_help"),
    (build_main_graph, "both", "clarify_problem"),
])
def test_both_topics(builder, category, node):
    graph = builder()
    initial = {"message": "My WI-FI PASSWORD does not work.", "category": "stale", "reply": "old reply"}
    updates = list(graph.stream(initial, stream_mode="updates"))
    assert list(updates[0]) == ["read_request"]
    assert updates[0]["read_request"] == {"category": category}
    assert list(updates[1]) == [node]
    if category == "both":
        reply = updates[1][node]["reply"]
        assert "first" in reply and "Wi-Fi connectivity" in reply and "password recovery" in reply


@pytest.mark.parametrize("builder", [build_original_graph, build_main_graph])
def test_reads_changed_fixture_at_execution(builder, tmp_path):
    path = tmp_path / "guide.json"
    data = json.loads(GUIDE_PATH.read_text(encoding="utf-8"))
    path.write_text(json.dumps(data), encoding="utf-8")
    graph = builder(guide_path=path)
    data["instructions"]["wifi"]["steps"] = ["Unique replacement instruction"]
    path.write_text(json.dumps(data), encoding="utf-8")
    result = graph.invoke({"message": "wifi", "category": "", "reply": ""})
    assert "Unique replacement instruction" in result["reply"]
    assert "Campus-Demo" not in result["reply"]


def test_conditional_route_is_read_only():
    state = {"message": "wifi", "category": "wifi", "reply": ""}
    before = state.copy()
    assert choose_route(state) == "wifi_help"
    assert state == before


def test_default_cli_main_without_keys_or_network():
    # A new interpreter proves that neither provider import nor network is needed.
    code = '''
import importlib.abc, json, socket, sys
class BlockProvider(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('openai', 'langchain_openai'):
            raise AssertionError('Part 1 attempted to import a model provider')
sys.meta_path.insert(0, BlockProvider())
def no_network(*args, **kwargs):
    raise AssertionError('Part 1 attempted a network connection')
socket.socket.connect = no_network
from helpdesk.cli import main
raise SystemExit(main(['--message', 'My Wi-Fi password does not work.']))
'''
    env = {key: value for key, value in os.environ.items()
           if not any(word in key.upper() for word in ("API_KEY", "LANGSMITH", "LANGCHAIN"))}
    run = subprocess.run([sys.executable, "-c", code], env=env, text=True, capture_output=True, check=True)
    result = json.loads(run.stdout)
    assert result["graph_version"] == "main"
    assert result["nodes_executed"] == ["read_request", "clarify_problem"]
    assert result["model_calls"] == []


@pytest.mark.parametrize("scenario,version,node", [
    ("S01", "original", "wifi_help"), ("S02", "original", "password_help"),
    ("S03", "original", "other_help"), ("S04", "original", "wifi_help"),
    ("S05", "main", "clarify_problem"),
])
def test_scenario_cli_and_committed_evidence(scenario, version, node, tmp_path, capsys):
    path = tmp_path / f"{scenario}.json"
    assert main(["--scenario", scenario, "--output", str(path)]) == 0
    actual = json.loads(path.read_text(encoding="utf-8"))
    assert json.loads(capsys.readouterr().out) == actual
    fixtures = json.loads((ROOT / "data/scenarios.json").read_text(encoding="utf-8"))["scenarios"]
    fixture = next(s for s in fixtures if s["id"] == scenario)
    assert actual["message"] == fixture["events"][0]["message"]
    assert actual["graph_version"] == version
    assert actual["nodes_executed"] == ["read_request", node]
    folder = "original" if version == "original" else "improved"
    saved = json.loads((ROOT / "outputs/part1" / folder / f"{scenario}.json").read_text(encoding="utf-8"))
    # Ignore command: the test deliberately writes to a temporary destination.
    assert {k: v for k, v in actual.items() if k != "command"} == {k: v for k, v in saved.items() if k != "command"}


def test_cannot_mislabel_s04_graph():
    with pytest.raises(SystemExit) as error:
        main(["--scenario", "S04", "--graph", "main"])
    assert error.value.code == 2


def test_original_has_no_clarification_node():
    assert "clarify_problem" not in build_original_graph().get_graph().nodes
    assert "clarify_problem" in build_main_graph().get_graph().nodes
