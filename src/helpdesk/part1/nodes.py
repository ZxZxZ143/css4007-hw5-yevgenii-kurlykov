import json
from pathlib import Path

from ..paths import GUIDE_PATH
from .state import HelpdeskState


def classify(message: str, *, clarify_both: bool) -> str:
    text = message.casefold()
    wifi = "wifi" in text or "wi-fi" in text
    password = "password" in text
    if clarify_both and wifi and password:
        return "both"
    if wifi:
        return "wifi"
    if password:
        return "password"
    return "other"


def read_request_original(state: HelpdeskState) -> dict:
    """Original / Initial Version: Wi-Fi has priority."""
    return {"category": classify(state["message"], clarify_both=False)}


def read_request_main(state: HelpdeskState) -> dict:
    """Improved / Main Version: detect both topics before Wi-Fi alone."""
    return {"category": classify(state["message"], clarify_both=True)}


def instruction_help(state: HelpdeskState, guide_path: Path = GUIDE_PATH) -> dict:
    """Read the current fixture, rather than embedding or caching its instructions."""
    guide = json.loads(guide_path.read_text(encoding="utf-8"))["instructions"][state["category"]]
    steps = "\n".join(f"{i}. {step}" for i, step in enumerate(guide["steps"], 1))
    return {"reply": f'{guide["title"]} ({guide["instruction_id"]})\n{steps}'}


def other_help(state: HelpdeskState) -> dict:
    return {"reply": "Does your problem concern Wi-Fi or a password?"}


def clarify_problem(state: HelpdeskState) -> dict:
    return {"reply": "Which problem would you like to solve first: Wi-Fi connectivity or password recovery?"}
