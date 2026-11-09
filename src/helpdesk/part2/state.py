"""Part 2 state channels. Only tool evidence and messages have reducers."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, TypedDict
from uuid import uuid4

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from ..paths import repository_path


class HelpdeskContext(TypedDict, total=False):
    category: str
    problem_description: str
    steps_tried: list[str]
    steps_known: bool
    ticket_requested: bool
    ticket_condition: str
    ticket_id: str | None
    ticket_request_id: str | None
    secret_detected: bool


class ToolEvidence(TypedDict):
    tool: str
    data: dict
    turn: int
    executed: bool


ToolResultsState = dict[str, ToolEvidence]


def merge_results(left: ToolResultsState, right: ToolResultsState) -> ToolResultsState:
    merged = dict(left)
    for key, value in right.items():
        if key in merged and merged[key] != value:
            raise ValueError("Conflicting evidence for the same tool-call identifier")
        merged[key] = value
    return merged


class ExecutionControl(TypedDict, total=False):
    turn: int
    model_calls: int
    tool_calls: int
    tool_attempts: int
    limit_reached: bool
    stop_reason: str | None
    errors: list[str]
    rejected_model_output: dict
    conversation_id: str


class HelpdeskAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    tool_results: Annotated[ToolResultsState, merge_results]
    helpdesk: HelpdeskContext
    control: ExecutionControl


@dataclass(frozen=True)
class SessionContext:
    user_id: str
    thread_id: str
    status_profile: str
    storage_dir: Path
    run_id: str = field(default_factory=lambda: uuid4().hex)

    def __post_init__(self):
        object.__setattr__(self, "storage_dir", repository_path(self.storage_dir))
        if self.status_profile not in ("normal", "wifi_outage", "recovery_outage", "status_unavailable"):
            raise ValueError("Unsupported status profile")
        if not self.user_id.strip() or not self.thread_id.strip():
            raise ValueError("User and thread identifiers must be nonempty")
