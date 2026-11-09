"""One additional tool: the model proposes, Python authorizes a durable preference."""
import re

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from pydantic import BaseModel, ConfigDict, Field, StrictStr

from ..part2.state import HelpdeskAgentState, SessionContext
from ..part2.ticket_policy import redact
from .memory_store import MemoryValidationError, normalize_preference


def authorized_preferences(text: str) -> dict:
    lower = text.casefold()
    if (redact(text) != text or "[redacted secret]" in lower
            or re.search(r"\b(?:do not|don't|never)\s+(?:save|remember|prefer|reply|respond)\b", lower)
            or re.search(r"\b(?:do|did|have|should|would)\s+i\b", lower)):
        return {}
    if not re.search(r"\b(?:remember|always|prefer|preferred|preference)\b", lower):
        return {}
    preferences = {}
    languages = {"kk": r"\b(?:kazakh|kk|kz)\b|қазақ", "ru": r"\b(?:russian|ru)\b|русск", "en": r"\b(?:english|en)\b|английск"}
    found = [code for code, pattern in languages.items() if re.search(pattern, lower)]
    response_context = re.search(r"\b(?:reply|replies|respond|response|responses|language|speak|answers?|style)\b", lower)
    if len(found) == 1 and response_context:
        preferences["preferred_language"] = found[0]
    styles = [value for value in ("concise", "detailed") if re.search(r"\b" + value + r"\b", lower)]
    if len(styles) == 1 and response_context:
        preferences["response_style"] = styles[0]
    return preferences


class MemoryArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    key: StrictStr = Field(max_length=32)
    value: StrictStr = Field(max_length=32)
    runtime: ToolRuntime[SessionContext, HelpdeskAgentState]


def memory_tool(store):
    @tool("save_user_memory", args_schema=MemoryArguments)
    def save_user_memory(key: str, value: str, runtime: ToolRuntime[SessionContext, HelpdeskAgentState]) -> dict:
        """Save a clearly expressed lasting language/style preference for the current user.

        Keys: preferred_language (en/ru/kk), response_style (concise/detailed).
        Do not save troubleshooting, personal details or secrets. Identity is application-controlled.
        """
        try:
            normalized = normalize_preference(key, value)
        except MemoryValidationError:
            return {"success": False, "reason": "Unsupported preference key or value"}
        latest = next((m.content for m in reversed(runtime.state["messages"]) if isinstance(m, HumanMessage)), "")
        if authorized_preferences(latest).get(key) != normalized:
            return {"success": False, "reason": "The current user message does not authorize this durable preference"}
        return store.save(runtime.context.user_id, key, normalized)

    return save_user_memory
