"""DETERMINISTIC FAKE: never contacts OpenAI; used only for tests/evidence."""
import json
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


def call(name, args=None, call_id=None):
    return {"name": name, "args": args or {}, "id": call_id or uuid4().hex, "type": "tool_call"}


class ScriptedModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.received = []

    def bind_tools(self, tools, **kwargs):
        self.tools = tools
        assert kwargs["parallel_tool_calls"] is True
        return self

    def invoke(self, messages):
        self.received.append(list(messages))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response(messages) if callable(response) else response


class DemoFakeModel:
    """A small test policy, not a substitute for measured OpenAI behavior."""
    def bind_tools(self, tools, **kwargs):
        assert kwargs["parallel_tool_calls"] is True
        return self

    def invoke(self, messages):
        if isinstance(messages[-1], HumanMessage):
            text = messages[-1].content.casefold()
            topic = "password" if "password" in text else "wifi"
            if "prepare a support ticket" in text:
                return AIMessage(content="FAKE: What problem are you having, and what steps have you tried?")
            if "issue is" in text and any("support ticket" in m.content.casefold() for m in messages if isinstance(m, HumanMessage)):
                return AIMessage(content="", tool_calls=[call("prepare_ticket")])
            if "status" in text or "down" in text:
                return AIMessage(content="", tool_calls=[call("find_instructions", {"topic": topic}),
                                                           call("check_service_status", {"service": "account_recovery" if topic == "password" else "wifi"})])
            return AIMessage(content="", tool_calls=[call("find_instructions", {"topic": topic})])
        batch = []
        for message in reversed(messages):
            if not isinstance(message, ToolMessage):
                break
            batch.append((message.name, json.loads(message.content)))
        for name, data in batch:
            if name == "prepare_ticket":
                return AIMessage(content="FAKE: " + (f'Local draft {data["ticket_id"]}: {data["path"]}' if data.get("created") else data["reason"]))
        status = next((data for name, data in batch if name == "check_service_status"), None)
        if status:
            requested = any("ticket" in m.content.casefold() for m in messages if isinstance(m, HumanMessage))
            if status["status"] == "operational" and requested:
                return AIMessage(content="", tool_calls=[call("prepare_ticket")])
            return AIMessage(content=f'FAKE: Fictional {status["service"]} status: {status["status"]}. No draft created. No repair time is supplied. Would you like help with the next step?')
        return AIMessage(content="FAKE: " + json.dumps(batch[0][1]))
