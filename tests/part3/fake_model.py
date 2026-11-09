"""Explicit deterministic test double. No OpenAI calls or scenario-specific answers."""
import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from helpdesk.part2.ticket_policy import derive_helpdesk
from helpdesk.part3.memory_tool import authorized_preferences
from tests.part2.fake_model import call


class FakeMemoryModel:
    def __init__(self):
        self.received = []

    def bind_tools(self, tools, **kwargs):
        assert kwargs["parallel_tool_calls"] is True
        assert len(tools) == 4
        self.tools = tools
        return self

    def invoke(self, messages):
        self.received.append(list(messages))
        humans = [m for m in messages if isinstance(m, HumanMessage)]
        latest = humans[-1].content
        text = latest.casefold()
        preferred = "preferred_language=kk" in messages[0].content
        prefix = "FAKE kk (Kazakh): " if preferred else "FAKE: "
        if isinstance(messages[-1], HumanMessage):
            changes = authorized_preferences(latest)
            if changes:
                return AIMessage(content="", tool_calls=[call("save_user_memory", {"key": k, "value": v}) for k, v in changes.items()])
            if "ticket" in text or "draft" in text:
                issue = derive_helpdesk(humans)
                if issue["ticket_condition"] == "operational":
                    return AIMessage(content="", tool_calls=[call("check_service_status", {"service": "wifi"})])
                return AIMessage(content="", tool_calls=[call("prepare_ticket")])
            if "which steps" in text or "which problem" in text:
                facts = derive_helpdesk(humans[:-1])
                return AIMessage(content=prefix + json.dumps({"category": facts["category"], "reported_steps": facts["steps_tried"]}))
            if "which language" in text:
                return AIMessage(content=prefix + ("Kazakh preference from user JSON; support scope: Wi-Fi and password recovery." if preferred else "No language preference is recorded."))
            if "have i told" in text:
                facts = derive_helpdesk(humans[:-1])
                return AIMessage(content=prefix + ("Prior problem: " + facts["problem_description"] if facts["problem_description"] else "No preference or previous problem is recorded for this user/thread."))
            topic = "password" if "password" in text else "wifi"
            return AIMessage(content="", tool_calls=[call("find_instructions", {"topic": topic})])
        results = []
        for message in reversed(messages):
            if not isinstance(message, ToolMessage):
                break
            results.append((message.name, json.loads(message.content)))
        checked = next((r for n, r in results if n == "check_service_status"), None)
        if checked and checked["status"] == "operational":
            return AIMessage(content="", tool_calls=[call("prepare_ticket")])
        return AIMessage(content=prefix + json.dumps({name: result for name, result in results}))
