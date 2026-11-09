"""One model, one explicit LangGraph loop, standard parallel ToolNode execution."""
import json
import os

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from langgraph.types import Command

from .agent_state import HelpdeskAgentState, SessionContext
from .nodes import ROOT
from .ticket_policy import derive_helpdesk
from .tools import agent_tools

SYSTEM_PROMPT = """You are a fictional university IT helpdesk. Only local demo data is available.
Help with Wi-Fi and password recovery. Retrieve instructions using find_instructions;
check wifi or account_recovery using check_service_status. Never invent status, repair
times, steps tried, or guide content. Independent guide/status reads may run in parallel.
prepare_ticket uses the user's reported details and saves a LOCAL DRAFT, never submits it.
Ask for the problem and steps tried (or none) when missing. Honor all user conditions:
if a ticket is conditional on an operational service, first complete the status check,
inspect its result, then request prepare_ticket in a later batch. Outage, unknown, and
not checked do not satisfy that condition. Read rejection results and ask for clarification.
After clarification in a later user turn, complete the previously requested draft if safe.
The draft requires only the problem category, a meaningful description and reported
steps tried (or an explicit statement that none were tried). Do not require contact
details, device/OS, location, time or urgency: those are not fields in this draft template.
Once the user supplies the missing issue and steps, call prepare_ticket in that turn;
the original request is still authorization, so do not ask for confirmation again.
Operational status is required ONLY for a conditional request. An explicitly unconditional
request may produce a draft during an outage or with status not_checked. Never describe
the operational-only restriction as applying to all tickets. Keep final responses concise.
Do not request or repeat passwords, API keys or recovery codes. Ground final replies in
tool results, including the actual ticket ID and path only when a draft was created.
State explicitly that statuses describe fictional fixtures, not real university services."""


class ConfigurationError(ValueError):
    pass


def openai_model():
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI

    load_dotenv(ROOT / ".env", override=False)
    key, name = os.getenv("OPENAI_API_KEY"), os.getenv("MODEL_NAME")
    if not key or not name:
        raise ConfigurationError("Set OPENAI_API_KEY and a tool-capable MODEL_NAME in the environment or ignored .env. Live verification is pending.")
    return ChatOpenAI(model=name, api_key=key, timeout=45, max_retries=0, use_responses_api=True)


def record_tool_result(request, execute):
    """ToolNode calls this per concurrent call, against the same pre-batch state."""
    response = execute(request)
    try:
        data = json.loads(response.content)
    except (ValueError, TypeError):
        data = {"error": "Tool input or execution failed."}
        response = ToolMessage(content=json.dumps(data), name=request.tool_call["name"],
                               tool_call_id=request.tool_call["id"], status="error")
    # Full guide content stays in ToolMessage; the parallel reducer stores compact evidence.
    compact = {key: data[key] for key in (
        "topic", "instruction_id", "service", "status", "status_profile", "source", "fictional",
        "created", "ticket_id", "path", "reason", "error", "service_status",
    ) if key in data}
    entry = {"tool": request.tool_call["name"], "data": compact,
             "turn": request.state["control"]["turn"], "executed": response.status != "error"}
    return Command(update={"messages": [response], "tool_results": {request.tool_call["id"]: entry}})


def build_agent_graph(model=None, *, max_model_calls: int = 6, max_tool_calls: int = 5):
    if max_model_calls < 1 or max_tool_calls < 1:
        raise ValueError("Invocation budgets must be positive")
    tools = agent_tools()
    bound = (model if model is not None else openai_model()).bind_tools(tools, parallel_tool_calls=True)

    def agent(state: HelpdeskAgentState, runtime: Runtime[SessionContext]):
        control = dict(state["control"])
        current_results = [e for e in state["tool_results"].values() if e["turn"] == control["turn"]]
        control["tool_calls"] = sum(e["executed"] for e in current_results)
        control["tool_attempts"] = len(current_results)
        helpdesk = derive_helpdesk(state["messages"], state["helpdesk"])
        for result in current_results:
            if result["data"].get("created"):
                helpdesk["ticket_id"] = result["data"]["ticket_id"]
        if control["model_calls"] >= max_model_calls or control["tool_attempts"] >= max_tool_calls:
            control.update(limit_reached=True, stop_reason="invocation_budget")
            created = [e["data"] for e in current_results if e["data"].get("created")]
            suffix = ""
            if created:
                suffix = f' Local draft: {created[-1]["ticket_id"]}, {created[-1]["path"]}.'
            return {"messages": [AIMessage(content="Execution limit reached; no further operations were attempted." + suffix)],
                    "helpdesk": helpdesk, "control": control}
        control["model_calls"] += 1
        try:
            response = bound.invoke(state["messages"])
            if not isinstance(response, AIMessage):
                raise TypeError("Expected an AIMessage")
        except Exception as error:
            control.update(stop_reason="model_error", errors=[*control.get("errors", []), type(error).__name__])
            return {"messages": [AIMessage(content="The model request failed. No further operation was attempted; retry after checking API configuration.")],
                    "helpdesk": helpdesk, "control": control}
        calls = response.tool_calls
        invalid = response.invalid_tool_calls
        ids = [call["id"] for call in calls + invalid]
        reason = None
        if invalid or len(ids) != len(set(ids)) or any(not key or key in state["tool_results"] for key in ids):
            reason = "invalid_tool_calls"
        elif len(calls) > max_tool_calls - control["tool_attempts"]:
            reason = "tool_batch_budget"
            control["limit_reached"] = True
        if reason:
            control["stop_reason"] = reason
            if reason == "invalid_tool_calls":
                # Malformed IDs cannot form a valid provider protocol. Keep the actual
                # rejected output in diagnostics, outside the next model's history.
                control["rejected_model_output"] = response.model_dump(mode="json")
                return {"messages": [AIMessage(content="The model returned malformed tool requests. No tools executed; please retry.")],
                        "control": control, "helpdesk": helpdesk}
            # Close every requested call, including rejected batches, before a final answer.
            replies = [ToolMessage(content=json.dumps({"error": reason, "executed": False}),
                                   tool_call_id=call["id"] or "invalid", name=call["name"], status="error")
                       for call in calls + invalid]
            return {"messages": [response, *replies, AIMessage(content="Tool requests were rejected (" + reason + "). No tools in this batch executed.")],
                    "control": control, "helpdesk": helpdesk}
        if not calls:
            control["stop_reason"] = "final_response"
        return {"messages": [response], "control": control, "helpdesk": helpdesk}

    def route(state: HelpdeskAgentState):
        return "tools" if state["messages"][-1].tool_calls else END

    graph = StateGraph(HelpdeskAgentState, context_schema=SessionContext)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools, wrap_tool_call=record_tool_result,
                                     handle_tool_errors="Tool input or execution failed; correct the request or ask for clarification."))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile().with_config(recursion_limit=2 * max_model_calls + 4)


def initial_agent_state() -> HelpdeskAgentState:
    return {"messages": [SystemMessage(content=SYSTEM_PROMPT)], "tool_results": {}, "helpdesk": {},
            "control": {"turn": 0, "model_calls": 0, "tool_calls": 0, "tool_attempts": 0, "limit_reached": False,
                        "stop_reason": None, "errors": []}}
