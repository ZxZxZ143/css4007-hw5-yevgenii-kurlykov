"""The existing single-agent graph with a connection-lifetime-safe SQLite session."""
from contextlib import ExitStack
from dataclasses import replace
import json
import os
from pathlib import Path
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from ..part2.agent import build_agent_graph, initial_agent_state
from ..part2.runner import AgentSession, serializable
from ..part2.state import SessionContext
from ..part2.ticket_policy import redact
from .locking import file_lock
from .memory_store import UserMemoryStore
from .memory_tool import memory_tool
from .persistence import checkpoint_config, checkpoint_key, database_path

MEMORY_INSTRUCTIONS = """Persistent mode has one additional tool, save_user_memory.
Use it to save only clearly stated lasting language/style preferences, especially when
the user asks you to remember. Language values: en, ru, kk; style: concise or detailed.
Never save incidental troubleshooting, identities, contact information or secrets.
Saved preferences below come from user_json_memory, not the current conversation.
Only messages in this thread describe its past problems/steps. Recall user-reported
steps exactly; never turn guide suggestions into steps the user already tried.
For a new user/thread with no history or preference, say so rather than inventing one.
Language codes: kk means Kazakh, ru means Russian, en means English.
Follow the saved response language in the final answer, including after saving it now.
When asked which problem/steps were discussed, answer directly from the restored history
instead of asking the user to repeat it or re-fetching instructions."""


class PersistentSession(AgentSession):
    def __init__(self, context: SessionContext, *, model=None, model_name=None,
                 evidence_kind="live_openai", max_model_calls=6, max_tool_calls=5):
        self.base_context = context
        self.config = checkpoint_config(context.user_id, context.thread_id)
        self.store = UserMemoryStore(context.storage_dir)
        self.model = model
        self.model_name = model_name
        self.evidence_kind = evidence_kind
        self.budgets = {"max_model_calls": max_model_calls, "max_tool_calls": max_tool_calls}
        self.context_reads = []
        self._stack = None

    def __enter__(self):
        directory = self.base_context.storage_dir
        directory.mkdir(parents=True, exist_ok=True)
        with ExitStack() as stack:
            saver = stack.enter_context(SqliteSaver.from_conn_string(str(database_path(directory))))
            graph = build_agent_graph(self.model, checkpointer=saver, extra_tools=[memory_tool(self.store)],
                                      model_context=self._model_memory_context, **self.budgets)
            super().__init__(graph, self.base_context, model_name=self.model_name or os.getenv("MODEL_NAME", "unconfigured"),
                             evidence_kind=self.evidence_kind)
            self.log.update(persistent=True, checkpoint_key=self.config["configurable"]["thread_id"], process_id=os.getpid())
            restored = self.graph.get_state(self.config).values
            if restored:
                self.state = restored
            self._stack = stack.pop_all()
        return self

    def __exit__(self, *args):
        self._stack.close()
        self._stack = None

    def _model_memory_context(self, state, context):
        preferences = self.store.load(context.user_id)
        self.context_reads.append({"source": "user_json_memory", "keys_loaded": sorted(preferences),
                                   "preferences": preferences})
        instruction = "; ".join(f"{key}={value}" for key, value in sorted(preferences.items())) or "none recorded"
        return MEMORY_INSTRUCTIONS + "\nSaved user preferences: " + instruction

    def turn(self, message: str) -> dict:
        if self._stack is None:
            raise RuntimeError("Use PersistentSession inside a with block")
        key = checkpoint_key(self.base_context.user_id, self.base_context.thread_id)
        with file_lock(self.base_context.storage_dir / "locks" / f"{key}.lock"):
            snapshot = self.graph.get_state(self.config)
            before = snapshot.values
            if snapshot.next:
                raise ValueError("This thread has an unfinished invocation; inspect it before adding another user turn")
            preferences_before = self.store.load(self.base_context.user_id)
            self.context_reads = []
            conversation_id = before.get("control", {}).get("conversation_id") or uuid4().hex
            self.context = replace(self.base_context, run_id=conversation_id)
            control = {"turn": before.get("control", {}).get("turn", 0) + 1,
                       "conversation_id": conversation_id, "model_calls": 0, "tool_calls": 0,
                       "tool_attempts": 0, "limit_reached": False, "stop_reason": None, "errors": []}
            user = HumanMessage(content=redact(message), id=uuid4().hex)
            if before:
                updates = {"messages": [user], "control": control}
            else:
                updates = initial_agent_state()
                updates["messages"] = [*updates["messages"], user]
                updates["control"] = control
            trace = self._execute(updates, message, config=self.config)
            trace["memory"] = {
                "checkpoint_key": key, "checkpoint_restored": bool(before),
                "checkpoint_messages_before": len(before.get("messages", [])),
                "checkpoint_human_messages_before": sum(m.type == "human" for m in before.get("messages", [])),
                "new_human_messages_submitted": 1,
                "available_fact_sources": {
                    "conversation_checkpoint": serializable(before.get("helpdesk", {})) if before else {},
                    "user_json_memory": preferences_before,
                    "helpdesk_reference": [entry["data"].get("instruction_id") for entry in self.state["tool_results"].values()
                                           if entry["turn"] == control["turn"] and entry["tool"] == "find_instructions" and entry["executed"]],
                },
                "model_context_reads": list(self.context_reads),
                "user_preferences_after": self.store.load(self.context.user_id),
                "message_counts_after": {kind: sum(m.type == kind for m in self.state["messages"])
                                         for kind in ("system", "human", "ai", "tool")},
            }
            return trace
