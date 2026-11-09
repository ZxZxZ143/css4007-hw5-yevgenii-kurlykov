# HW5 — Build a university IT helpdesk with LangGraph

**CSS-4007 · Artificial Intelligence**  
**Due:** 8 November 2026, 23:59 (UTC+5)

## Implementation status

Tested with **CPython 3.13.12 on Windows**, LangGraph **1.2.20**, SQLite checkpoint adapter **3.1.3**, and pytest **9.2.0**. Parts 1–3 are verified, including live OpenAI Part 2 and Part 3 scenarios. The model was `gpt-5-mini`, reported by the API as `gpt-5-mini-2025-08-07`. **200 tests pass** without requiring an API key. Part 4 and GitHub Actions remain future work. The original assignment remains below.

From the repository root, create a fresh environment and install the pinned dependencies and editable local package:

```powershell
python -m venv .venv-part1
.\.venv-part1\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
```

On macOS/Linux, activate with `source .venv-part1/bin/activate`. Use Python 3.13 for the tested setup. Run the following commands from the repository root after activation. Part 1 never loads `.env`, calls a model, or requires an API key. The pinned provider and dotenv dependencies support Parts 2–3; the already pinned SQLite adapter supplies Part 3 checkpoints. No new dependency was required.

```powershell
# Original / Initial Version: Wi-Fi wins when both keywords appear.
python -m helpdesk.cli --graph original --message "My Wi-Fi password does not work."

# Improved / Main Version: both topics lead to the new clarification node.
python -m helpdesk.cli --graph main --message "My Wi-Fi password does not work."

# Without --graph, an ordinary message uses Improved / Main.
python -m helpdesk.cli --message "My Wi-Fi password does not work."

# Replay the supplied Part 1 scenarios unchanged and save one log per scenario.
python -m helpdesk.cli --scenario S01
python -m helpdesk.cli --scenario S02
python -m helpdesk.cli --scenario S03
python -m helpdesk.cli --scenario S04
python -m helpdesk.cli --scenario S05

python -m pytest -q
```

Scenario replay binds S01–S04 to Original and S05 to Improved / Main, as required by the fixtures. A contradictory explicit `--graph` is rejected. The fixture name `with_clarification` maps to CLI version `main`. S06–S09/S14 use the nonpersistent Part 2 agent; S10–S13 automatically select persistent Part 3 mode.

Each invocation prints JSON. Scenario commands also save traces in `outputs/part1/original/S01.json` through `S04.json` and `outputs/part1/improved/S05.json`. For a custom message, add `--output outputs/part1/custom.json` to save its trace. These files contain the command, fixture context, prediction, executed nodes, partial updates, complete states and final reply. `START` and `END` are boundary sentinels, so the executed-node list contains only real Python nodes. Traces use LangGraph's `updates` and `values` streams; they are not reconstructed by calling the node functions separately.

The predictions were recorded before implementation/execution in [PREDICTIONS.md](outputs/part1/PREDICTIONS.md). The explanation and measured results are in [SUBMISSION.md](SUBMISSION.md). Instructions are read from `data/helpdesk.json` at execution time. No supplied fixture was edited. Part 1 has no persistent memory, so every invocation starts fresh.

Code: `src/helpdesk/part1/{state,nodes,graphs,trace,cli}.py`. Import `build_original_graph` or `build_main_graph` from `helpdesk.part1.graphs` to compile either graph independently. Both use shared help nodes; only Main includes `clarify_problem` and the two-topic classification. The tests execute real graphs, compare saved evidence with fresh runs, verify fixture retrieval and preserved state, and run the default CLI with provider imports and network connections blocked. The Part 1 tests use no fake model.

## Part 2 — Single OpenAI agent

Use the same installed environment. Copy `.env.example` to ignored `.env` and set `OPENAI_API_KEY` and `MODEL_NAME` to a tool-capable OpenAI model available to your account. Environment variables take precedence over `.env`. No model identifier or account availability is assumed. `ChatOpenAI` uses the Responses API with a 45-second timeout and no automatic retries. See [OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling).

```powershell
python -m helpdesk.cli --graph agent --message "Show me Wi-Fi troubleshooting steps."
python -m helpdesk.cli --graph agent --interactive --user-id demo-user-a --thread-id demo-thread --status-profile normal --output outputs/part2/session.json

python -m helpdesk.cli --scenario S06
python -m helpdesk.cli --scenario S07
python -m helpdesk.cli --scenario S08
python -m helpdesk.cli --scenario S09
python -m helpdesk.cli --scenario S14

python -m pytest -q
# Explicitly offline evidence; this does NOT contact OpenAI.
python -m scripts.part2.record_offline
```

Enter `/quit` or EOF to end interactive mode. The same process retains the current conversation, including S06/S09 follow-up details. Scenario replay reads exact fixture events and snapshot names, starts isolated state for each scenario, and writes live execution logs to `outputs/part2/Sxx.json`. The scenario fixture controls identity and snapshot; ordinary sessions use CLI options. This nonpersistent Part 2 mode deliberately has no user preference tool, checkpointer or restart recovery; add explicit `--persistent` for Part 3.

The compiled `build_agent_graph()` uses one model and the loop `START → agent → tools → agent`, ending when the model answers without tool calls. `bind_tools(..., parallel_tool_calls=True)` permits multiple requests; standard `ToolNode` runs independent tools concurrently. There is no second agent. Python direct functions in `helpdesk.part2.tools` provide:

| Tool | Inputs visible to the model | Local behavior |
| --- | --- | --- |
| `find_instructions` | `topic`: wifi/password | Reads complete JSON guide, identifier and steps. |
| `check_service_status` | `service`: wifi/account_recovery | Uses the trusted active snapshot; password maps to account_recovery. Missing data is unknown. |
| `prepare_ticket` | None | Validates user-derived context and completed tool evidence, then saves a local JSON draft using the supplied template. |

The root `HelpdeskAgentState` has `messages` (`add_messages`), `tool_results` (merge by call ID), `helpdesk` (current issue, exact reported steps, request/condition and draft ID), and `control` (per-turn model calls, completed tools, dispatch attempts, limits and errors). Only the agent updates the complete nested helpdesk/control values. ToolNode wrappers return `Command` updates for messages and compact evidence, including errors; parallel results merge without overwriting one another. Full tool content stays in `ToolMessage` and is sent back to the model. Traces record actual updates, relevant state snapshots, message indexes and usage metadata without repeating complete histories in every snapshot.

Frozen runtime context holds the application-selected user/thread IDs, snapshot and storage path. These are absent from the model's tool schemas and are not included in its prompt. The model receives conversation/tool messages plus system instructions and minimal application-derived ticket-readiness flags; it never receives the full internal State.

Ticket validation uses original user statements. It accepts conservative English descriptions and exact reported troubleshooting spans; unsupported/ambiguous intent requests clarification. Merely mentioning a ticket is not permission. Conditional requests require a completed operational check for the correct service in the **current user turn**. Outage, unknown, not checked, an earlier turn, or a concurrent lookup cannot authorize that write. Explicitly unconditional requests may record `not_checked`. A logical ticket-request ID, scoped by the application run, user and thread, survives changed tool-call IDs and follow-ups. An explicit new/another ticket request, or a request for a clearly separate issue, starts a new draft. A complete private file is published atomically without overwrite; concurrent calls return the same saved draft. Recognizable credential disclosures are redacted before model input and logging and block draft creation; do not enter real credentials. Unrecognized language or phrasing may require clarification. Switching topics clears earlier issue details and ticket permission.

Default budgets are six model calls and five tool dispatch attempts per user turn. Successful tool completions and total dispatch attempts are counted separately. An oversized batch is rejected as a whole, with one error ToolMessage per requested call. Limits are checked before another model call; pending calls are resolved before final termination. The LangGraph recursion limit is 16 by default (`2 * max_model_calls + 4`). API errors report an error class rather than credential-bearing exception text; tool errors return a safe result. Malformed protocol IDs are kept in diagnostics outside future model history.

Drafts default to ignored `local-storage/tickets/`; change with `HELPDESK_STORAGE_DIR` or `--storage-dir`. JSON traces default to `outputs/part2/Sxx.json` for live scenarios. Accepted traces are selected by [ACCEPTED_RUNS.json](outputs/part2/ACCEPTED_RUNS.json). S07/S09/S14 now use the corrective reruns under `outputs/part2/audit/`; their exact replay commands and draft paths are in [LIVE_STATUS.json](outputs/part2/LIVE_STATUS.json). Offline traces and explicitly fictional drafts remain separate in [outputs/part2/fake](outputs/part2/fake). Direct tool evidence is [direct_tools.json](outputs/part2/fake/direct_tools.json). Regenerating offline evidence creates new draft IDs; it never substitutes for live verification. Tests write temporary drafts under pytest's temporary directory. `.env` is ignored and must remain untracked.

Offline verification covers real LangGraph transitions with a labelled fake model, direct tools, true concurrent reads using a thread barrier, sequential write guards, malformed input/protocol, budgets, redaction, multi-turn sessions and all Part 1 regressions. No additional dependencies were required beyond the existing pinned environment. The five final accepted live traces contain **14 model calls, 8 tool executions and 18,048 tokens** (11,855 input; 6,193 output, including reported reasoning tokens). The retained S08 trace includes actual parallel guide/status requests; S06 and the corrected S09 save real local draft files. Concurrent reads and duplicate writes are also covered by deterministic graph tests. Run `python -m scripts.part2.verify_live` to verify the saved protocol, counts and draft contents without API calls. See [Part 2 verification](outputs/part2/VERIFICATION.md).

The initial S09 live response requested unnecessary contact/device details and did not create the draft after the required clarification. The system instructions were corrected to require only the fixture's problem/steps and to preserve the original request. S08 was rerun after clarifying that an explicit unconditional ticket does not require operational status. The initial traces are retained under `outputs/part2/initial/`; the current accepted traces are selected by the manifest, including the audit reruns. Responses API protocol blocks remain in in-process message history but are omitted from display/logs; CLI output uses UTF-8 for Windows Unicode replies. Live behavior remains probabilistic, and conservative English intent extraction may request clarification for unfamiliar wording.

### Part 2 audit corrections

Logical requests use a serializable `helpdesk.ticket_request_id` assigned from user-message intent within the running session. Repeated tool calls and ordinary follow-ups reuse it; `Prepare a new ticket` or `Prepare another ticket` starts a distinct request. The draft key includes run/user/thread/request identity, never the model's call ID. The immutable winner is returned even if a follow-up adds details; requesting another draft requires explicit consent. Creation writes and flushes a private temporary file, then publishes it with atomic no-overwrite `os.link`; competitors read the completed winner. Temporary files are cleaned after normal write/publish errors, leaving retries unblocked. This requires a local filesystem supporting hard links (verified on Windows/NTFS); no database or persistent conversation memory is added.

`prepare_ticket` returns only the saved category, summary and reported steps in `saved_fields`, plus its ID/path. The model is instructed to quote those facts without inferring guide details. Ticket confirmations shown by the CLI are rendered deterministically from that tool result: `response_source` is `saved_ticket_fields`, while `model_response` and the streamed AIMessage retain the real unmodified model output. Status-only replies remain model-generated. The same Python required-field checks supply a small readiness hint to prevent repeat requests for already supplied facts. An outage blocks only an operational-only request, not an explicit unconditional draft; optional personal/contact fields are never required.

All Part 2 relative application/storage/output paths resolve against the **repository root**, independent of working directory. New in-repository paths use `/` and are repository-relative; actual external paths remain absolute. The reader supports historical relative Windows backslashes without rewriting the logs and preserves existing native filenames. A foreign absolute Windows drive path on POSIX needs an explicit local mapping and is never guessed. Run `python -m scripts.part2.verify_live` (or audit-compatible `python -m tests.verify_live_evidence`) without API calls. Existing traces, new confirmations and copied-checkout loading are verified; native Linux/macOS execution has not been performed on this Windows host.

The new accepted S07/S09/S14 live logs are under `outputs/part2/audit/`; earlier root logs and first corrective attempts under `audit/initial/` remain historical. See [VERIFICATION.md](outputs/part2/VERIFICATION.md) for the corrected results and actual usage.

## Part 3 — Persistent conversation and user memory

Part 3 reuses the same single model, `HelpdeskAgentState`, reducers, graph loop and three tools. An optional SQLite checkpointer and one additional tool, `save_user_memory`, enable persistence. The root still has exactly `messages`, `tool_results`, `helpdesk` and `control`. A restored turn submits only its new HumanMessage and a reset per-turn control update; it never resubmits the accumulated history. One base SystemMessage is checkpointed. Preferences are dynamically composed into model input and are never appended as repeated SystemMessages.

After the existing setup and OpenAI configuration, start a persistent conversation with this one-line Windows command:

```powershell
python -m helpdesk.cli --graph agent --persistent --user-id demo-user-a --thread-id conversation-1 --interactive
```

Enter `/quit`, close the process, and run **the identical command** to resume. Separate one-message launches work too:

```powershell
python -m helpdesk.cli --graph agent --persistent --user-id demo-user-a --thread-id conversation-1 --message "Please remember that I prefer replies in Kazakh. My issue is Wi-Fi. I enabled it and tried reconnecting, but it still fails."
python -m helpdesk.cli --graph agent --persistent --user-id demo-user-a --thread-id conversation-1 --message "Which problem and steps did we discuss?"
```

Runtime files default to ignored `local-storage/` (override with `--storage-dir` or `HELPDESK_STORAGE_DIR`):

```text
local-storage/
  conversations.sqlite       # Genuine SqliteSaver checkpoints for all user/thread pairs
  users/<sha256-user>.json   # Separate minimal durable preferences per user
  users/<sha256-user>.lock   # OS-backed synchronization, not user data
  locks/<thread-key>.lock    # Serialize concurrent turns/resets of the same conversation
  tickets/                  # Existing atomic local ticket drafts
```

User/thread IDs are trusted application inputs, validated as 1–128 ASCII letters/digits/dot/underscore/hyphen, starting with a letter or digit. The internal checkpoint key hashes the unambiguous `(user_id, thread_id)` pair, so different users may safely use the same raw thread name. The checkpointer connection stays open inside `with PersistentSession(...)` for the graph's complete lifetime. A UUID in checkpointed `control.conversation_id` supplies the stable runtime namespace for the existing logical ticket identity. It survives process restart; resetting a thread creates a fresh UUID and cannot accidentally reuse an old draft.

The user document contains only normalized allowed preferences, for example:

```json
{"schema_version": 1, "preferences": {"preferred_language": "kk", "response_style": "concise"}}
```

Supported languages are `en`, `ru`, `kk` (English/Russian/Kazakh); style is optional `concise` or `detailed`. The model-facing memory schema has only `key` and `value`. Python validates a matching durable preference in the current user's message, allowed keys/values, length and trusted identity before writing. It rejects arbitrary facts, secrets, other-user/path/operation arguments and unsupported JSON values. JSON updates use an OS-backed lock plus a flushed temporary file and atomic replacement; simultaneous changes do not overwrite one another, and identical saves do not rewrite the file. Corrupted JSON is reported and preserved until explicitly reset. No preference is loaded from `demo_users.json`.

Preferences are read automatically before each turn and again before each model invocation, so a successful memory-tool save affects the next invocation in the same turn. Source summaries distinguish `conversation_checkpoint`, `user_json_memory` and `helpdesk_reference`. A new thread receives only user preferences, not another thread's problem history. Recognizable credential disclosures are redacted before checkpoint input, and the preference schema cannot store credentials or conversations.

Inspect or reset only the selected memory, without needing an API key:

```powershell
python -m helpdesk.cli --graph agent --persistent --user-id demo-user-a --thread-id conversation-1 --inspect-thread
python -m helpdesk.cli --graph agent --persistent --user-id demo-user-a --thread-id conversation-1 --reset-thread
python -m helpdesk.cli --graph agent --persistent --user-id demo-user-a --reset-user-memory
python -c "from pathlib import Path; from helpdesk.part3.memory_store import UserMemoryStore; print(UserMemoryStore(Path('local-storage')).load('demo-user-a'))"
```

Resetting a thread preserves preferences and existing draft files; resetting preferences preserves conversation checkpoints. That conversation may still contain the user's earlier language statement, so reset both or use a fresh thread/storage directory for a completely clean manual demonstration. Busy conversations are protected by a scoped OS lock; retries fail safely with a timeout rather than overwriting another turn. Unfinished graph invocations are rejected for new input until inspected/reset; automatic crash recovery of an interrupted tool batch is outside this implementation.

Run the exact supplied demonstrations and verifiers:

```powershell
python -m helpdesk.cli --scenario S10
python -m helpdesk.cli --scenario S11
python -m helpdesk.cli --scenario S12
python -m helpdesk.cli --scenario S13
python -m scripts.part3.verify_evidence

# No private API key or network needed:
python -m scripts.part3.record_offline
python -m pytest tests/part3 -q
python -m pytest -q
python -m pip check
```

Each scenario uses its own `local-storage/part3-scenarios/Sxx/`, resetting only that scenario's user/thread data before its first event. It does not touch manual-session storage, reset between events, or delete old ticket drafts. S11/S12 terminate one worker before starting a new Python process against the same database/users directory. Only the new fixture-event indices are sent to the second process; it receives no copied transcript. Logs record PIDs, checkpoint restoration, exact messages, available fact sources, memory-tool results and usage. Use `--output outputs/part3/another-run.json` to retain an additional run instead of replacing the default scenario log.

The [live Part 3 evidence](outputs/part3/LIVE_STATUS.json) contains **12 model calls, 8 tool executions (3 memory saves) and 22,850 tokens** (15,668 input; 7,182 output, including reported reasoning). S11 resumed in a new process and recalled only reported Wi-Fi steps in Kazakh; S12 recalled the language in a fresh thread; S13 isolated user B. Offline evidence is clearly separated under `outputs/part3/fake/`. The 67 added tests exercise actual file-backed SQLite, process restarts, concurrent JSON updates, memory-tool authorization, same-raw-thread user isolation and ticket identity after restart, alongside all 128 original regressions. See [Part 3 verification](outputs/part3/VERIFICATION.md).

Live conversation/scenario commands require OpenAI configuration. Offline tests/evidence, inspection and reset do not. Durable-preference authorization intentionally recognizes conservative preference wording; unsupported phrasing is rejected rather than saved. This is a local application with trusted application identity, not an authentication service. Native verification was on Windows; no database server, RAG, web app, Part 4 or GitHub Actions was added.

## Project layout

```text
src/helpdesk/
  cli.py                 # Shared CLI dispatcher; existing run commands stay the same
  paths.py               # Shared repository/fixture paths
  part1/
    cli.py               # Predictions and deterministic scenario execution
    state.py
    nodes.py
    graphs.py            # Original and Improved / Main graphs
    trace.py
  part2/
    cli.py               # Agent and interactive sessions
    state.py
    agent.py             # OpenAI / ToolNode graph
    runner.py            # In-process sessions and execution traces
    tools.py
    ticket_policy.py
  part3/
    locking.py             # Local cross-process synchronization
    memory_store.py        # Validated per-user JSON documents
    memory_tool.py         # The one additional preference tool
    persistence.py         # Trusted checkpoint identities and scoped inspect/reset
    runner.py              # Persistent lifecycle around the same agent graph
    scenarios.py           # Reset and real-process restart orchestration
    worker.py              # A fresh Python interpreter per process segment
    cli.py
tests/
  part1/test_routing.py
  part2/test_agent.py
  part2/test_audit_fixes.py
  part2/fake_model.py
  part3/                   # Memory, persistence, security and scenario tests
scripts/part2/
  record_offline.py       # Offline evidence generation
  verify_live.py          # Validate saved live evidence without API calls
scripts/part3/
  record_offline.py
  verify_evidence.py
data/                    # Original shared fixtures
outputs/
  part1/                 # Original and improved graph evidence
  part2/                 # Live/fake evidence and fictional draft files
  part3/                 # Live/fake memory evidence, no runtime databases/user files
```

Run each test group with `python -m pytest tests/part1 -q`, `python -m pytest tests/part2 -q` or `python -m pytest tests/part3 -q`; `python -m pytest -q` runs all three. Public CLI commands still use `python -m helpdesk.cli`. Import Part 1 builders from `helpdesk.part1.graphs`, the shared agent builder from `helpdesk.part2.agent`, and the persistence lifecycle from `helpdesk.part3.runner`. Shared fixtures, environment configuration and dependency files remain at the repository root. Historical JSON evidence retains the commands and module names used when collected; helpers are grouped by part under `scripts/`.

## Start from this repository

1. Fork [ardakshalkar/css4007-hw5](https://github.com/ardakshalkar/css4007-hw5) and keep your fork public, as for the earlier homeworks.
2. Clone your fork and work there. Do not open a pull request to the starter repository.
3. Read [SCENARIOS.md](SCENARIOS.md) to see the example conversations and what the finished application should do.
4. Read [data/README.md](data/README.md), then inspect the supplied JSON files. Use these fixtures for your demonstrations.
5. Write your code and fill in SUBMISSION.md as you complete the parts. The starter supplies the task and data; you implement the graph, tools, memory and tests.

### Set up your environment

Create a virtual environment with python -m venv .venv. Activate it with .venv\Scripts\Activate.ps1 in PowerShell, or source .venv/bin/activate on macOS/Linux. Then run python -m pip install -r requirements.txt.

Part 1 needs no API key. For the model parts, copy .env.example to .env and set your key and an available tool-capable model identifier. The dependency list includes the OpenAI adapter; if you choose another provider, replace it with the appropriate adapter and document your setup. The SQLite checkpoint package is supplied for a local storage option; explain your final storage choice.

The starter dependencies are not a tested, pinned environment for a completed application. Pin the versions you actually use once your code runs, and record your Python version. No application entry point is supplied: include your own exact run commands in your README.

## The scenario

A university IT helpdesk receives messages such as “I cannot connect to Wi-Fi” and “I forgot my password.” The assistant should find the relevant instructions, check whether a service is down, and help prepare a support ticket if the problem remains unresolved. When a user returns, it should remember the conversation and their saved preference for the language of replies.

Build this assistant in stages using **LangGraph**. Start with a small program whose route you can explain. Then add model decisions, tools and memory. A terminal application is enough.

This is a practice version of a real helpdesk workflow. Use the supplied fictional helpdesk information and invented users. It does not connect to the university's systems or submit real tickets.

## Part 1 — Understand how a graph makes a decision

First build a small helpdesk graph **without an LLM**. It handles only Wi-Fi questions and password questions.

Your graph must follow this structure:

```text
START → read_request → choose the next node
                         ├─ wifi_help     → END
                         ├─ password_help → END
                         └─ other_help    → END
```

Use these routing rules:

- If the message contains `wifi` or `wi-fi`, send it to `wifi_help`.
- Otherwise, if it contains `password`, send it to `password_help`.
- Otherwise, send it to `other_help`, which asks the user whether the problem concerns Wi-Fi or a password.
- Ignore differences between uppercase and lowercase letters.

The two help nodes should return the corresponding instructions from `data/helpdesk.json`. Use separate LangGraph nodes and a conditional edge.

### Decide what your state holds

Use at least these fields:

| Field | What it holds |
| --- | --- |
| `message` | The user's original message |
| `category` | The category identified by `read_request` |
| `reply` | The response produced by a help node |

Before writing code, draw the graph and answer:

1. What information does each node need to read? What does it change?
2. What is the difference between a node returning an update to the state and a conditional edge choosing the next node?
3. Does the next node receive only the previous node's update, or the current graph state?
4. What do `START`, `END`, and compiling a graph mean in this program?

### Predict, run, explain

Predict the route for each message before running your code:

- `I cannot connect to Wi-Fi.`
- `I forgot my PASSWORD.`
- `My laptop screen is broken.`
- `My Wi-Fi password does not work.`

Then run all four. For each run, save the order of nodes and the state after each node. Compare it with your prediction. Explain what happens when a message mentions both Wi-Fi and a password, and which rule causes that result.

Now change the graph: when both topics appear, send the request to a new `clarify_problem` node that asks which problem the user wants to solve first. Show the changed graph and the new execution trace.

Finish by explaining **what LangGraph controls here** and **what your own Python functions decide**. There is no model making decisions in this part.

## Part 2 — Let a model choose tools

Extend the same helpdesk with a model that can use three reusable tools:

| Tool | What it does |
| --- | --- |
| `find_instructions` | Looks up Wi-Fi or password instructions in the local helpdesk data |
| `check_service_status` | Looks up whether the requested service is operating or has a known outage |
| `prepare_ticket` | Produces a local ticket draft from the problem description and steps already tried |

Each tool must work when called directly, without the agent. Then connect the tools to the model through your graph. A ticket is a draft saved locally; no message is sent to a real helpdesk.

Show one request that needs more than one tool. For example: “Wi-Fi still does not work after I followed the instructions. Is it down? If it is working, help me prepare a ticket.” The assistant's choice should depend on the tool results.

Draw the execution loop: the model requests a tool, your program runs it, its result returns to the model, and the model either continues or answers. Include a limit so the loop cannot run forever, and explain where you enforce it.

In `SUBMISSION.md`, explain:

1. How does the graph decide between running a tool and ending with an answer?
2. Which component requests a tool, and which component actually executes it?
3. What changes in the state after a tool result returns?
4. What makes this part different from the fixed routing rules in Part 1?

Save an actual run showing the model requests, tool results, and final answer. State which model and provider you used. Use the supplied service-status snapshots to show that the same conditional request leads to different behaviour when Wi-Fi is operating, down, or its status is unknown. See SCENARIOS.md for the interactions.

## Part 3 — Remember the user and the conversation

Add two separate kinds of memory:

- **Conversation memory:** the current problem, previous messages and steps already tried in this conversation.
- **User memory:** a saved preference, such as “reply in Kazakh,” that is available in a later conversation for the same invented user.

Use a LangGraph checkpointer to save conversation state, and persistent storage for the user preference. Explain what a `thread_id` identifies and how it differs from a user identifier. Memory must survive closing and restarting the program.

Demonstrate these situations:

1. A user reports a Wi-Fi problem, then says “I already tried that.” The assistant uses the earlier messages to understand what “that” means.
2. Close the program and start it again. Resume the same conversation with its previous state.
3. Start a new conversation for the same invented user. The saved language preference is available, while the previous conversation is a separate thread.
4. Start a conversation for a different invented user. The first user's preference must not appear.

For each demonstration, show what was read from conversation memory, user memory, or the helpdesk reference file. Explain why looking up Wi-Fi instructions is retrieval, while recalling the user's chosen language is memory.

## Part 4 — Make it easy for someone else to run your work

This means another person can download your code, follow your instructions, and run the same demonstrations on their computer.

In your repository:

1. Add a `README.md` with the exact steps to install the required packages, configure a model API key, start the program, and run the examples. State your Python version.
2. Include a dependency file with the package versions you used. Do not make the reader guess which LangGraph version works with your code.
3. Explain where memory files are saved and how to clear them to start again.
4. Add tests that check routing, the three tools, saved memory after a restart, and separation between users and conversations. Explain what each test would catch if the code broke.
5. Set up GitHub Actions to run these tests automatically whenever you push changes. This is continuous integration, or CI. The tests must work without a private model API key; you may substitute a clearly labelled fake model in tests. Keep the real model demonstration from Part 2 separate.
6. Commit as you complete each stage, with messages explaining what changed.

Keep API keys in environment variables or an ignored `.env` file. Do not commit keys, live memory databases, or personal information. Use invented identifiers such as `demo-user-a` and `demo-user-b` in your saved examples.

## Demonstrate the finished application

Run S01–S14 from data/scenarios.json. Follow reset and restart events as explained in data/README.md. Fill in the scenario table in SUBMISSION.md and save your own logs. SCENARIOS.md shows the expected user experience; it is not a set of measured results.

## What to hand in

Submit your repository link. It should contain:

- Your LangGraph code for all four parts.
- `SUBMISSION.md` with your drawings, explanations and predictions compared with actual runs.
- Small execution logs in `outputs/` showing which nodes and tools ran and what changed in the state.
- Setup instructions, a dependency file, tests and the GitHub Actions workflow.

Push your finished work to your public fork and submit its link on the LMS. The exact HW5 LMS submission destination is still to be provided by the professor.

## References

- [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)
- [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- [LangGraph memory](https://docs.langchain.com/oss/python/langgraph/add-memory)


