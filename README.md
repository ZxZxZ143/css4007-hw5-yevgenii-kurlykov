# HW5 — Build a university IT helpdesk with LangGraph

**CSS-4007 · Artificial Intelligence**  
**Due:** 8 November 2026, 23:59 (UTC+5)

## Implementation status

Tested with **CPython 3.13.12 on Windows**, LangGraph **1.2.20**, and pytest **9.2.0**. Part 1 is verified. Part 2 implementation and offline tests are verified; live OpenAI demonstrations are **pending** because neither `OPENAI_API_KEY` nor `MODEL_NAME` is configured in the current environment. Parts 3–4 remain future work. The original assignment remains below.

From the repository root, create a fresh environment and install the pinned dependencies and editable local package:

```powershell
python -m venv .venv-part1
.\.venv-part1\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
```

On macOS/Linux, activate with `source .venv-part1/bin/activate`. Use Python 3.13 for the tested setup. Run the following commands from the repository root after activation. Part 1 never loads `.env`, calls a model, or requires an API key. The pinned provider and dotenv dependencies support Part 2; SQLite is reserved for Part 3.

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

Scenario replay binds S01–S04 to Original and S05 to Improved / Main, as required by the fixtures. A contradictory explicit `--graph` is rejected. The fixture name `with_clarification` maps to CLI version `main`. S06–S09 and S14 use the Part 2 agent below; S10–S13 are not implemented yet.

Each invocation prints JSON. Scenario commands also save traces in `outputs/part1/original/S01.json` through `S04.json` and `outputs/part1/improved/S05.json`. For a custom message, add `--output outputs/part1/custom.json` to save its trace. These files contain the command, fixture context, prediction, executed nodes, partial updates, complete states and final reply. `START` and `END` are boundary sentinels, so the executed-node list contains only real Python nodes. Traces use LangGraph's `updates` and `values` streams; they are not reconstructed by calling the node functions separately.

The predictions were recorded before implementation/execution in [PREDICTIONS.md](outputs/part1/PREDICTIONS.md). The explanation and measured results are in [SUBMISSION.md](SUBMISSION.md). Instructions are read from `data/helpdesk.json` at execution time. No supplied fixture was edited. Part 1 has no persistent memory, so every invocation starts fresh.

Code: `src/helpdesk/{state,nodes,graphs,trace,cli}.py`. Import `build_original_graph` or `build_main_graph` from `helpdesk.graphs` to compile either graph independently. Both use shared help nodes; only Main includes `clarify_problem` and the two-topic classification. The tests execute real graphs, compare saved evidence with fresh runs, verify fixture retrieval and preserved state, and run the default CLI with provider imports and network connections blocked. The Part 1 tests use no fake model.

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
python -m tests.record_part2
```

Enter `/quit` or EOF to end interactive mode. The same process retains the current conversation, including S06/S09 follow-up details. Scenario replay reads exact fixture events and snapshot names, starts isolated state for each scenario, and writes live execution logs to `outputs/part2/Sxx.json`. The scenario fixture controls identity and snapshot; ordinary sessions use CLI options. No user preferences, checkpointer or restart recovery are implemented.

The compiled `build_agent_graph()` uses one model and the loop `START → agent → tools → agent`, ending when the model answers without tool calls. `bind_tools(..., parallel_tool_calls=True)` permits multiple requests; standard `ToolNode` runs independent tools concurrently. There is no second agent. Python direct functions in `helpdesk.tools` provide:

| Tool | Inputs visible to the model | Local behavior |
| --- | --- | --- |
| `find_instructions` | `topic`: wifi/password | Reads complete JSON guide, identifier and steps. |
| `check_service_status` | `service`: wifi/account_recovery | Uses the trusted active snapshot; password maps to account_recovery. Missing data is unknown. |
| `prepare_ticket` | None | Validates user-derived context and completed tool evidence, then saves a local JSON draft using the supplied template. |

The root `HelpdeskAgentState` has `messages` (`add_messages`), `tool_results` (merge by call ID), `helpdesk` (current issue, exact reported steps, request/condition and draft ID), and `control` (per-turn model calls, completed tools, dispatch attempts, limits and errors). Only the agent updates the complete nested helpdesk/control values. ToolNode wrappers return `Command` updates for messages and compact evidence, including errors; parallel results merge without overwriting one another. Full tool content stays in `ToolMessage` and is sent back to the model. Traces record actual updates, relevant state snapshots, message indexes and usage metadata without repeating complete histories in every snapshot.

Frozen runtime context holds the application-selected user/thread IDs, snapshot and storage path. These are absent from the model's tool schemas and are not included in its prompt. The model receives only the system instructions and conversation/tool messages.

Ticket validation uses original user statements. It accepts conservative English descriptions and exact reported troubleshooting spans; unsupported/ambiguous intent requests clarification. Merely mentioning a ticket is not permission. Conditional requests require a completed operational check for the correct service in the **current user turn**. Outage, unknown, not checked, an earlier turn, or a concurrent lookup cannot authorize that write. Explicitly unconditional requests may record `not_checked`. Same-call retries use the same draft ID and exclusive creation. Recognizable credential disclosures are redacted before model input and logging and block draft creation; do not enter real credentials. Unrecognized language or phrasing may require clarification. Switching topics clears earlier issue details and ticket permission.

Default budgets are six model calls and five tool dispatch attempts per user turn. Successful tool completions and total dispatch attempts are counted separately. An oversized batch is rejected as a whole, with one error ToolMessage per requested call. Limits are checked before another model call; pending calls are resolved before final termination. The LangGraph recursion limit is 16 by default (`2 * max_model_calls + 4`). API errors report an error class rather than credential-bearing exception text; tool errors return a safe result. Malformed protocol IDs are kept in diagnostics outside future model history.

Drafts default to ignored `local-storage/tickets/`; change with `HELPDESK_STORAGE_DIR` or `--storage-dir`. JSON traces default to `outputs/part2/Sxx.json` for live scenarios. Offline traces and explicitly fictional drafts are in [outputs/part2/fake](outputs/part2/fake); [LIVE_STATUS.json](outputs/part2/LIVE_STATUS.json) records the missing live configuration. Direct tool evidence is [direct_tools.json](outputs/part2/fake/direct_tools.json). Regenerating offline evidence creates new draft IDs; it never substitutes for live verification. Tests write temporary drafts under pytest's temporary directory.

Offline verification covers real LangGraph transitions with a labelled fake model, direct tools, true concurrent reads using a thread barrier, sequential write guards, malformed input/protocol, budgets, redaction, multi-turn sessions and all Part 1 regressions. No additional dependencies were required beyond the existing pinned environment. See [Part 2 verification](outputs/part2/VERIFICATION.md). Live model choice, real token usage and real scenario behavior remain unverified until credentials are configured and the five live commands succeed.

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


