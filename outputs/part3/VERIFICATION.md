# Part 3 verification — persistent conversation and user preferences

Verified on 10 November 2026 with CPython 3.13.12 / Windows, LangGraph 1.2.20, langgraph-checkpoint-sqlite 3.1.3, langchain-openai 1.7.4 and pytest 9.2.0. The existing dependency pins were sufficient.

`python -m pytest -q`: **200 passed, 0 failed, 0 skipped** in the second verification environment, including all 128 previous tests and 72 new Part 3 cases. `python -m pip check`: no broken requirements.

## Genuine live demonstrations

Provider: OpenAI. Configured model: `gpt-5-mini`. Actual API model identifier: `gpt-5-mini-2025-08-07`.

| Scenario | Actual process(es) | Verified result | Model / tool / memory calls | Input / output / total tokens |
| --- | --- | --- | --- | --- |
| [S10](S10.json) | 29896 | Second turn loaded the SQLite checkpoint and recalled only enabling Wi-Fi, forgetting the saved network and failed reconnection. | 3 / 2 / 0 | 3,787 / 1,558 / 5,345 |
| [S11](S11.json) | 4524 → 8616 | First process saved Kazakh through the real memory tool and exited. Fresh process restored conversation UUID/history and user JSON; recalled the issue and reported enabling/reconnecting in Kazakh. | 3 / 3 / 1 | 4,270 / 2,203 / 6,473 |
| [S12](S12.json) | 27604 → 2336 | New process/thread had no earlier conversation messages but loaded the same user's JSON preference. It identified Kazakh and support scope without presenting the first thread's Wi-Fi problem as a discussion in this thread. | 3 / 2 / 1 | 3,939 / 2,151 / 6,090 |
| [S13](S13.json) | 23508, two isolated identities | User A saved Kazakh. User B's checkpoint and preference lookup were empty; its response reported no recorded preference or prior problem. | 3 / 1 / 1 | 3,672 / 1,270 / 4,942 |

Totals: **12 model calls, 8 completed tool executions, including 3 successful save_user_memory calls; 15,668 input tokens, 7,182 output tokens, 22,850 total tokens**. Output usage includes reported reasoning tokens. All turns terminated normally, with no budget stop or tool error.

The runner used the exact `data/scenarios.json` inputs. It reset each scenario's relevant state/preferences only before the first event, inside its private scenario directory. For restart events, the parent waited for worker termination and then launched a fresh Python interpreter with the same storage. Each worker received only its new fixture-event indices. No previous conversation was copied into the next process input. PIDs, exit status, event indices, checkpoint identity/restoration and before/after message counts are captured in each log.

`python -m scripts.part3.verify_evidence` produced [LIVE_STATUS.json](LIVE_STATUS.json) from saved actual traces. It checks exact messages, call/ToolMessage IDs, successful memory writes, same-turn refreshed preference reads, user/thread namespacing, process boundaries, restored versus new conversations, message counts and token usage without reading private database files or making API calls. Final replies were inspected for correct reported steps, Kazakh responses and user-B isolation.

## Offline verification

`python -m scripts.part3.record_offline` created separately labelled [fake/](fake/) evidence using the real graph, SQLite and JSON stores. Its model is explicitly deterministic and is not presented as OpenAI. S11/S12 still cross real subprocess boundaries. Offline totals: 12 fake model invocations, 4 local tools, including 3 preference saves; no measured API token usage.

The 72 new cases verify:

- File-backed checkpoint creation, same-thread continuity, new-process recovery, preserved helpdesk fields/tool evidence and reset per-turn counters.
- Exactly one base SystemMessage, one newly submitted HumanMessage per turn, no preference prompt duplication, and no replay of completed side effects when a conversation is reopened.
- Same ticket ID/path after actual process restart with a different call ID, a distinct draft only after an explicit new request, conditional status guards after restart, and parallel duplicate calls returning one immutable published draft.
- Missing/corrupted JSON, strict schema/key/value validation, normalized languages, idempotent saves, atomic update failures, concurrent thread/process updates and immediate next-model visibility of a changed preference.
- User/thread isolation including identical raw thread names, rejected path-like identifiers and forbidden model identity/path/operation arguments. Secrets are rejected by the preference schema and recognizable user disclosures are redacted before checkpoint input.
- Actual S10–S13 fixture replay/reset behavior, persistent CLI and inspection/reset commands, interactive mode, and all original Part 1/Part 2 regressions.

Runtime data remains private and ignored: default manual storage is `local-storage/conversations.sqlite` plus `local-storage/users/<safe-hash>.json`; live scenarios use `local-storage/part3-scenarios/Sxx/`, while offline scenarios use `local-storage/part3-offline/part3-scenarios/Sxx/`. No SQLite database, actual preference file or runtime lock is included in these evidence artifacts. Provenance summaries expose only the invented demonstration identities and allowed preference values.

The original Part 1/Part 2 fixtures, reports and execution evidence are preserved. No model/agent duplication, extra service, database server, RAG, website, Part 4 or GitHub Actions was added. The same four root State channels and reducers remain; stable conversation identity is one optional field in `control`.

Remaining limits: local trusted identity and a small allowlisted preference schema, conservative intent matching, bounded lock waits, no automatic checkpoint pruning, and explicit rejection of unfinished invocations pending inspection/reset. Native tests ran on Windows; the OS lock implementation also has a POSIX branch. Persistent ticket/JSON atomicity still depends on the documented local filesystem support from Part 2.
