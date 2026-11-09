# Part 2 verification — offline and live verified

Date: 10 November 2026. Environment: CPython 3.13.12 / Windows, LangGraph 1.2.20, langchain-openai 1.7.4, pytest 9.2.0. Existing exact dependency pins were retained; no new package was needed.

`python -m pytest -q`: **98 passed, 0 failed, 0 skipped**, including all 27 Part 1 tests. `python -m pip check`: no broken requirements. Both Part 1 graph builders and the agent compile and execute. Tests remain independent of private credentials; separate live demonstrations below use the configured OpenAI account.

Verified directly and through real graph execution:

- JSON instructions for both supported topics; both services in all four snapshots; unknown and invalid input handling.
- Template-shaped local draft, actual reported steps, application identity, missing-details rejection and same-call idempotence.
- Real overlapping reads using a two-thread barrier; preserved result IDs, exactly one ToolMessage per admitted call, and both results received by the next model invocation.
- Same-batch status/ticket rejection followed by a successful later write; wrong service, earlier-turn status, outage and unknown cannot authorize an operational-only draft.
- Model-generated snapshot/condition/steps arguments cannot bypass trusted context. Merely mentioning a ticket, hypothetical troubleshooting and ambiguous conditions do not authorize writes.
- Whole-batch budget rejection, bounded repetitive loops, tool/model exceptions, malformed arguments and malformed protocol termination.
- Multi-turn state, isolated sessions, interactive CLI, token-metadata capture and unchanged Part 1 defaults/evidence.
- Responses API final text excludes encrypted/protocol blocks; those blocks remain available only in the running model history. UTF-8 CLI output handles Unicode outside the Windows legacy code page.

`python -m scripts.part2.record_offline` generated [direct tool evidence](fake/direct_tools.json) and the five **deterministic fake-model** scenario traces under `fake/`. These are actual local graph/tool executions, not OpenAI results. S06 and S09 created fictional local example drafts; S07/S08/S14 created no draft. Paths and draft identifiers are recorded in the actual tool results.

## Live OpenAI evidence

The model was configured as `gpt-5-mini`; all actual AIMessage metadata identifies `gpt-5-mini-2025-08-07`. The same application scenario runner used by the CLI executed the exact fixture inputs, with separate fresh state per scenario and shared state across the two turns of S06/S09. Logs identify direct Python collection and equivalent CLI replay commands, including the draft-storage directory.

| Scenario | Verified outcome | Model calls | Tools | Input / output / total tokens |
| --- | --- | --- | --- | --- |
| [S06](S06.json) | Guide retrieval, completed operational status, then a real local draft with only reported steps. | 5 | 3 | 3,785 / 1,257 / 5,042 |
| [S07](S07.json) | Wi-Fi outage; no conditional draft or fabricated repair time. | 3 | 2 | 2,576 / 1,659 / 4,235 |
| [S08](S08.json) | Unknown status; no conditional draft; appropriate next-step offer. | 2 | 2 | 1,497 / 1,331 / 2,828 |
| [S09](S09.json) | No first-turn write; clarification followed by a real draft with supplied steps and checked operational status. | 4 | 3 | 3,662 / 1,389 / 5,051 |
| [S14](S14.json) | account_recovery outage; no Wi-Fi check, draft or fabricated restoration time. | 2 | 1 | 864 / 1,196 / 2,060 |

Final accepted totals: **16 model calls, 11 tool executions, 12,384 input tokens, 6,832 output tokens, 19,216 total tokens**. Output totals include reported reasoning-token usage. S08/S09 contain actual parallel instruction/status requests; S06/S09 create draft files only after prerequisite tool results. The two committed live example drafts are under `live_drafts/S06/tickets/` and `live_drafts/S09/tickets/`.

`python -m scripts.part2.verify_live` checked actual tool-call/response pairing, normal completion, usage counts, expected statuses/services, draft existence/template fields, exact reported steps and earlier-step authorization for the conditional write. [LIVE_STATUS.json](LIVE_STATUS.json) is generated from these saved traces; verification makes no API calls.

The [initial S09 attempt](initial/S09.json) did not create the requested second-turn draft because the model required irrelevant contact/device information. The system prompt now specifies the actual template requirements and instructs it to honor the earlier request when clarification is supplied. The [initial S08 attempt](initial/S08.json) respected the condition but described the restriction too broadly; the prompt now distinguishes unconditional requests explicitly. Both were rerun and passed. Initial attempts are retained separately and excluded from the final totals above.

The key is stored only in local `.env` at the user's request. Existing `.gitignore` rules ignore it; it is not tracked, staged or included in evidence. No secret is printed in configuration checks. Earlier missing-configuration status has been superseded by the verified live results.

Parts 3–4, S10–S13, supplied fixtures and the completed Part 1 implementation/report/evidence are preserved. Existing `.idea/` and virtual environments are excluded from the implementation commit. Persistent conversation/user memory remains outside Part 2.
