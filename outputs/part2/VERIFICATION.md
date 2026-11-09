# Part 2 verification — offline and live verified

Date: 10 November 2026. Environment: CPython 3.13.12 / Windows, LangGraph 1.2.20, langchain-openai 1.7.4, pytest 9.2.0. Existing exact dependency pins were retained; no new package was needed.

`python -m pytest -q`: **128 passed, 0 failed, 0 skipped**, including all original 98 tests and 30 new audit-regression cases. `python -m pip check`: no broken requirements. Both Part 1 graph builders and the agent compile and execute. Tests remain independent of private credentials; separate live demonstrations below use the configured OpenAI account.

Verified directly and through real graph execution:

- JSON instructions for both supported topics; both services in all four snapshots; unknown and invalid input handling.
- Template-shaped local draft, actual reported steps, application identity and missing-details rejection. Logical request idempotence covers same/different call IDs, follow-ups, concurrent ToolNode calls and two publishers synchronized immediately before atomic publication.
- Real overlapping reads using a two-thread barrier; preserved result IDs, exactly one ToolMessage per admitted call, and both results received by the next model invocation.
- Same-batch status/ticket rejection followed by a successful later write; wrong service, earlier-turn status, outage and unknown cannot authorize an operational-only draft.
- Model-generated snapshot/condition/steps arguments cannot bypass trusted context. Merely mentioning a ticket, hypothetical troubleshooting and ambiguous conditions do not authorize writes.
- Whole-batch budget rejection, bounded repetitive loops, tool/model exceptions, malformed arguments and malformed protocol termination.
- Multi-turn state, isolated sessions, interactive CLI, token-metadata capture and unchanged Part 1 defaults/evidence.
- Responses API final text excludes encrypted/protocol blocks; those blocks remain available only in the running model history. UTF-8 CLI output handles Unicode outside the Windows legacy code page.
- Same-request retries return the immutable saved file; explicit new/another requests create a new logical ID. User/thread isolation, changed follow-up data, rejected conditional retries, interrupted private writes and failed publication are tested. All temporary files are cleaned on handled failure and a legitimate retry succeeds.
- Ticket confirmations contain exactly the actual saved ID/path/category/summary/steps. Tests deliberately return invented model details, verify that none are shown as saved facts, and confirm that the real AIMessage/model_response is preserved.
- Relative Windows and POSIX evidence paths, external path identity, copied-checkout loading and working-directory independence are tested. Native Linux/macOS execution was not available on this Windows host; the same tests are portable to those systems.

`python -m scripts.part2.record_offline` generated [direct tool evidence](fake/direct_tools.json) and the five **deterministic fake-model** scenario traces under `fake/`. These are actual local graph/tool executions, not OpenAI results. S06 and S09 created fictional local example drafts; S07/S08/S14 created no draft. Paths and draft identifiers are recorded in the actual tool results.

## Live OpenAI evidence

The model was configured as `gpt-5-mini`; all actual AIMessage metadata identifies `gpt-5-mini-2025-08-07`. The same application scenario runner used by the CLI executed the exact fixture inputs, with separate fresh state per scenario and shared state across the two turns of S06/S09. [ACCEPTED_RUNS.json](ACCEPTED_RUNS.json) selects the retained S06/S08 evidence plus new corrected S07/S09/S14 runs under `audit/`. Earlier root and initial logs remain unchanged. Log commands identify the equivalent CLI replay including output/storage arguments.

| Scenario | Verified outcome | Model calls | Tools | Input / output / total tokens |
| --- | --- | --- | --- | --- |
| [S06](S06.json) | Guide retrieval, completed operational status, then a real local draft with only reported steps. | 5 | 3 | 3,785 / 1,257 / 5,042 |
| [S07](audit/S07.json) | Explains that this particular operational-only condition failed; unconditional drafting remains possible. No draft or fabricated repair time. | 2 | 1 | 1,931 / 1,531 / 3,462 |
| [S08](S08.json) | Unknown status; no conditional draft; appropriate next-step offer. | 2 | 2 | 1,497 / 1,331 / 2,828 |
| [S09](audit/S09.json) | Required-fields clarification followed by one unconditional draft marked not_checked; confirmation exactly matches the saved file. | 3 | 1 | 2,933 / 1,395 / 4,328 |
| [S14](audit/S14.json) | account_recovery outage; no invented ticket condition, mandatory personal/contact fields, Wi-Fi check or draft. | 2 | 1 | 1,709 / 679 / 2,388 |

Current accepted totals: **14 model calls, 8 tool executions, 11,855 input tokens, 6,193 output tokens, 18,048 total tokens**. Output totals include reported reasoning-token usage. Retained S08 demonstrates parallel instruction/status requests. S06 creates its conditional draft after a completed operational check; corrected S09 is unconditional and does not require a status check. Accepted draft files are under `live_drafts/S06/tickets/` and `audit/drafts/S09/tickets/`; the older S09 draft is retained as historical evidence.

`python -m scripts.part2.verify_live` and the audit-compatible `python -m tests.verify_live_evidence` checked actual tool-call/response pairing, normal completion, usage counts, expected statuses/services, draft existence/template fields, exact reported steps, structured confirmation fields and earlier-step authorization for the conditional write. [LIVE_STATUS.json](LIVE_STATUS.json) is generated from these selected traces; verification makes no API calls. The verifier was also run against all five original committed root traces and a copied checkout without rewriting them. Historical relative backslashes are normalized only on reading; foreign absolute drive paths require a mapping rather than being rebased incorrectly.

The [initial S09 attempt](initial/S09.json) did not create the requested second-turn draft because the model required irrelevant contact/device information. The system prompt now specifies the actual template requirements and instructs it to honor the earlier request when clarification is supplied. The [initial S08 attempt](initial/S08.json) respected the condition but described the restriction too broadly; the prompt now distinguishes unconditional requests explicitly. Both were rerun and passed. Initial attempts are retained separately and excluded from the final totals above.

The first corrective attempts are separately preserved under `audit/initial/`: S09 asked for an already supplied description, while S14 invented a ticket condition. The final prompt now applies the conditional explanation only to an actual conditional request. Minimal readiness flags, computed with the same Python required-field function used by the tool, prevent the model from guessing whether supplied information is sufficient. Accepted S07/S09/S14 reruns passed. The model's raw S09 response also matches the saved fields; the application confirmation is additionally deterministic and labelled separately from that raw output.

Atomic creation uses a fully written/flushed temporary file in the destination directory and `os.link` to publish without overwrite. This is tested on the local Windows filesystem and requires hard-link support. Other I/O errors propagate, leaving no incomplete final reservation. An already completed request is returned as a read without another operational-status prerequisite; a genuinely new conditional write retains all guards. Idempotency is scoped to the in-process run/user/thread/logical request, not cross-process conversation recovery. Part 3 has not been implemented.

The key is stored only in local `.env` at the user's request. Existing `.gitignore` rules ignore it; it is not tracked, staged or included in evidence. No secret is printed in configuration checks. Earlier missing-configuration status has been superseded by the verified live results.

Parts 3–4, S10–S13, supplied fixtures and the completed Part 1 implementation/report/evidence are preserved. Existing `.idea/` and virtual environments are excluded from the implementation commit. Persistent conversation/user memory remains outside Part 2.
