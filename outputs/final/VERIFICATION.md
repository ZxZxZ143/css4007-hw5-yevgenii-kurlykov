# Final HW5 verification

Verification date: 10 November 2026 (UTC+5). All four parts are implemented. The original fixtures and Part 1–3 live, fake and historical debugging evidence are retained unchanged.

## Tested revisions and CI record

The implementation revision **2b9faeeae8f59f258ed478ee8de05fc0588a2f8d** passed [GitHub Actions run 42719394013](https://github.com/ZxZxZ143/css4007-hw5-yevgenii-kurlykov/actions/runs/42719394013) on both Ubuntu and Windows. [CI_RECORD.json](CI_RECORD.json) preserves the observed head SHA, conclusions, jobs and steps. This final report/documentation revision adds documentation validation to that workflow and is subject to the same complete checks. Its exact final SHA and successful run URL are supplied in the delivery response; no self-referential future commit hash is invented here. Check the [workflow history](https://github.com/ZxZxZ143/css4007-hw5-yevgenii-kurlykov/actions/workflows/tests.yml) against the submitted commit.

## Installation and tests

| Check | Observed result |
| --- | --- |
| Local platform/interpreter | Windows, CPython 3.13.12 |
| Clean local installation | New `.venv-part1-final`, `python -m pip install -e .`: successful; no system-site packages |
| Hosted clean checkouts | Ubuntu CPython 3.13.18 and Windows CPython 3.13.17; editable installation successful without `.env` or credentials |
| Dependency consistency | `python -m pip check`: no broken requirements |
| Complete pytest suite | **203 passed, 0 failed, 0 skipped**; clean local run took 67.02 seconds |
| Part 1 | 27 routing/trace/CLI regression cases; original and improved graphs independently exercised |
| Part 2 | 101 tools/agent/ticket/protocol/parallelism regression cases |
| Part 3 | 72 JSON/SQLite/process-restart/isolation/ticket-compatibility cases |
| Final delivery | 3 cases: full offline replay with empty provider settings, read-only evidence validation, failure exit status |
| Complete scenario replay | **14 PASS, 0 FAIL**, S01–S14; [SCENARIO_RESULTS.json](SCENARIO_RESULTS.json) |

The pinned environment is unchanged: LangGraph 1.2.20, langchain-core 1.7.6, langchain-openai 1.7.4, langgraph-checkpoint-sqlite 3.1.3, Pydantic 2.14.3, OpenAI 3.48.0, pytest 9.2.0. The complete exact versions are in [requirements.txt](../../requirements.txt). Hosted jobs install from a fresh clone and independently execute the same suite; no development files, runtime databases or API keys are required. macOS was not tested.

## Scenario and saved-live audit

S01–S04 execute Original and S05 executes Main. Actual streamed routes match pre-recorded predictions: Wi-Fi, password, other, Wi-Fi priority, then two-topic clarification. S06 checks operational status before saving a local draft; S07 outage and S08 unknown do not authorize conditional writes. S09 clarifies missing facts and then saves one unconditional draft with exact user-reported steps. S14 checks account_recovery rather than Wi-Fi. Tests also cover unconditional drafts during outages, concurrent read barriers, immutable concurrent publication, retries with changed call IDs, and exact saved-field confirmations.

S10 recalls reported steps from checkpointed messages. S11 restores the same conversation and Kazakh preference after an actual subprocess exit/restart. S12 starts a fresh thread with the same user's preference but no earlier discussion. S13 isolates the second user; separate tests use identical raw thread names for different users. Temporary offline processes, observed restoration and preferences are summarized in SCENARIO_RESULTS.json. Its retained evidence paths refer to earlier independently labelled examples; temporary full traces are deliberately discarded. Fake models verify application behavior, not fresh OpenAI language generation.

`python -m scripts.part2.verify_live --check` validated the five accepted real OpenAI traces selected by [ACCEPTED_RUNS.json](../part2/ACCEPTED_RUNS.json), including causal status evidence, actual draft contents, usage and tool-call pairing. Totals: **14 model calls, 8 tool executions, 11,855 input + 6,193 output = 18,048 tokens**. Superseded traces remain historical, excluded by the manifest.

`python -m scripts.part3.verify_evidence --check` validated genuine S10–S13 logs: **12 model calls, 8 tools including 3 preference saves, 15,668 input + 7,182 output = 22,850 tokens**. Both parts report `gpt-5-mini-2025-08-07`; no new paid OpenAI run was needed for Part 4. Live S11 process IDs are 4524 → 8616 and S12 IDs are 27604 → 2336, with zero exit status and new-event-only process inputs. Final replies were compared with their reported steps and preference sources. Saved-live evidence is never replaced with fake output. Both validators' `--check` modes preserve all artifacts byte-for-byte.

## Security and integrity audit

- Application architecture retains two deterministic graphs and one shared OpenAI model/tool loop. Persistent mode adds a checkpointer and one memory tool, preserving exactly four root State channels.
- Tool schemas expose only topic/service or allowed preference key/value. Trusted user/thread/storage/profile settings remain runtime inputs. Hashed validated identity pairs isolate checkpoints; per-user hashes isolate JSON paths.
- Tests reject forged identity/path/operation arguments, invalid preferences, secrets, corrupt JSON and path-like IDs. OS locks and flushed atomic replacement protect concurrent user-memory updates. The connection remains alive for the compiled graph's lifetime.
- Ticket writes require user-derived consent/details. Operational-only writes require a completed current-turn check for the correct service. A same-batch lookup cannot authorize a write. Stable checkpointed conversation UUIDs preserve draft identity across fresh processes; explicit new requests get new logical IDs. Atomic hard links publish one immutable completed file without overwrite.
- Tool/model failures close or reject protocol safely, redact internal exception details, and respect bounded calls. Recognizable secret disclosures are redacted before model/checkpoint input and block ticket/preference storage. Detection is heuristic, so this is not a general data-loss-prevention system.
- `.env` remains ignored/untracked. Tracked paths exclude private runtime databases and user-memory documents. All reachable Git blobs were checked for the configured key and long-form API credential patterns without printing values; none were found. Existing short synthetic rejection-test values are labelled test fixtures.
- Fixtures and prior evidence have no diff from the Part 3 baseline. Unrelated local `.idea/` files are not included. Local Markdown destinations, all fourteen scenario rows, CLI routing modes, storage resets and read-only verification commands were checked. Existing CLI tests cover live-facing modes with fake adapters, avoiding unnecessary paid requests.

## Reproduction and limitations

From the source checkout, install with `python -m pip install -e .`, then run `python -m pip check`, `python -m pytest -q`, `python -m scripts.verify_all`, both evidence commands above and `python -m scripts.check_docs`. All are credential-free. Optional `--output <path>` on verify_all selects a summary destination; otherwise no tracked artifacts change. [README.md](../../README.md) provides Windows/POSIX setup, model configuration, start/resume, inspect/reset, scenario commands and storage locations. [SUBMISSION.md](../../SUBMISSION.md) completes all four parts and all fourteen PASS rows.

Limits remain local trusted identity rather than authentication, conservative English intent/preference parsing, heuristic redaction, local-filesystem hard-link requirements, no history pruning, and explicit inspection/reset for unfinished tool batches. Fictional fixtures are not real university services. A source checkout is required for fixtures/scripts/tests. CI validates Ubuntu and Windows; macOS is untested. Live wording is probabilistic. No required scenario or regression failure remains in the recorded implementation run; the final delivery must additionally name a successful CI run on its exact pushed SHA.
