# Part 1 verification

Verified on 10 November 2026 with CPython 3.13.12 on Windows, LangGraph 1.2.20 and pytest 9.2.0.

- Installed dependencies in a new `.venv-part1` environment, recorded their exact versions, then installed `requirements.txt` and the editable package in a second clean `.venv-part1-check` environment.
- `python -m pip check`: `No broken requirements found.`
- `python -m pytest -q` in the second environment: **27 passed, 0 failed**.
- Both graph builders compile independently. Original has no clarification node; Main contains it.
- Independent CLI processes with `--graph original`, `--graph main`, and no graph argument returned `wifi_help`, `clarify_problem`, and `clarify_problem`, respectively, for the identical two-topic message.
- Ran `python -m helpdesk.cli --scenario S01` through S05. All five predictions matched real streamed node execution. Saved JSON files are included alongside this report; the tests compare them with fresh runs.
- Tests verify both graph versions on Wi-Fi-only, password-only, mixed-case and unsupported inputs; two-topic behavior; JSON guide changes after compilation; partial updates and complete states; read-only conditional routing; scenario version enforcement; and default execution without provider imports, API keys or network connections.
- The supplied `data/` files are unchanged. Parts 2–4, Choices and limitations, and the S06–S14 submission rows were compared with the starter and remain unchanged.
- `git diff --check` passed. Existing `.idea/` and `.venv/` were left untouched and excluded from this implementation's commit.

Part 1 has no unresolved execution failures. Only the editable checkout setup on Python 3.13/Windows was verified. Model tools, service-status decisions, persistent memory and CI remain future assignment work.
