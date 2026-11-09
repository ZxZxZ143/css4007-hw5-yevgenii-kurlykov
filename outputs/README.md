# Save your own results here

Commit small text or JSON logs for the supplied scenarios. Start each log with its scenario identifier and record the command, snapshot, user and thread. Include your actual node/tool trace and final reply. Keep local ticket drafts here if they contain only invented demo data. Store live memory under the ignored local-storage directory.

This starter contains no completed runs. Do not present the illustrative conversations from SCENARIOS.md as results from your code.

## Implemented Part 1 evidence

The saved actual runs are now in `part1/original/S01.json` through `S04.json` and `part1/improved/S05.json`. Regenerate them from the repository root using `python -m helpdesk.cli --scenario S01` (and similarly S02–S05) after the README setup. Predictions made before implementation/execution are in [part1/PREDICTIONS.md](part1/PREDICTIONS.md).

Each JSON log contains the command, fixture context, selected graph version, exact message, predicted and executed node order, node updates, complete post-node states, final response and prediction comparison. Updates and states come from the same real LangGraph stream. START/END are documented boundary sentinels, not entries fabricated as executed Python nodes. The selected status profile is recorded as fixture context only; Part 1 does not query service status, use tools or retain memory.

`pytest` compares fresh scenario executions against these saved logs (excluding only the destination-dependent command).

## Part 2 evidence status

[part2/fake](part2/fake) contains real local graph/tool executions driven by an explicitly labelled deterministic fake model, plus fictional example drafts and direct tool-call results. Regenerate with `python -m tests.record_part2`. These files are **not OpenAI demonstrations**. [part2/LIVE_STATUS.json](part2/LIVE_STATUS.json) records configuration checks showing that the five live commands could not start because the API key and model name were missing. Live runs will save `part2/S06.json`, `S07.json`, `S08.json`, `S09.json` and `S14.json` once configured. See [part2/VERIFICATION.md](part2/VERIFICATION.md). Part 3 evidence is not supplied yet.
