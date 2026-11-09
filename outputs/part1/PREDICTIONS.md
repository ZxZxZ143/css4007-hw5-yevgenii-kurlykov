# Part 1 predictions — recorded before implementation and execution

The fixture messages are used unchanged. These predictions follow the assignment's keyword rules; they are not measured results.

| Scenario | Version | Exact message | Predicted nodes (between START and END) |
| --- | --- | --- | --- |
| S01 | Original / Initial | I cannot connect to Wi-Fi. | read_request → wifi_help |
| S02 | Original / Initial | I forgot my PASSWORD. | read_request → password_help |
| S03 | Original / Initial | My laptop screen is broken. | read_request → other_help |
| S04 | Original / Initial | My Wi-Fi password does not work. | read_request → wifi_help |
| S05 | Improved / Main | My Wi-Fi password does not work. | read_request → clarify_problem |

Original: `START → read_request → {wifi_help, password_help, other_help} → END`.
Improved: `START → read_request → {wifi_help, password_help, other_help, clarify_problem} → END`.

State design before coding: `message` preserves the original input; `read_request` reads it and updates only `category`. A conditional edge reads `category` and chooses a node without changing state. Wi-Fi and password nodes read the category to retrieve the matching JSON guide and update only `reply`; the other and clarification nodes return their question as `reply`. Every node receives the complete current state, including fields preserved from earlier steps. START and END are entry/termination sentinels rather than executed Python nodes. Compilation validates and prepares the graph for invocation or streaming.

Original checks Wi-Fi first, so S04 must take Wi-Fi precedence. Improved checks for both topics first, so S05 must ask which to solve first. The improved graph should retain S01–S03 behavior.
