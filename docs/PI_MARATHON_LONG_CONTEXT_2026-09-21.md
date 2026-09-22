# Pi versus Marathon: six-turn long-context project

Marathon completed this project in 13.36 minutes versus Pi's 17.53 minutes, taking 23.8% less time.
Both passed every cumulative check, including all 13 final checks, with no observed correctness regressions.
Actual context reached 139043 tokens for Marathon and 148695 for Pi.
This is one controlled project run per harness, not a general quality or speed ranking.

## Results

| Metric | Marathon | Pi |
| --- | ---: | ---: |
| Completion time, excluding backend loading | 801.71 s | 1051.99 s |
| Final independent checks | 13/13 | 13/13 |
| Cumulative check executions across six turns | 48/48 | 48/48 |
| Newly processed input tokens | 148577 | 119386 |
| Generated tokens, including reasoning | 34202 | 61268 |
| Cached input tokens summed over requests | 6815276 | 6553781 |
| Inference requests | 67 | 80 |
| Maximum actual request context | 139043 | 148695 |
| Backend prompt-processing time | 260.15 s | 197.86 s |
| Backend generation time | 484.82 s | 807.48 s |

Pi processed 19.6% fewer uncached input tokens but generated 79.1% more output tokens.
Its additional generation time outweighed its prefill savings.
Backend generation averaged approximately 70.5 tokens/s for Marathon and 75.9 tokens/s for Pi, computed as all predicted tokens divided by all generation time.
Pi therefore did not lose this comparison because its aggregate backend decode throughput was lower.
The result reflects different implementation, reasoning, tool-use, and testing trajectories under the native harnesses; it does not isolate a single causal prompt or tool difference.

Cached input sums count repeated history and must not be interpreted as unique conversation size.
Actual request context is measured separately as backend `prompt_n + cache_n`.
All 147 inference requests returned HTTP 200, all harness turns exited successfully, and no request reached the 6144-token generation cap.

| Turn | Feature | Marathon seconds | Pi seconds | Cumulative checks, both | Marathon max context | Pi max context |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| 1 | SQLite persistence, validation, idempotency | 96.06 | 98.84 | 3/3 | 34213 | 29144 |
| 2 | Persistent reservations | 116.31 | 207.15 | 5/5 | 57859 | 43287 |
| 3 | Atomic transfers | 100.77 | 104.51 | 7/7 | 79975 | 67095 |
| 4 | Reversals respecting availability and replay | 177.03 | 320.86 | 9/9 | 108554 | 109184 |
| 5 | Atomic batches and CSV imports | 211.28 | 210.00 | 11/11 | 135783 | 141111 |
| 6 | Snapshot reports and real CLI | 100.26 | 110.63 | 13/13 | 139043 | 148695 |

Most of the completion-time separation came from reservations and reversals.
On reversals, Pi generated 19001 tokens compared with Marathon's 7566.
Both ultimately met the tested reversal contracts.
Neither run was given the independent test results as corrective feedback.

## Workload and controls

The fixture is `scripts/evals/long_context_project.py`.
Each harness started from the same nine-file Python package and evolved it through six user turns, retaining its own conversation and modified project throughout.
The package implements an inventory ledger with interacting persistence, replay, reservation, transfer, reversal, rollback, reporting, and command-line requirements.
Independent cumulative checks were validated against a separate reference implementation before execution.
Each harness was also free to write and run its own project tests.

The first five prompts each included approximately 18000 backend-tokenizer-calibrated tokens of synthetic historical audit records.
Both harnesses received byte-identical task text and audit packets.
The packets were explicitly background material, not events to import or instructions.
This is controlled long-context loading, not naturally accumulated months of project history, a large existing repository, or a retrieval benchmark.
It tests implementing interacting requirements while carrying substantial prior context, but does not fully reproduce a real 100K-token debugging conversation.
The six-turn project remains modest compared with a production codebase.

Controls matched the earlier small benchmark:

- Same deployed Swift/uncensored Qwen 3.8 27B IQ4_XS target, tuned DFlash2 drafter, production image, Q8 target cache, and 196000-token context capacity.
- Same isolated RTX 3090 on GPU 3 at its existing 250 W limit, protected by the shared Marathon GPU lease.
- Medium reasoning, temperature zero, seed 8123, top-p 1, top-k 0, min-p 0, repeat penalty 1, and 6144 output tokens per inference request.
- Same personal AGENTS instructions and skill-description catalog, with Pi explicitly receiving the shared policy file from the matched small benchmark.
- Marathon 0.4.0 using its Responses adapter; Pi 0.73.1 using Chat Completions.
- Native system prompts, tool schemas, and sandbox behavior retained as part of the whole-harness comparison.
- Fresh workspaces and a cold backend for each harness; cache reuse enabled within each six-turn session.
- Pi automatic compaction disabled; no compaction event observed in either run.
- Ten-minute timeout per turn, with no timeout reached; network access and delegation prohibited by the task prompts.

Marathon ran first and Pi second.
There was no counterbalanced repeat, so order effects and generation-path variability remain limitations.
Identical greedy sampling settings do not make different harness prompts generate identical code or test suites.
Passing these checks demonstrates the specified cases, not proof of complete correctness.

## Evidence and cleanup

Detailed artifacts are under `.marathon/diagnostics/pi-marathon-long-20260921/`.
`long_bench.py` is the runner, `results.json` contains staged checks and backend timings, and `summary.json` contains aggregates.
Each turn retains original and normalized requests, event streams, source checkpoints, and grading results.
The directory also contains the private reference validator, exact shared policy, prompt packets, server logs, and launch commands.
These bulk artifacts stay outside Git.
Generated workspaces are under `/tmp/pi-marathon-long-20260921/`; the reference project is `/tmp/stockroom-reference-20260921/`.

The isolated inference container was stopped and GPU 3 returned to 15 MiB.
The central GPU configuration passed a byte-for-byte unchanged check.
No private production conversations or production application settings were modified.

The practical result is a completion-time advantage for Marathon on this project and a correctness tie at long context.
This run does not support the claim that Marathon necessarily makes this model less accurate, nor that Pi's lighter input guarantees faster project completion.
