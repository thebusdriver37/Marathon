# Swift merge: runtime versus repeated user-prompt thinking instruction

Moving `Minimize thinking.` from Marathon's runtime instructions to the end of every user turn did not produce a useful speed or correctness advantage in this targeted comparison.
Both placements passed all 120 cumulative check executions across two challenging two-turn projects.
Per-prompt placement produced 9.1% fewer reasoning words overall, but 1.2% more generated tokens and essentially equal completion time.
Reasoning volume increased on one of the four paired turns, so the reduction was not consistent.
This experiment compares placement; it does not test removing the phrase entirely.

## Results

| Metric | A: runtime instruction | B: every user prompt |
| --- | ---: | ---: |
| Cumulative check executions passed | 120/120 | 120/120 |
| Final checks: ledger / scheduler | 11/11 and 54/54 | 11/11 and 54/54 |
| Completion time excluding model loading | 697.38 s | 703.97 s |
| Reasoning words | 8717 | 7924 |
| Reasoning characters | 54943 | 47652 |
| Generated tokens, including reasoning and code | 43031 | 43558 |
| Newly processed input tokens | 80748 | 80638 |
| Inference requests | 66 | 91 |

B used 13.3% fewer reasoning characters and 9.1% fewer whitespace-separated reasoning words.
These are measurements of emitted reasoning text, not model-token counts, invisible computation, or reasoning quality.
Total generated tokens include reasoning, tool arguments, code, and visible answers.
The approximately seven-second aggregate time difference, 0.95%, should be treated as a tie in this small screen.
More concise reasoning did not translate into less total generation or fewer inference calls.

| Paired turn | Checks, both | A reasoning words | B reasoning words | A seconds | B seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| Ledger: reversals | 9/9 | 2030 | 1788 | 169.34 | 160.14 |
| Ledger: atomic batches and CSV | 11/11 | 1455 | 2206 | 151.25 | 204.70 |
| Exact scheduler | 46/46 | 3835 | 2930 | 244.01 | 219.84 |
| Scheduler: blackout intervals | 54/54 | 1397 | 1000 | 132.77 | 119.29 |

The ledger project took 320.59 seconds under A and 364.84 under B.
The scheduler took 376.78 seconds under A and 339.13 under B.
Thus, project-level timing effects went in opposite directions and largely canceled in aggregate.
Passing all checks is evidence for those cases, not proof that either implementation has no other bugs.
The 120 executions include repeated earlier contracts after follow-ups, not 120 independent model trials.

## Exact comparison

A retained the existing single `Minimize thinking.` sentence in the backend `instructions` field, supplied on each inference request through Marathon's normal runtime policy.
A user tasks had no added sentence.

B removed that sentence from the outgoing `instructions` field and appended the exact sentence once to each user task, including follow-ups.
Earlier user turns remained in conversation history normally.
B did not retain a runtime copy, so this was a relocation experiment rather than doubling the instruction.
Host metadata, tool descriptions, tool results, and assistant messages were not modified.
“Every prompt” here means every actual user turn, not every tool continuation or every text block in an API request.

The private adapter made the request transformation only for the benchmark.
The production router source was not edited.
All 157 captured backend requests were checked for the expected runtime phrase count, and the paired user task files were verified identical after removing B's appended sentence.
All requests returned HTTP 200 with completed status.
No response reached the 6144-token output limit; the largest generated response contained 3900 tokens.

## Tasks and quality checks

The fixture is `scripts/evals/phrase_placement_tasks.py`.
Neither model run saw the independent graders or their outcomes between turns.
Both could inspect project code and create and run their own tests.

The ledger started from the same existing SQLite project implementing receipts, shipments, persistent idempotency, reservations, and transfers.
The first turn added reversals with reservation-aware availability checks, replay behavior, persistence, and atomic failure semantics.
The follow-up added all-or-nothing batches and CSV imports, including retries after rollback and preservation of prior behavior.
Its seed is the retained source checkpoint from the earlier long-context benchmark's transfer stage, frozen by `seed-manifest.json` in the evidence directory.

The scheduler started from a stub and required exact optimization of optional jobs on one machine with releases, deadlines, prerequisite chains, and potentially negative-profit prerequisites.
Its ordered objective was maximum profit, then earliest finish, then lexicographically smallest execution order.
The follow-up introduced overlapping, nested, unsorted blackout intervals with nonpreemptive execution and endpoint rules.
Checks covered invalid graphs and fields, input immutability, greedy counterexamples, 32 deterministic generated cases per stage, and blackout edge cases.
The hidden oracle exhaustively enumerated feasible schedules.
Before model runs, it was checked against a separate permutation-based implementation plus a hand-calculated prerequisite example.
The ledger grader also passed against a separate reference implementation.

## Runtime controls and limits

Both used the deployed Swift/Qwen 3.8 27B uncensored merged IQ4_XS model, the same tuned DFlash2 drafter, Q8 target cache, and production backend image.
The worker configuration came from the central GPU-control source of truth.
GPU 3 remained at its existing 250 W power limit, with one benchmark stream protected by the shared lease.
The existing production worker on GPU 1 was left running.

Both variants used Marathon, medium reasoning, temperature zero, seed 8123, top-p 1, top-k 0, min-p 0, repeat penalty 1, and a 6144-token output cap per request.
The existing native 2048-token post-tool thinking cap was retained identically; initial user requests had no such cap.
Consequently, this measures phrase placement within Marathon's current reasoning policy, not unrestricted reasoning.
The native cap may constrain how strongly a wording change can express itself.

Each two-turn session used a fresh workspace and cold backend; follow-ups resumed their own conversation and modified project.
Order was ledger A then B, scheduler B then A.
There was one run per project and placement, with no replicated seeds or statistical significance test.
Tool use and downstream conversation histories naturally diverged after the initial placement change.
Actual maximum request context per turn ranged from about 27K to 45K tokens; this was a targeted reasoning comparison, not another 100K-context stress test.
No private production conversations were used.

## Evidence and cleanup

The experiment started September 21 and completed September 22, 2026, local time.
Artifacts are retained under `.marathon/diagnostics/swift-phrase-placement-20260921/`.
They include the runner, exact prompts, original and transformed requests, response streams, backend timings, source checkpoints, grader outcomes, reference validator, summary, and placement audit.
Reasoning volume was counted from `response.reasoning_text.delta` events in each recorded response stream, once per generated response rather than repeatedly from replayed history.
Raw reasoning is retained privately in the diagnostic evidence and is not reproduced in this report.
Generated projects are under `/tmp/swift-phrase-placement-20260921/`.
Bulk evidence and model files remain outside Git.

The benchmark container was stopped and GPU 3 returned to 15 MiB.
The central GPU configuration and production router both passed byte-for-byte unchanged checks.
The production `Minimize thinking.` clause is still present.

These results do not justify adding the phrase to every user prompt for speed or quality.
They also do not establish whether the Swift merge benefits from the phrase at all; answering that requires an otherwise matched no-phrase control.
