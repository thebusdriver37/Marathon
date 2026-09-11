# Compaction evaluation

## Summary validation fix, 2026-09-11

The prompt audit recorded a second-cycle failure where Qwen returned introductory prose followed by a textual tool call instead of a summary.
The previous validator only rejected responses starting with `<tool_call>`, allowing this response to replace useful context and lose all twelve fixture facts.
The router now detects unquoted tool-call markup after prose, while allowing inline-code, fenced-code, and blockquote examples in legitimate summaries.
Compaction responses are buffered until validation succeeds, including on the streaming frontend path.
On invalid markup or a bare acknowledgement, the router retries once using the original context and unchanged reasoning settings, with tools disabled and a direct summary instruction.
The rejected response is never replayed as an accepted assistant turn.
If the retry also fails validation, the router returns an error without publishing the invalid summary.
The production base prompt and `Minimize thinking.` remain unchanged.

Regression tests replay the exact failed response and cover successful recovery, bounded failure without published output, quoted protocol examples, and preservation of context and medium reasoning.
The Python suite passed 393 tests with seven skipped.
All six live compaction cycles across three fresh sessions preserved all twelve facts and recalled them without tools, with medium reasoning and live prefix-cache reuse.
Live terminal verification evidence is retained in `.marathon/diagnostics/compaction-validation-fix-20260911`.
New Marathon router processes pick up the fix; already-running sessions are not restarted automatically.

## Prefix-cache evaluation

Cache-preserving compaction is now enabled by default for the local Responses websocket path.
A live long-context comparison reduced compaction from 188.14 seconds to 11.58 seconds, about 16.25 times faster, with all 12 fixture facts preserved in both the summary and subsequent tool-free recall.

| Long-context measurement | Original cache behavior | Preserve live prompt prefix |
| --- | ---: | ---: |
| Wall time | 188.14 s | 11.58 s |
| Input-processing time | 176.34 s | 0.72 s |
| Generation time | 10.31 s | 10.14 s |
| Cached tokens | 4,635 | 118,759 |
| Newly processed tokens | 111,375 | 103 |
| Generated tokens | 594 | 621 |
| Exact fact retention and recall | 12/12 | 12/12 |

This comparison held coding and compaction reasoning at medium and used the same generated fixture with 6,000 historical-noise lines.
The cache-preserving request contains about 2.9K additional tool-description tokens because it retains the original prompt prefix.
The two conversations were generated independently, so their exact token counts differ slightly.
Generation time was nearly unchanged; the measured improvement came from avoiding redundant input processing.
The 16.25-times wall-time gain applies to this approximately 119K-token synthetic conversation, whose summary was short.
Real conversations with several thousand summary and reasoning tokens will still spend longer generating their handoff.
The earlier estimate of roughly 2 to 3 minutes for the user's recorded workload remains an estimate until that workload is replayed with the new code.

The small-context comparison also passed two successive compactions and recalls.
Its first compaction processed only 103 new tokens in 0.58 seconds, versus 6,745 tokens in 7.49 seconds with the original cache behavior.
Its repeated compaction processed only four new tokens in 0.25 seconds.

The router retains the last response's tool descriptions and parallel-call rendering option in its bounded in-memory response history.
It borrows them only for compaction in the same model and conversation with unchanged base instructions.
It sets `tool_choice=none`, rejects tool events before forwarding or dispatching them, and keeps tool execution disabled during recovery.
The backend receives the complete conversation and performs its normal token-exact prefix matching; the router does not assume that unmatched KV state is valid.
The router rechecks live-slot ownership after acquiring the backend lock and falls back if that slot was lost or replaced.
Cold sessions and HTTP fallback continue to use the original path.
New router processes use this automatically; `MARATHON_COMPACTION_PREFIX_CACHE=0` restores the original behavior.

Long baseline evidence: `/tmp/marathon-compaction-d5j0y0xx`.
Long cached evidence: `/tmp/marathon-compaction-tc60vwpa`.
Small cached evidence: `/tmp/marathon-compaction-eb42fwo2`.
The default-enabled automatic-trigger check in `/tmp/marathon-compaction-sqltiq20` performed one automatic compaction, retained all 12 facts, and passed subsequent manual compaction and tool-free recall.
That automatic compaction reused 10,441 tokens and processed 3,637 new tokens from the pending tool result; its following manual compaction took 6.11 seconds.
This run did not repeat the earlier low-threshold loop, but one successful run does not prove that the model can never repeat work.
The 311-test Python suite completed successfully with one optional test skipped, including cache isolation, lost-slot fallback, slot replacement while waiting, blocked tool dispatch, blocked stream events, and tool-disabled recovery.

The installed Marathon frontend and real Qwen 3.8 27B IQ4_XS DFlash2 worker were exercised through a pseudo-terminal using `/compact`.
The initial controlled comparison used medium coding reasoning, two independent sessions per compaction setting, and two successive compactions per session, with the original cache behavior.
Each session read 12 exact handoff facts from tool output, mixed with historical noise.
The evaluator verified that the router's output truncation policy still exposed every fact, moved the source file outside the workspace, and checked both the stored summary and subsequent JSON recall without tools.
The second cycle also includes the first recall answer in its history, so it checks repeated compaction rather than providing another independent sample.

| Compaction reasoning | First compaction mean | Repeated compaction mean | Combined mean | Summary and recall checks |
| --- | ---: | ---: | ---: | --- |
| medium | 19.18 s | 14.17 s | 16.68 s | 4/4 passed |
| low | 30.33 s | 34.67 s | 32.50 s | 4/4 passed |
| none | 18.11 s | 12.00 s | 15.06 s | 4/4 passed |

All passing checks retained all 12 values and used no tools during recall.
Inputs to the first compaction were approximately 11.2K tokens; subsequent compactions used approximately 8.3K to 9.1K tokens.
The `none` result was about 10% faster on average than medium, but the latency ranges overlap and two independent sessions per setting cannot establish a reliable general improvement.
Changing compaction to low lost the small cached prefix as well: all four low requests reported zero cached tokens, versus 4,635 for medium and none.
The default remains inherited reasoning.
The opt-in `MARATHON_COMPACTION_REASONING_EFFORT` setting allows further testing without lowering coding reasoning.

The user's existing long-session telemetry explains the reported multi-minute delay:

| Recorded compaction | Total | Input processing | Generation | Reprocessed input | Cached input |
| --- | ---: | ---: | ---: | ---: | ---: |
| First | 389.0 s | 253.6 s | 132.1 s | 144,094 tokens | 4,635 tokens |
| Second | 429.0 s | 264.4 s | 161.5 s | 150,598 tokens | 4,635 tokens |

Queue waits were below one millisecond.
Both requests restored only the starter cache.
Compaction constructs a new request with no tool definitions, changing the early prompt compared with ordinary coding turns and preventing reuse of most of the conversation prefix.
Preserving a compatible prompt prefix was the largest optimization opportunity identified in the initial evaluation.
The subsequent implementation retains tool descriptions from the same live conversation while disabling tool calls, allowing the backend to reuse the matching prompt prefix.
Generation also includes internal reasoning, despite the backend reporting zero reasoning tokens in its usage breakdown.
The recorded outputs contained reasoning items with approximately 14.6K and 19.0K characters, respectively.
The second request reached the 8,192-token generation ceiling; its summary should not be assumed complete merely because a compaction event was recorded.
Reaching the ceiling alone does not prove that the summary was truncated.

An exploratory run with reasoning disabled for both coding and compaction exposed a correctness failure.
The first summary contained the facts, but subsequent continuation repeated the original file-reading task, and the second compaction installed raw `<tool_call>` markup as its summary.
The router now rejects this observed malformed summary shape and bare acknowledgements such as `READY` on the Responses websocket path, allowing the frontend to retry or fail instead of accepting them as completed compactions.
This structural check does not establish semantic completeness for arbitrary summaries.
The existing frontend missing-summary validation remains in place.

An additional automatic-trigger stress run lowered the threshold to 13,000 tokens.
It performed six automatic compactions while repeatedly re-reading the fixture; the fifth saved only `READY` as its summary.
The session eventually recovered by reading the still-available file again and passed final recall, which would conceal the intermediate loss if only the last summary were checked.
The evaluator now inspects every automatic summary, and the router rejects the observed bare acknowledgement.
This artificially tight threshold is far below the normal configuration and does not establish that the same loop occurs at the production threshold.
It does show that shortening individual compactions by triggering them much earlier can increase total work.
The original stress evidence is in `/tmp/marathon-compaction-pnltdujr`.
The guarded rerun in `/tmp/marathon-compaction-ucmggxhg` completed with five automatic compactions followed by one manual compaction.
Every automatic summary retained all 12 facts, and final recall passed without tools.
The repeated-work behavior at this low threshold remains; rejecting obvious invalid summaries does not fix that behavior or guarantee semantic accuracy.

The initial fixture also exposed a separate confound: facts in the middle of a large tool output can be trimmed before compaction.
Those exploratory results were excluded from the controlled table.
The reusable evaluator appends a focused extraction of the facts and checks the actual bounding policy before measuring compaction.

Reproduce the controlled comparison with:

```bash
MARATHON_COMPACTION_PREFIX_CACHE=0 ./bin/marathon eval compaction --run-gpu --repeats 2
```

The evidence from this run is in `/tmp/marathon-compaction-e4hkrnkp`, including `results.json`, summaries, recall answers, terminal transcripts, session rollouts, and router timings.
The malformed-summary reproduction is in `/tmp/marathon-compaction-1raqaj3w/none-0/summary-1.json`.
Long-session timings came from `~/.local/state/marathon/runs/20260907-214437_qwen3.8-27b-iq4-xs_bdd99388ff38.jsonl`.
These are local evidence paths, not portable repository fixtures.

Regression checks cover override scoping, malformed-summary rejection, acknowledgement rejection, and valid summaries in languages without spaces.
The initial Python suite completed successfully: 305 tests, with one optional test skipped.
The controlled fixture does not establish near-limit semantic accuracy, performance on other models, or a universal Pareto optimum.
Broader evaluation should include contradictory decisions, unfinished multi-step work, long tool histories, cancellation, and resumed sessions before changing the default.
