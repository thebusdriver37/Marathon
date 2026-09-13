# Marathon coding-session profile

Date: 2026-09-11.

## Result

All eight coding sessions passed the independent functional and protected-file checks.
Across 674 seconds of session time, generation accounted for 80.3%, prompt processing for 10.6%, and tool execution/delivery for 2.2%.
The remainder includes startup, shutdown, slot preparation, transport, and frontend overhead.
The useful next experiments concern avoiding corrective work and understanding suffix reprocessing, not making shell commands faster.
No measured optimization or general 10% gain is claimed by this profile.

| Task | Repeat | Elapsed seconds | Generation seconds | Prefill seconds | Tool seconds |
|---|---:|---:|---:|---:|---:|
| Ledger | 0 | 98.15 | 78.25 | 11.96 | 1.56 |
| Ledger | 1 | 147.64 | 125.70 | 11.76 | 2.81 |
| Recovery | 0 | 47.23 | 24.74 | 7.84 | 1.84 |
| Recovery | 1 | 27.02 | 16.05 | 6.79 | 1.30 |
| Retry | 0 | 57.61 | 46.65 | 6.65 | 1.43 |
| Retry | 1 | 74.30 | 61.65 | 7.81 | 1.71 |
| TTL/LRU cache | 0 | 75.40 | 62.29 | 7.71 | 2.22 |
| TTL/LRU cache | 1 | 146.32 | 125.43 | 10.91 | 2.20 |

The batch contained 73 timed inference requests, 65 completed tool calls, and 43,660 generated tokens.
All eight completed trials used GPU 3 at its unchanged 250 W limit.
The first recovery trial included approximately 9.75 seconds of cold model loading.
Aggregate slot preparation took 4.20 seconds, slot saving about 0.022 seconds, and inference queue wait less than 0.001 seconds.

## Scope

Real Marathon headless sessions executed four small coding tasks twice: retry logic, TTL/LRU cache, transactional ledger, and recovery from an unavailable test runner.
Each task has independent functional checks and protected files.
The target model, inference settings, medium reasoning, and existing baseline instructions were retained.
Normal slot snapshots were enabled.
Each trial used an isolated workspace, home, and private temporary directory.
The evaluator and analysis are retained under `.marathon/diagnostics/session-profile-20260911/`.

This is a diagnostic profile, not a controlled comparison of a proposed optimization.
These small repositories do not represent large codebases, network tools, builds, or day-long sessions.
Elapsed time includes launcher startup and shutdown; prompt and generation timings come from the backend, and tool intervals come from the Codex rollout.
Tool intervals include dispatch and delivery, not only subprocess execution.
Request timings contain prompt processing, generation, transport, and slot preparation, so those columns must not be added together.

## Findings from transcript review

### Corrective work is a concrete source of delay

The first recovery trial emitted a patch with missing indentation.
The patch applied, but the next test failed to import the module.
The agent reread the file and repaired the indentation, and independent final checks passed.
The added read, repair generation, and failed-test round trip are genuine rework.
The faster second recovery trial also benefited from an already loaded worker, so the full elapsed-time difference cannot be attributed to the patch mistake.

The slower ledger trial chained a Git-history query before the requested test using `&&`.
The fixture repository had no commits, so the Git command failed and prevented the test from running.
The agent then had to invoke the test separately.
Both ledger runs also corrected an erroneous expectation in a generated test concerning mutation of an input event and duplicate-ID conflict handling.
The independent evaluator passed the final implementations.

The slower cache run generated 11,164 tokens versus 5,104 in the other repeat.
After its sole implementation patch, it twice rewrote extended checks with inconsistent cache-capacity/eviction expectations.
The implementation needed no additional patch and passed the independent final oracle.
This is a specific example of test-construction rework, not a reason to skip boundary-condition tests.

These observations favor testing narrowly targeted improvements to edit reliability, command exit handling, and test construction.
They do not establish that additional prompt instructions will help; earlier prompt experiments did not find a general improvement.
Tests should not be removed indiscriminately, since they also caught the indentation defect.

### Cache reuse works, with occasional suffix reprocessing

Completed traces restore the shared 6,861-token starter prefix and then reuse the live parent across tool turns.
The first ledger run nevertheless reprocessed 3,741 tokens after a long generated tool command, despite being labeled `reuse-live-parent`.
This cost about 4.5 seconds of prompt processing on that request.
It is evidence of suffix reprocessing, not proof that the whole cache was lost or that a router bug caused it.
Serialized tool-call differences and recurrent checkpoint rewind are possible explanations requiring a dedicated reproduction.
The same behavior is not yet established as a broad 10% opportunity.

### Faster shell execution is not the main lever here

Tool execution and delivery consumed only a few seconds per completed session.
Queue wait was negligible in the successfully acquired sessions.
Most elapsed time was spent generating tokens, including code, test scripts, explanations, and corrective work.
Generation time must not be equated with unnecessary reasoning.

## Operational notes

The batch was continued after interruptions, retaining completed trials and preserving incomplete attempts separately through the evaluator's continuation mechanism.
An initial separate compaction probe could not acquire a worker and exited before inference.
It was queued behind the coding batch rather than interrupting another session.
No production source or inference configuration was changed for this profile.

## Compaction probe

The queued probe completed two manual compaction cycles with coding and compaction effort fixed at medium.
Both summaries preserved all 12 fixture facts, and both subsequent recall answers recovered all 12 without tools.
The source fact file was removed from the project before recall by moving it into retained evidence.
Compaction took 10.474 and 8.379 seconds, respectively.
Both requests used `reuse-live-compaction-prefix`, reporting 14,342 cached tokens of 14,444 input tokens and 11,513 of 11,615.
This small recall test does not establish long-running task continuity in general, but it exposed no compaction-related rework.

## Recommendation

Keep production settings unchanged.
The highest-signal follow-up is a controlled experiment on reliable, reusable verification: preserve test scripts when they will be rerun, use clear assertions, and avoid regenerating large scripts to correct a small expectation.
Also reproduce the long tool-call suffix reprocessing before attributing it to a cache defect or attempting a fix.
Edit indentation handling and shell command-chain reliability are specific additional candidates.
Measure changes by completed-task time and independent correctness, keeping the current model and reasoning settings fixed.
The observed run-to-run differences are opportunities to investigate, not estimates of guaranteed savings.
