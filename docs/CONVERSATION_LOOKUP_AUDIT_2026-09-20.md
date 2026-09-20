# Ordinary conversation lookup comparison and remaining speed leads

The six-versus-127 lookup comparison found no material ordinary-conversation throughput change in this synthetic screen.
All four executions of each prompt returned identical message hashes, generated-token counts, proposed-token counts, and accepted-token counts.
Production configuration and power settings were unchanged, private production conversations and snapshots were not read, and the isolated test worker was removed afterward.

## Method

An exclusively leased, initially unloaded regular RTX 3090 on GPU 3 ran at its existing 250 W cap.
The other two Marathon workers continued their live workloads.
Four separate diagnostic containers ran startup lookup lengths 6, 127, 127, and 6, sequentially.
Both settings used the currently deployed candidate runtime, original six-token neural draft, same Swift uncensored IQ4_XS target, Q8_0 target cache, and 196000 configured context capacity.
This isolates the lookup setting in the current runtime, rather than comparing two runtime binaries.
Each container used a fresh synthetic slot directory and warmed each prompt for 32 tokens before the measured request.
The three prompts cover explanation, personal project planning, and fiction, at temperature 0.7, seed 8123, and medium reasoning.
Each measured output was capped at 768 tokens and ended at that cap.
These are decode-throughput checks, not naturally completed task-quality evaluations.
Actual prompt lengths were 47, 54, and 51 tokens, so this screen does not establish behavior at long occupied context or during a large project.
The source and production-configuration hashes, exact commands, synthetic outputs, counters, and GPU telemetry are saved with each run.

## Results

| Prompt | Lookup 6 mean tokens/s | Lookup 127 mean tokens/s | Change |
|---|---:|---:|---:|
| explanation | 64.84 | 65.22 | +0.59% |
| planning | 55.76 | 56.00 | +0.42% |
| fiction | 60.58 | 60.51 | -0.12% |

There is no demonstrated speed gain or meaningful regression in these measurements.
The sub-percent differences are smaller than observed repeat variation.
Only two measurements per setting per prompt were taken; no statistical equivalence claim is made.
No tokens beyond the sixth proposal were accepted on these prompts.
Keep lookup127 for its previously validated copy/edit benefit, without promising faster novel conversation.

## Remaining leads and Jev review

Four Jev calls produced 112 typed advisory judgments across an initial screen/challenge and a final screen/challenge with the completed measurements.
Only authored aggregate evidence and source findings were sent, with no conversation text.
Total reported usage was 19636 input tokens and 2810 output tokens.
Jev favored analyzing verification and feature-processing cost, and advised against repeating confidence/width sweeps, Q4 cache plus wider drafts, or the exhausted training objectives.
Its judgments are prioritization advice, not measurements or independent votes.
The final review still marked the lookup check as validate despite receiving its completed results; that output is not a reason to repeat a completed check without a new question.

The most concrete unmeasured area is `common_speculative_process` and the DFlash implementation's `process` method.
The existing begin/draft/accept timers do not include a separate timer for this method.
For short verification batches, the inspected source gathers target features into host arrays, executes a separate encoder graph, copies the encoded output into the injection batch, and decodes that batch into draft state.
The embedding getter APIs explicitly synchronize their contexts.
This establishes a path to measure, not proof that the copies or synchronization dominate runtime or can simply be removed.
The source already selects fused injection for larger batches and says separate cached graphs are faster for small speculative batches, so blindly enabling fusion would disregard existing work.
The next bounded investigation should attribute per-round time to target verification, feature processing, draft generation, and checkpoint/rollback work before changing execution.
Any resulting optimization must preserve target verification, recurrent-state correctness, media behavior, and the existing context and precision constraints.

Seven-token neural drafting remains an unresolved execution failure, but is a weaker ordinary-prose lead than it initially appeared.
In these fixtures, sixth-position acceptance was only 16, 13, and 21 times per 768 output tokens.
Under the simplifying assumption that a seventh token only appends to unchanged accepted prefixes at unchanged round cost, even perfect seventh-position acceptance would add only about 2.1%, 1.7%, and 2.7% useful tokens on those trajectories.
That is a conditional opportunity estimate, not a universal bound: changing the neural block may alter earlier predictions and round cost.
The current trained block size is eight, and standard DFlash is clamped to at most seven draft proposals.
Fixing the seven-token failure cannot be assumed to unlock arbitrary neural block lengths.

Historical 250-to-275 W results suggest roughly 6.5% extra throughput on the older regular-3090 setup, with an efficiency tradeoff and no matched current-merge measurement.
Power was left unchanged per the user's preference.
Further drafter or target training needs a materially different, evidence-backed mechanism; earlier objectives and sweeps are substantially exhausted.
Reducing answer length can reduce task completion time but is distinct from increasing decode tokens per second.
No unclaimed, already-proven 10-20% ordinary-conversation software gain was found in the reviewed evidence.

## Reproduction

Use `scripts/evals/speculation_cost_probe.py --gpu 3 --mode mixed-copy --lookup-width 6 --conversation-screen --output <new-output-directory>`, followed by 127, 127, and 6 with separate output directories.
The harness refuses to use an already loaded or leased worker and changes only its diagnostic container.
Artifacts are in `.marathon/diagnostics/conversation-lookup-20260920`.
The initial protocol metadata was corrected afterward to record the actual hardcoded conversation temperature and three-case order; measured responses were not modified.
The test worker was removed, GPU 3 returned to 15 MiB allocated, and both live Marathon workers were left running.
