# Real Marathon failure probe

## Finding and implementation

The initial real-CLI graph sessions reproduced the same Marathon patch-compilation defect with both the baseline and Swift merge.
The router advertises structured `replace` operations and explicitly encourages several small replacements.
Its compiler emitted a separate `*** Update File` section for each replacement, including multiple replacements targeting the same file.
The hardened Codex frontend rejects multiple file sections resolving to the same path with `multiple operations target ...`.
Both models recovered with additional tool calls, but the rejection was avoidable application overhead rather than evidence that those replacements required model fine-tuning.

`scripts/routers/codex_local_router.py::_structured_patch_to_input` now groups replacements into one file section with multiple hunks.
It preserves per-file replacement order and first-seen file order, including interleaved operations on different files.
Conflicting add/delete/replace combinations for a single exact path are rejected before emitting a patch.
The native frontend's path checks and duplicate-target rejection remain intact.
No model-specific instructions, model weights, or production GPU configuration were changed.
This is a Python change and requires no frontend rebuild; new Marathon router processes load it.

Three regression tests were added for repeated replacements, interleaved files, and conflicting actions.
They failed against the old compiler and pass with the fix.
The router context, security, and accounting suites pass all 139 tests with `python -m unittest discover -s tests -p 'test_router*.py'` using the Marathon virtual environment.

## Actual execution path

The reusable runner is `scripts/evals/marathon_failure_probe.py`.
It launches the installed `/home/deforest/.local/bin/marathon --instance ... exec` command and does not implement a substitute agent loop, tool executor, or system prompt.
Session metadata records `codex_exec`, frontend version `0.153.4`, full base instructions, and actual tool calls/results.
Task prompts, commands, router telemetry, session transcripts, code changes, independent test results, and backend configuration are retained under `/home/deforest/AI/experiments/marathon-failure-probe-20260919/`.
This is diagnostic evidence, not a certified training dataset or a claim to capture every final backend wire payload.

Both models used IQ4_XS, the production-pinned inference image, DFlash2/6 with the installed V1 drafter, 196K context, medium reasoning, and temperature 1.0.
Independent repetitions were not seed-pinned.
The baseline used the production broker's GPU 3 worker, while the merge used an isolated broker on GPU 2 with the central configuration copied and its model/projector paths substituted.
The candidate retained the baseline routing alias and profile; its protocol file records the actual candidate backend command.
Neither model was promoted or fine-tuned.

Administrative differences were isolated session homes, experiment catalog and log paths, and a workspace-write sandbox.
No custom model-instruction override was supplied.
The first graph run had no staged Git baseline, so its `git diff --check` is not useful proof of whitespace cleanliness.
Original test-file hashes and independent behavioral checks are still valid, and focused follow-ups stage starter files before execution.

## Initial two-task screen

The graph fixture came from `/home/deforest/AI/experiments/swift-uncensored/marathon-validation/templates/graph`.
The prompt explicitly added independent cycle rejection for direct `criticalPath()` calls and non-strict skipping of semantically invalid rows.
The strict-mode requirement for semantic errors to include a field name remained a hidden contract extension, so its failure should not be treated as an unambiguous instruction-following defect.

| Initial task | Baseline | Swift merge |
| --- | --- | --- |
| Graph project | Completed; 8/8 visible tests, 2/3 hidden tests | Five-minute timeout; resulting code passed 8/8 visible and 3/3 hidden tests |
| Direct cycle rejection | Passed | Passed in post-timeout snapshot |
| Fixture audit | Correct 48 accepted / 1 duplicate | Correct 48 accepted / 1 duplicate in saved audit |
| Counting with execution | Correct 4,738 | Correct 4,738 |

The baseline graph miss was strict parsing silently skipping an invalid duration instead of raising a source/line/field error.
The merge's timeout is an evaluation limit, not a demonstrated model defect or a completed full-session success.
Its generated CLI still printed 49 services while its audit correctly reported 48, a cross-file integration discrepancy worth revisiting without the cutoff.
The merge also spent effort trying shell commands named `search` and `browse`; do not infer a router web-tool failure solely from those attempts.

The counting oracle was independently verified by exhaustive enumeration and subset dynamic programming before grading.
Both models created executable verification and reported 4,738, so the earlier standalone wrong answer was not reproduced with Marathon tools in this screen.
The baseline additionally made an unsupported claim that its exponential subset DP scales to inputs in the low hundreds; this unscored explanatory error is retained for review but was not independently reproduced across sessions.

## Focused follow-up design

Two fresh sessions per model request two replacements in one `apply_patch` call and verify the exact resulting file.
All four patch sessions passed with one native file section and two hunks, without duplicate-target rejections.

Two further sessions per model start from the baseline's failing parser implementation and explicitly specify the behavior of both parsing modes, including semantic-error source, line, and field reporting.
They are focused repair probes, not identical replays of the longer graph task.
Results are retained separately under `focused-after-fix/`.
Success on those probes cannot be attributed solely to the router fix, because the task scope and contract also changed.

| Focused probe | Baseline | Swift merge |
| --- | --- | --- |
| Two-replacement patch | 2/2 sessions completed with the exact intended file | 2/2 sessions completed with the exact intended file |
| Explicit parsing contract | 2/2 sessions completed and passed all three held-out checks | Both code snapshots passed all three checks; one session completed and one hit the 180-second limit |

No duplicate-target patch rejection appeared in any of the eight focused sessions.
One baseline parser session had a separate old-text/context mismatch and recovered; the compiler fix does not guarantee every model-proposed edit matches existing source.
The merge's second parser run had reached passing tests but continued verification and was interrupted at the limit.
Completion efficiency is therefore still worth investigating, but these limits do not establish an inherent model defect or a reliable speed ranking.

The complete probe comprised 12 real CLI sessions, with two capped merge sessions clearly distinguished from completed successes.
All experiment workers were released, the production broker configuration remained unchanged, and roughly 4.6 GiB of private experiment backend snapshots were removed after shutdown.
Approximately 79 MiB of local evidence remains in the experiment directory; no training tensors or new model weights were generated.

## Decision

The confirmed repeatable defect belongs in Marathon's Python patch compiler and has been fixed there.
This screen does not justify fine-tuning or promoting the merged model.
Retain unresolved graph and explanatory issues as candidates, clarify their contracts, and reproduce them before collecting training examples.
Do not equate the absence of repeated failures in a small focused probe with broad model reliability.
