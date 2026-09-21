# Bounded Marathon optimization hunt

This is a cooperative queue for two Marathon workers, not an autonomous optimizer or a security sandbox.
Workers investigate and implement candidates; the runner coordinates claims, records results, and serializes benchmark commands.
No production deployment is authorized by this workflow.

## Start here

Work from `/home/deforest/Documents/DEV/Marathon`.
Use `.marathon/venv/bin/python scripts/optimization/queue.py claim` to claim one task.
The command prints a unique owner token, the task ID, an isolated workspace, and the investigation brief.
Keep those values for subsequent commands; do not claim another task until the first has a recorded outcome.
Both workers can use exactly the same prompt below.
Claims are atomic and are not automatically recycled, so a slow or disconnected worker cannot silently lose its task to another worker.
A crashed worker's task remains claimed; report its ID for explicit recovery rather than editing the database or taking it over.

The queue is already initialized at `.marathon/optimization-queue/queue.sqlite`.
Use `.marathon/venv/bin/python scripts/optimization/queue.py status` to inspect tasks and benchmark history.
Re-running `init` inserts missing seed tasks but does not reset claims, outcomes, or the recorded baseline.

## Worker prompt

Copy this entire block to each of the two workers:

```text
Run a bounded quick-win optimization hunt in /home/deforest/Documents/DEV/Marathon.
First read docs/OPTIMIZATION_WORKERS.md completely, then follow its workflow.
Claim one task with .marathon/venv/bin/python scripts/optimization/queue.py claim.
Use the returned owner token and workspace; the other worker has a different task.
Read the prior evidence linked in that document before proposing changes.
Spend at most 30 minutes total and attempt at most two single-mechanism candidates, with no more than one revision per candidate.
Only claim a second task after recording the first task's outcome.
Start from the deployed compact Q8 kernel, not the old pre-staging baseline.
Investigate and build only in your assigned workspace or uniquely named build directories.
Use existing measurements first, and implement only a candidate with a concrete reason it could reduce measured decode work.
Run every GPU command through the benchmark gate described in the document, after applying the manage-local-gpu skill.
Use only a free GPU 3; do not unload, restart, reconfigure, or compete with any active worker.
Check correctness before timing; compare unchanged control and candidate using identical fixed work and reverse-order repetitions.
Treat small or noisy changes as inconclusive, not wins.
Do not deploy anything, alter production, change model weights or quantization, reduce context, change cache formats, change power settings, access private conversations, or send private data to external services.
Jev is optional for short public-code or aggregate-data reviews; do not set up a new integration if it is unavailable.
Save a patch, reproducible command, measurements, and concise conclusion in your workspace.
Record rejected, blocked, or promising with queue.py finish, even when you find no gain.
If GPU 3 is occupied, continue useful CPU analysis briefly, then record blocked if a measured answer cannot fit the time budget.
Finish with the strongest finding and its evidence path, or say no convincing quick win.
Do not spawn additional workers or keep searching beyond the budget.
```

## Reference and exclusions

Read these local reports, prioritizing their conclusions and measurement tables:

- `docs/Q8_STAGING_PROBE_2026-09-20.md`: the compact single-buffer Q8 kernel is now deployed; the separate two-buffer version was rejected.
- `docs/MARATHON_RUNTIME_PROFILE_2026-09-20.md`: existing 4K/75K profile and workload boundaries.
- `docs/TARGET_KERNEL_PROBE_2026-09-20.md`: IQ4 activation double buffering did not establish a gain.
- `docs/GPU_KERNEL_TRIALS_2026-09-08.md`: grid, tile, L1-prefetch, and shared-memory trials already rejected.
- `docs/GPU_WARM_PREFILL_2026-09-16.md`: negative prefill trials; do not mistake prefill counters for decode counters.

These are hypotheses to investigate, not promises that four optimization opportunities exist.
The initial tasks divide IQ4 load/dequantization, residual Q8 attention stalls, copy/gather, and recurrent decode work.
Review shared task results before branching into another task's subsystem.
Do not repeat a rejected mechanism unless new evidence identifies a specific flaw in the prior test, and document that evidence first.
Do not research new target models, KV quantization, power increases, or drafter training in this bounded run.

The baseline image is `sha256:8af4fa77e493f6765b7c66d4f6cbfb673e0add0919b470c11526e08d54365e04`.
It already preloads `/app/libq8-compact.so`.
When preloading an unrelated experimental kernel, preserve this library too; otherwise you silently remove the deployed gain.
For an attention replacement, the unchanged control must contain the deployed compact source, not the original unstaged source.

Each claim copies the qualified CUDA header directory into `workspace/source`.
This is a header/build-input snapshot, not a complete independently buildable llama.cpp checkout.
The existing CPU build recipe is `.marathon/diagnostics/q8-staging-probe-20260920/build.py`, and its compiler entry is `/tmp/marathon-kernels-20260908/compile-entry.json`.
The existing build container is `iq4-build-q8`; never stop it or edit its mounted shared source files.
Adapt the recipe into your workspace with unique `/tmp/marathon-opt-OWNER-*` source and output paths, and cap compiler jobs at one per worker.
Do not execute the old scripts unchanged: they use shared output names, old images, and fixed container names.
Never rebuild the entire backend as an uncontrolled candidate comparison.

Existing synthetic attention fixture source and binary are `.marathon/diagnostics/q8-staging-probe-20260920/attention_probe.cpp` and `attention-probe`.
Existing matrix fixtures and build records are under `.marathon/diagnostics/target-kernel-probe-20260920/`.
Adapt only the necessary fixture into your workspace; do not modify shared harnesses or read their private conversation payloads.
Public synthetic data is sufficient for this initial screening pass.

## Benchmark gate

After reading `/home/deforest/.codex/skills/manage-local-gpu/SKILL.md` and the central GPU-control README and full configuration, run your finite benchmark script like this:

```bash
.marathon/venv/bin/python scripts/optimization/queue.py bench TASK_ID --owner OWNER_TOKEN --timeout 240 --wait 120 -- /home/deforest/Documents/DEV/Marathon/.marathon/venv/bin/python /ABSOLUTE/WORKSPACE/probe.py
```

The command runs in the claimed workspace and receives `OPT_WORKSPACE`, `OPT_IMAGE`, `OPT_GPU=3`, and a unique `OPT_CONTAINER` name.
It also receives `CUDA_VISIBLE_DEVICES=3` for host CUDA processes.
If your script uses Docker, it MUST use `--name "$OPT_CONTAINER"`, `--gpus '"device=3"'`, `--rm`, and the pinned image.
Docker does not inherit host `CUDA_VISIBLE_DEVICES` as an adequate GPU restriction.
Use at most one GPU container at a time and no detached children or background jobs.
Mount only your own writable workspace and required read-only runtime/model inputs.
Keep any serving ports on loopback and use one test port, such as 19979, only while holding the gate.
Do not invoke another GPU lease wrapper from inside the gate; that would attempt to acquire the same lease twice.
The gate holds the shared benchmark lock and Marathon's actual GPU-3 worker lease until cleanup, checks broker conflicts and free GPU memory, detects production-config drift, and limits runtime to at most 600 seconds.
It stops the specifically assigned container on exit.
The gate is cooperative scheduling, not containment against arbitrary shell commands; these restrictions remain mandatory.
No GPU may be used for a build-time test, profiling, warmup, or measurement outside the gate.

Fail closed if a registered service, GPU process, changed configuration, or missing baseline prevents a controlled run.
Do not change locks, conflict checks, pool assignments, or memory thresholds to make the test run.
The worker's own inference GPU and the other worker's inference GPU are off limits to benchmarks.

## Measurement and result contract

A single run may perform control warmup, candidate warmup, numerical comparison, and unprofiled control/candidate/candidate/control repetitions.
Use identical inputs, dimensions, iteration counts, sampling, cache state, and output lengths.
Report raw timings, run-to-run spread, register/shared-memory changes if known, and output comparisons.
Do not time profiler-instrumented runs as performance results.
Start with the existing representative fixed-work shapes; only request a long-context serving test when a numerical pass and repeatable improvement justify it.
For this quick-win screen, changed numerical outputs are a rejection or a request for deeper qualification, not permission to relax accuracy tolerances.
A kernel-only improvement is not a demonstrated Marathon speedup.
A promising result must beat observed run-to-run noise in both order comparisons; tiny ambiguous differences should be recorded as rejected with an inconclusive explanation.
Do not mark a candidate promising based on LLM confidence, source-code appearance, a shorter answer, or speculative-acceptance changes.

Write `result.md` or `result.json` inside your workspace with the hypothesis, source/artifact identities, patch path, exact commands, correctness results, raw timings, conclusion, and limitations.
Then record the task outcome:

```bash
.marathon/venv/bin/python scripts/optimization/queue.py finish TASK_ID --owner OWNER_TOKEN --outcome rejected --summary 'No repeatable gain above timing noise' --evidence /ABSOLUTE/WORKSPACE/result.md
```

Use `promising` instead of `rejected` only when the evidence supports it, or `blocked` when required hardware or artifacts are unavailable.
The database stores benchmark logs and task outcomes but does not automatically judge correctness or promote a candidate.
Keep patches and useful evidence small; models, compiled libraries, and database contents remain ignored under `.marathon`.
Do not delete shared sources, models, or prior experiment artifacts.
Stop after your time/candidate budget and leave promising changes for review.
