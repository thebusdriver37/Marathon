# Development

Use the same checkout as normal users.
Machine-specific overrides belong in `~/.config/marathon/catalog.toml`, not in the shipped catalog.

## Checks

```bash
./bin/marathon setup-deps
.marathon/venv/bin/python -m unittest discover -s tests -v
```

The CI workflow runs shell syntax checks and the Python suite on Python 3.10 and 3.12.
Set `MARATHON_NETWORK_TESTS=1` to also test real pip bootstrap and interrupted-environment recovery; CI enables this check.
Native runtime patching and GPU benchmarks have separate requirements described in [runtime packaging](RUNTIME.md).
Python test success alone does not establish GPU correctness or speed.
Normal frontend installation builds and smoke-tests the binary without requiring developer test runners.
To run the native patch suite as part of a build, install a recent `just` and `cargo-nextest`, then run `MARATHON_CODEX_RUN_TESTS=1 ./bin/marathon build-codex`.
Both runtime patchers preserve upstream sources and existing worktree edits, and reuse unchanged patch stacks.

## Cached frontend builds

`marathon build-codex` reuses a managed source directory for each upstream revision and stores Cargo artifacts in `.marathon/codex-target-cache` by default.
Patch changes update only affected files, preserving unchanged dependencies and their timestamps.
The first managed build adopts the existing patched directory when possible.
Builds are serialized, and patch updates refuse to overwrite staged, unstaged, or untracked user edits.
A new upstream revision gets a separate source directory.
Release optimization and linking still take time after native code changes; a cold build still needs to compile dependencies.
`CARGO_TARGET_DIR` or `MARATHON_CODEX_TARGET_DIR` can override the artifact directory.
Direct calls to `scripts/apply_codex_patches.sh` retain immutable patch worktrees unless `MARATHON_PATCH_UPDATE=1` is set.

## Live user workflows

```bash
./bin/marathon eval smoke --run-gpu
# Also check allocation of all three workers and rejection of a fourth session:
./bin/marathon eval smoke --run-gpu --workers 3
```

This opt-in suite uses the installed frontend and remembered model through a real terminal and real inference.
It checks conversation memory and prompt-cache reuse, file editing and tests, background command cancellation and recovery, the printed resume command, and headless configuration overrides with shell tools.
The cache check requires the configured backend's slot-cache support.
The default uses one worker; the capacity check requires three available workers.
Existing workloads are never evicted.
Transcripts, session logs, router timings, fixture files, and `results.json` remain in the printed temporary directory for inspection.
Use `--output-dir /path/to/new-directory` to choose the evidence location.
The suite closes its sessions; inference workers follow their normal idle policy.
This suite is separate from the normal Python checks and needs no additional Python packages.

Measure compaction separately with the installed frontend and real inference:

```bash
./bin/marathon eval compaction --run-gpu --repeats 2
```

This compares `none`, `low`, and `medium` compaction reasoning while keeping coding reasoning at `medium`.
It checks that facts from tool output survive both the summary and subsequent recall, including repeated compaction.
`--auto-compact-token-limit` additionally requires automatic compaction during setup and checks every automatic summary for retained facts.
The source fixture is moved out of the test workspace before recall, and tool use during recall fails the check.
Router timings, summaries, terminal transcripts, and `results.json` remain in the printed evidence directory.
The small synthetic fixture measures a specific workload, not a universal optimum or near-limit context accuracy.
Use `--help` for sample counts and evidence options.

`MARATHON_COMPACTION_REASONING_EFFORT=low marathon` overrides reasoning only for compaction requests on models that advertise reasoning levels.
Omitting the variable preserves the normal inherited effort; unsupported values are rejected.
The setting takes effect when a new router process starts.
Do not assume `none` is safe merely because its summaries are faster; validate subsequent task continuation and repeated compaction.

Local Responses websocket compaction now reuses the active conversation's tool definitions as prompt text and sets `tool_choice=none`.
The router also rejects tool events during compaction, including managed web tools and recovery attempts.
This retains the reusable prompt prefix while the backend still matches the full input token by token.
Reuse requires the same model, conversation key, and instructions; the router rechecks the live slot after waiting for the backend lock.
Missing or replaced cache state uses the normal fallback path.
The change takes effect when the router next starts; `MARATHON_COMPACTION_PREFIX_CACHE=0 marathon` disables it for rollback or comparison.

For a long-context comparison, run the following with `MARATHON_COMPACTION_PREFIX_CACHE=0` and then `=1`:

```bash
./bin/marathon eval compaction --run-gpu --efforts medium --cycles 1 \
  --filler-lines 6000 --tool-output-max-chars 700000
```

The larger output limit applies only to the evaluator's isolated process and lets it construct a long history in one tool read.
It does not change the normal tool-output limit.

Measure normal terminal startup and verify its first reply with:

```bash
./bin/marathon eval startup --run-gpu --repeats 3
```

This uses the installed frontend, normal worker selection, and existing session-home preparation.
Readiness requires both the terminal's Ready title and its successful Responses connection probe, so the initial loading screen does not count as ready.
Each run then requires an exact, unique reply from a real model turn.
The evaluator saves terminal transcripts, runtime traces, milestone timings, and `results.json` in the printed evidence directory.
It never unloads workers; the first sample uses the worker's existing state and subsequent samples can reuse it.
Only unload a test-owned, unleased worker when measuring GPU-cold startup, and leave active workloads alone.

The September 8 startup comparison used the Qwen 3.8 27B IQ4_XS DFlash2 worker on GPU 2 while other Marathon sessions remained active.
Three warm launches averaged 2.88 seconds before and 1.62 seconds after, about 44% less waiting.
One GPU-cold launch per version measured 12.68 and 11.78 seconds respectively; these did not flush the operating system's file cache.
Model loading still dominates cold startup, and these samples do not establish performance on other machines or after a reboot.
All measured launches passed their first-reply checks.
Warm baseline evidence is in `/tmp/marathon-startup-s6w1vijq`, optimized warm evidence in `/tmp/marathon-startup-_fr3juz2`, and cold evidence in `/tmp/marathon-startup-9px45kr9` and `/tmp/marathon-startup-5drqbbwl`.
These evidence paths are local, not portable repository fixtures.

Startup now skips a redundant `lsof` scan after a successful empty `ss` listing and checks model/router readiness more frequently within the existing timeouts.
The HTTP server framework is imported only when installing router middleware, and the router defers Crawl4AI imports until the first browser request.
The first browser request pays that deferred import cost; ordinary coding sessions avoid it entirely.
Regression coverage retains port-ownership fallback and checks optional browser discovery, disabled or broken dependencies, crawler reuse, and cleanup.
The 316-test Python suite passed with one optional skip, and a real browser rendered a local JavaScript fixture after the deferred import.

Keep first-run setup in the README and advanced configuration in [advanced usage](ADVANCED_USAGE.md).
Document actual tested support separately from expected hardware compatibility.
Preserve existing user selections and never change power caps or stop unrelated inference services during setup.
