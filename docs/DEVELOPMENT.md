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

## Releases

Set `marathon_app.__version__` to the release version, commit it, and push the matching `vX.Y.Z` tag.
The release workflow tests Marathon, builds the patched Codex frontend on Linux x86_64, packages both together, and publishes the archive with its SHA-256 checksum.
Do not create the tag until the ordinary CI workflow is green.
Native runtime patching and GPU benchmarks have separate requirements described in [runtime packaging](RUNTIME.md).
Python test success alone does not establish GPU correctness or speed.
Normal frontend installation builds and smoke-tests the binary without requiring developer test runners.
To run the native patch suite as part of a build, install a recent `just` and `cargo-nextest`, then run `MARATHON_CODEX_RUN_TESTS=1 ./bin/marathon build-codex`.
Both runtime patchers preserve upstream sources and existing worktree edits, and reuse unchanged patch stacks.

## Cached frontend builds

The Codex baseline is pinned by both `config/codex.ref` and the `codex` submodule.
The current baseline is `rust-v0.162.1` (`092d3acd6bec3e3a14bdc7e7a2810ab628ab759d`).
Its patch rebase preserves per-step model usage accounting, validated compaction
summaries with upstream compaction metadata, provider-filtered session lookup,
and local-only routing through the asynchronous HTTP client builder. The terminal
uses upstream's current layout with Marathon branding and performance indicators.
Protocol export bundles and the generated Python response types must be regenerated
when Marathon's response metrics schema changes. Native patch tests use synthetic
fixtures and temporary homes, not personal conversation history.

Marathon resumes each conversation with its persisted model and reasoning effort.
The launcher sets `MARATHON_RESTORE_THREAD_MODEL=1` so its startup defaults do not
override saved settings in the TUI or `exec resume`. Explicit model, reasoning,
provider, or profile arguments disable restoration for that launch. New conversations
still use the selected startup defaults. The native behavior is covered by
`MARATHON_TEST_CODEX_BIN=/path/to/codex .marathon/venv/bin/python3 -m unittest discover -s tests -p test_conversation_model.py -v`,
which uses a local scripted Responses server and isolated conversation history.

Marathon builds bake in a loopback-only frontend transport boundary. Cloud login,
shared Codex daemons, remote-control attachment, feedback and telemetry are disabled;
unsetting `MARATHON_LOCAL_ONLY` cannot re-enable them in shipped binaries. The launcher
also strips inherited cloud credentials and remote-executor settings. The local
router still reaches explicitly configured inference backends, including LAN servers.
This frontend boundary is not an OS firewall for shell tools or MCP subprocesses:
their network access follows their sandbox policy.

Before promoting an upstream update, run `scripts/test_local_security.py` against
the candidate binary using Marathon's Python environment. It traces startup, a
synthetic tool turn and resume inside an offline network namespace, rejecting any
non-loopback connection attempt. It uses fixture configuration and a temporary home,
never personal conversations or account files.

`marathon build-codex` reuses a managed source directory for each upstream revision and stores Cargo artifacts in `.marathon/codex-target-cache` by default.
Patch changes update only affected files, preserving unchanged dependencies and their timestamps.
The first managed build adopts the existing patched directory when possible.
Builds are serialized, and patch updates refuse to overwrite staged, unstaged, or untracked user edits.
A new upstream revision gets a separate source directory.
Release optimization and linking still take time after native code changes; a cold build still needs to compile dependencies.
`CARGO_TARGET_DIR` or `MARATHON_CODEX_TARGET_DIR` can override the artifact directory.
Direct calls to `scripts/apply_codex_patches.sh` retain immutable patch worktrees unless `MARATHON_PATCH_UPDATE=1` is set.

## Live user workflows

### Repeated task-completion gate

```bash
./bin/marathon eval reliability --run-gpu \
  --model MODEL_ALIAS_OR_GGUF_BASENAME --profile PROFILE_ID --repeats 3
```

This suite uses the installed Marathon frontend, normal router, and real inference.
Specify the model and profile explicitly so a changed remembered selection cannot
silently change a comparison. It covers code repair, error recovery, respecting
scope, reporting remaining test failures, finding authoritative requirements,
atomic CSV reporting, user corrections, compaction, resume, cancelled background work,
and a six-turn inventory project with cumulative checks after every feature.
`--cases` selects a smaller set. `--timeout` bounds each turn.

Each run creates a fresh synthetic project and isolated Codex/Marathon homes,
instructions, skills, and history. Native filesystem permissions expose only the
fixture, OS dependencies, and the sandbox binary to model tools. Network and
delegation are disabled. Hidden checkers run outside the model's accessible
filesystem in a separate bubblewrap sandbox, using disposable copies so grading
cannot modify the agent's project. Personal conversations and private
project files are not inputs. The existing global GPU lease mechanism remains
active; the suite never evicts another workload. Linux, bubblewrap, the installed
hardened frontend, and an available model worker are required.

Results retain failures, timeouts, infrastructure errors, protected-file changes,
and failures at earlier project stages. A later repair cannot erase an earlier
failed stage. A model's assertion that tests passed does not satisfy the hidden
checks. Scripted corrections and cancellations are part of the task; there are no
adaptive rescue prompts or manual interventions. The success rate is therefore
the fraction of complete scenarios passing without operator intervention.
Failed scenarios stay unresolved; the suite does not guess how many additional
corrections would recover them.

The command defaults to a 100% success gate and exits nonzero below
`--min-success-rate`. Raw terminal/session evidence, per-stage grades, and
`results.json` stay in a new temporary directory outside the checkout. Pass
`--output-dir` to choose a new evidence directory. The report records the model
file metadata, frontend hash, source hashes, repetitions, and turn limit. Disk
starter caches and conversation snapshots are disabled for isolation; live cache
reuse still operates within each conversation. Timing starts after the frontend
is ready and includes inference loading if the worker is cold.

For an interleaved baseline/candidate experiment without changing the default
runtime, supply `--router-source /path/to/baseline.py` and
`--candidate-router /path/to/candidate.py`. Each scenario runs through both arms;
the order reverses on alternate repetitions. `comparison.json` reports every
matched pass-to-fail regression, gains, and the median paired time difference
only for runs where both arms passed. Any matched regression fails the gate.
`--instructions-file` supports a standalone prompt experiment. Alternatively,
`--compare /path/to/baseline/results.json` compares separate runs, requiring the
same evaluator, scenarios, repetitions, model/profile, and timeout.
Explicit router and prompt inputs are copied into the evidence directory before
inference. Incompatible or incomplete comparison reports are rejected before
allocating a worker.

Use this gate before promoting model, prompt, or runtime changes. Ordinary hosted
CI tests the evaluator's grading and comparison logic; it does not run GPU
inference. These fixtures provide a bounded regression check, not a general
coding-ability score or proof of reliability at the model's context limit.

#### October 18 evaluation

The evaluation used Swift Qwen3.8 27B Uncensored Merge IQ4_XS with the existing
DFlash2 pool profile, 196K context, medium reasoning, and temperature 1.0. The
paired trials used the same registered GPU worker. Each of eight scenarios ran
twice per arm, reversing arm order on the second repetition, with a 240-second
turn deadline. No model, broker, or production prompt settings were changed.

The candidate appended guidance to verify behavior and boundary cases, resolve
failed checks, inspect output as well as exit status, and finish without repeating
successful checks unnecessarily. It was evaluated through an isolated router
source override. It is not promoted: it introduced a matched correctness
regression and has not established a reliability improvement.

| Scenario | Baseline | Candidate |
| --- | ---: | ---: |
| TTL/LRU cache repair | 2/2 | 2/2 |
| Report an unrelated failing test honestly | 2/2 | 2/2 |
| Find and implement the authoritative specification | 1/2 | 0/2 |
| Atomic CSV reporting | 1/2 | 2/2 |
| Apply a corrected requirement | 2/2 | 2/2 |
| Preserve requirements through compaction | 2/2 | 2/2 |
| Preserve corrections through resume | 2/2 | 2/2 |
| Cancel background work and finish a new task | 2/2 | 2/2 |
| **Unassisted completion** | **14/16 (87.5%)** | **14/16 (87.5%)** |

There were no timeouts or infrastructure errors in these 32 paired runs. One
matched pass became a failure and one failure became a pass. Across the 13 pairs
where both arms passed, the candidate was 1.1 seconds slower at the median; this
small sample does not establish a speed benefit. The comparison gate exited
nonzero. Rechecking all 32 generated projects with the final isolated,
non-mutating checker reproduced every score.

The exposed failures concern independent correctness, despite successful
model-written tests: Unicode combining marks were converted into separators
instead of removed, and one baseline CSV implementation accepted a delimiter-only
row despite the explicit requirement to validate its cells. The baseline also
produced the Unicode error in one repetition, so it is an existing weakness the
candidate did not solve.

The separate six-turn inventory project used a 300-second turn deadline. Its
first four feature stages passed all cumulative checks, including the stage
after native compaction. Stage five exceeded the deadline before completing its
reply, even though its code already passed all 11 independent checks at that
point. That run remains an unassisted timeout. In a separately recorded recovery,
one generic continuation prompt completed stage five; the original final feature
prompt then completed with all 13 cumulative checks passing. Recovery and the
final feature took another 213 seconds. This demonstrates a completion-latency
problem under this deadline, not an established permanent stall.

The final Python suite ran 567 tests successfully with 14 optional skips. The
evaluator's own tests exercise hidden-checker isolation, protected files,
incorrect implementations, terminal readiness, and comparison rejection.

The rejected candidate is retained only as
[`reliability-completion-guidance-20261018.patch`](../scripts/experiments/reliability-completion-guidance-20261018.patch).
To repeat the comparison, apply that patch to a temporary copy of
`scripts/routers/codex_local_router.py`, then run:

```bash
./bin/marathon eval reliability --run-gpu \
  --model Swift-Qwen3.8-27B-Uncensored-Merge-IQ4_XS.gguf \
  --profile one-gpu-196k-uncensored --repeats 2 \
  --cases ttl_cache honest_failure doc_buried atomic_report correction compaction resume interruption \
  --candidate-router /path/to/temporary/candidate.py
```

Raw paired evidence and frozen evaluator/router inputs are retained at
`/tmp/marathon-reliability-confirmation-20261018`. The separate six-turn project
run is at `/tmp/marathon-reliability-project-final-20261018`. Its original timeout
state was copied to `/tmp/marathon-reliability-project-at-timeout-20261018`
before resuming; assisted recovery results are at
`/tmp/marathon-reliability-project-recovery-20261018/result.json`. These are local
temporary artifacts. Calibration runs are excluded: they uncovered and corrected
multiline-paste submission and a race between journal completion and the
terminal's Ready state. One calibration
contract also needed an explicit literal value for the backup policy.

### Terminal and runtime checks

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

Pass `--model MODEL_ID` to test a specific frontend catalog entry, including
an external model such as `qwen3.8-flash-next-spark`.

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

### Launcher and model-picker performance, October 14

Normal chat/resume launches now defer `prompt_toolkit` imports until an arrow-key
menu actually opens. Model-picker identity lookup reads the broker file on every
request but reuses the parsed identity for identical contents and worker lists
(an eight-entry cache). Same-size edits with preserved timestamps still refresh
immediately, and unreadable/invalid configurations never reuse an old identity.
Worker acquisition overlaps the two NVIDIA inventory queries and Docker container
listing; it retains fresh per-lease checks, ownership validation, and failure handling.

Measured on AiGuy with Python 3.12 and the warm Qwen3.8 local worker pool:

| Measurement | Before | After | Less time |
| --- | ---: | ---: | ---: |
| Launcher Python imports, median of 25 fresh processes | 188.51 ms | 128.17 ms | 32.0% |
| Live model-picker HTTP refresh | 37.56 ms | 23.54 ms | 37.3% |
| GPU availability check, 25 alternating paired samples | 234.05 ms | 162.05 ms | 30.8% |
| Real terminal launch through Ready and successful connection probe | 2.178 s | 1.910 s | 12.3% |

The terminal comparison used three baseline and five optimized launches, with 30
catalog requests per launch. HTTP figures are medians of the per-launch medians;
every catalog response was nonempty and unchanged within its trial. All eight
launches passed a unique real-model reply check. Median prompt-to-reply time was
9.088 s before and 9.098 s after, so this establishes reduced launch/picker waiting,
not increased inference throughput. These short warm-worker tests do not measure
GPU-cold startup or establish results on other machines. The first setup/dashboard
menu still pays the deferred toolkit import cost.

The paired availability probes returned identical decisions before and after,
without probe failures. Content invalidation, worker selection, and concurrent
probe regressions passed; a real terminal also passed Down/Enter, Back, and Ctrl-C
menu checks. The full Python suite ran 508 tests successfully, with 14 skips.

Reproduce on each revision with:

```bash
.marathon/venv/bin/python scripts/benchmarks/bench_launcher.py \
  --repeats 25 --gpu-probes --output /tmp/marathon-launcher-results.json
.marathon/venv/bin/python scripts/evals/startup.py \
  --run-gpu --repeats 5 --catalog-requests 30
```

The component benchmark only queries inventory; the startup evaluator uses normal
worker leasing and never evicts other jobs. Catalog probes run after Ready and
before the prompt, so use `prompt_reply_seconds` to exclude their time from reply
comparisons. Local evidence is in `/tmp/marathon-perf-20261014`,
`/tmp/marathon-perf-before-picker-20261014`, and
`/tmp/marathon-perf-after-picker-20261014`; these are not portable fixtures.

### Nonempty warmup cache reuse, October 14

A non-generating frontend warmup can contain base instructions or conversation
history. It creates a new response ID without generating a new GPU prompt. The
router previously recognized only empty warmups as roots; following a nonempty
warmup could therefore erase an already valid live slot or restore it needlessly.

Warmups without generated output now retain root cache selection. For the same
conversation and matching model/prompt scaffold, the router reuses the live slot.
A new conversation or fork still prefers its own disk checkpoint, then reuses
the compatible live prefix on a checkpoint miss. The full prompt reaches the
backend, which validates the token-exact prefix. Lost slots and changed scaffolds
retain their existing fallback; experimental delta-only requests are excluded.
No model, sampling, reasoning, context limit, or cache capacity setting changed.

The real-GPU protocol evaluator uses a synthetic 4,492-token recall request,
normal worker leasing, and three repetitions per revision. On AiGuy with the
Qwen3.8 27B IQ4_XS worker, median request-to-completion times were:

| Measurement | Before | After | Time saved |
| --- | ---: | ---: | ---: |
| Real terminal: first reply after `/new` | 15.750 s | 0.919 s | 14.831 s (94.2%) |
| Same-conversation base warmup, next reply | 5.155 s | 0.286 s | 4.869 s (94.5%) |
| Fork base warmup, next reply | 5.355 s | 0.247 s | 5.108 s (95.4%) |
| Ordinary continuation control | 0.378 s | 0.280 s | 0.098 s |

All 12 generated answers per revision passed exact-output checks, with matching
output hashes across revisions. Affected requests went from zero cached tokens
to 4,467/4,501 cached tokens. Full-history warmups were also measured in three
repetitions, saving 4.90/5.11 seconds respectively. The ordinary-continuation
control is not a claim of improved decoding throughput. These measurements
cover the affected warmup paths, not every reply or GPU-cold startup.

The separate terminal test used three `/new` actions per revision. Each new chat
returned the exact requested answer. Cached tokens increased from 0 to 13,960,
leaving only 23 tokens to prefill. Median time to first activity fell from
15.484 s to 0.611 s. The initial launch reply was 9.298 s before and 9.227 s after;
the large improvement applies to new chats within an already active session.

Nine focused regression tests cover native base warmups, history warmups, fork
checkpoint preference, lost-slot fallback, changed scaffolds, and delta-only
exclusion. The Python suite passed 517 tests with 14 skips.
Five live terminal/headless workflows also passed: conversation recall and cache
reuse, file edits with real shell tests, background-tool cancellation and recovery,
the printed resume command with fact recall, and headless configuration overrides
with shell tools. The smoke evaluator now verifies cached-token reuse for both
delta and full-history reconnects, and waits for resume readiness without requiring
an old message to remain visible in the terminal viewport.

Reproduce on each revision with the same model and worker configuration:

```bash
.marathon/venv/bin/python scripts/evals/warmup_cache.py --run-gpu --repeats 3
.marathon/venv/bin/python scripts/evals/warmup_cache.py --run-gpu --repeats 3 --warmup-input history
.marathon/venv/bin/python scripts/evals/warmup_cache.py --run-gpu --repeats 3 --native-new-chat
```

The native variant drives `/new` in a real terminal; the other variants exercise
the Responses WebSocket protocol against real inference. All use synthetic
prompts and isolated evidence directories, with no worker eviction. Local A/B
evidence is in `/tmp/marathon-warmup-base-{before,after}-20261014` and
`/tmp/marathon-native-new-{before,after}-20261014`; these are not portable fixtures.
The baseline router was byte-identical to commit `e0d7f907`; hashes and comparison
results are in `/tmp/marathon-perf-pass2-20261014`, and live workflow evidence is in
`/tmp/marathon-pass2-smoke-complete-20261014`.

### Retained worker state and web responsiveness, October 14

Three additional changes were measured against `5ca7f79`, including the preceding
warmup optimization. Medians from live tests on AiGuy:

| Change and measured interaction | Before | After | Time saved |
| --- | ---: | ---: | ---: |
| Retain validated GPU prefix across clean relaunches: first prompt to reply | 9.263 s | 3.934 s | 5.329 s (57.5%) |
| Convert HTML outside the router event loop: model-picker HTTP response during conversion | 3.177 s | 0.015 s | 3.162 s (99.5%) |
| Reuse extraction for identical freshly downloaded bytes: repeat page fetch | 3.216 s | 0.050 s | 3.166 s (98.5%) |

The relaunch comparison used three terminal launches per revision with exact,
unique real-model reply checks on the Qwen3.8 27B IQ4_XS worker. Optimized launches
followed an initial priming launch; the optimization requires a compatible slot
left by a clean shutdown on the same worker process. Fresh isolated frontend
homes per launch still changed part of the input prefix. Cached tokens rose from
6,519 to 11,166, with the complete prompt still sent for token-exact validation.
An earlier four-launch verification also passed, with a 4.076 s median reply.
These are warm relaunch results, not GPU-cold startup or faster decoding.

The handoff stores only private, bounded metadata next to the worker lease.
It validates worker PID/start time, backend identity, profile, slot, scaffold,
and live slot presence. A hashed conversation key preserves checkpoint preference.
Only the last successfully completed profile is eligible; interrupted mutations,
restarted workers, invalid metadata, and experimental delta-only requests fall
back to existing behavior. It allocates no additional GPU cache and records no
prompt text. Optional adoption runs after frontend readiness. The startup evaluator
now waits for the terminal's Ready state before Ctrl-C rather than assuming the
input prompt always redraws. A premature-interrupt trial correctly missed the
handoff after forced shutdown; it is retained separately in the evidence.

Web tests fetched the live 3,095,308-byte Wikipedia List of Unicode characters page
through the actual managed tool handler and probed the real `/v1/models` endpoint
from a separate process. Each fetch used a different turn scope. The second row
compares three baseline samples with four offload-only samples. The third isolates
the additional reuse improvement: three warm repetitions with offload alone versus
three with both changes. The original baseline warm-fetch median was 3.222 s.
All formatted extracted-text SHA-256 hashes matched across configurations.

Every repeated fetch still downloads the current response and validates redirects,
status, and byte limits. Reuse requires identical response bytes, content type,
final URL, and output cap. Eight weak references reuse only results already owned
by existing tool history; the table retains neither HTML nor additional result
bodies. Changed content invalidates even when the server sends an unchanged ETag.
The converter and all model, reasoning, sampling, context, and cache-capacity
settings remain unchanged. The two web measurements describe different waits and
must not be added together as a single end-to-end improvement.

The Python suite ran 533 tests: 519 passed and 14 skipped. New coverage checks
worker restarts/PID reuse, incompatible slots, metadata failures, profile ownership,
fresh downloads, content invalidation, weak-reference lifetime, bounded metadata,
HTTP errors, and event-loop responsiveness. Five live workflows passed: recall
and cache reuse, file edits with shell tests, background cancellation and recovery,
resume with fact recall, and headless overrides with tools. No functional or
accuracy regressions were observed in these checks.

Reproduce with the same worker/model configuration and a stable public page:

```bash
.marathon/venv/bin/python scripts/evals/startup.py --run-gpu --repeats 4
.marathon/venv/bin/python scripts/evals/web_responsiveness.py --run-network --repeats 4
.marathon/venv/bin/python scripts/evals/e2e_smoke.py --run-gpu
.marathon/venv/bin/python -m unittest discover -s tests -v
```

For relaunch comparisons, prime a clean handoff before collecting warm samples.
The web evaluator's optional `--web-source` loads a baseline or offload-only module
in an isolated child. Its authenticated evaluation route exists only in that child.
Baseline source files were byte-identical to `5ca7f79`. Local evidence is under
`/tmp/marathon-pass3-relaunch-{before-final,complete,verified}`,
`/tmp/marathon-pass3-web-{before-v2,offload-final,after}`, and
`/tmp/marathon-pass3-smoke`. Source hashes, raw page identity, and test logs are in
`/tmp/marathon-perf-pass3-20261014`; these are not portable fixtures.

### Fresh-conversation instruction prefix, October 15

Native startup sends Marathon's base instructions as an initial developer
message. Normalization preserves it as a chronological instruction envelope in
the input. The starter cache previously saved only top-level instructions and
tools, leaving this roughly 4,646-token common input prefix uncached even after
background warmup. A real fresh conversation spent 8.630 s processing 7,063 new
tokens before its first model activity at 9.438 s.

Starter snapshots now include the exact leading normalized instruction envelopes,
stopping at the first ordinary user/history item. Their content participates in
the snapshot identity. The conversation scaffold identity remains unchanged, so
followup deltas that omit the initial instructions still match their parent.
On a new seed's first build, an existing tools-only snapshot is reused before
prefilling the additional instructions. Full inference prompts, model settings,
and token-exact backend prefix validation are unchanged.

The controlled terminal comparison used three fresh launches per revision on the
same normally leased Qwen3.8 27B IQ4_XS worker, a loaded model, zero intentional
typing delay, and **live cache handoff disabled**. Every launch restored its
persistent starter snapshot rather than relying on a previous conversation's GPU
state. All six unique exact-answer checks passed. Medians:

| Measurement | Before (`d2dc384`) | After | Reduction |
| --- | ---: | ---: | ---: |
| Terminal time to first model token | 9.372 s | 4.493 s | 4.879 s (52.1%) |
| Router request to first model activity | 8.749 s | 3.949 s | 4.800 s (54.9%) |
| Prompt submission through complete reply | 9.733 s | 4.856 s | 4.877 s (50.1%) |
| Tokens reused from the starter snapshot | 6,519 | 11,165 | 4,646 additional cached tokens |

The optimized terminal first-token samples were 4.507, 4.459, and 4.493 s. An
earlier three-launch candidate check measured 4.431-4.686 s. These are small-sample
local results, not a guarantee for every prompt or an unloaded model.

**First construction is separate from reuse.** The reported 13.024 s request
included 8.070 s waiting for the new cache to finish building; subsequent prompt
processing was already down to 3.221 s. The controlled candidate's first build
also cost 11.424 s to first token. These builds are retained in the evidence and
excluded from the restored-cache medians above. The user explicitly accepted a
slower first cache build. New tool/instruction/backend identities or cache eviction
can require another build. The larger tested seed uses about 1.0 GiB on disk,
roughly 315 MiB more than the tools-only seed, within the existing cache limits;
GPU allocation, context size, and cache budget settings are unchanged.

Seven focused tests cover exact instruction preservation, exclusion of later
history, identity invalidation, matching followup scaffolds, incremental seed
creation, restart restore, and non-mutating restore-only misses. The full suite
ran 540 tests: 526 passed and 14 skipped.
Five live workflows also passed: conversation recall and cache reuse, file edits
with shell tests, background cancellation and recovery, resume with fact recall,
and headless overrides with tools. The initial smoke attempt typed into the
transient Ready screen before resume finished loading history, so no recall
request reached the router. The evaluator now waits for the completed connection
probe and the final Starting-to-Ready transition; the rerun passed all five checks.
Evidence is in `/tmp/marathon-fresh-smoke-verified-20261015`.

Reproduce the restored-cache comparison with one free worker:

```bash
.marathon/venv/bin/python scripts/evals/startup.py \
  --run-gpu --no-live-handoff --repeats 4
```

Record the first build separately; use three subsequent launches that report a
restored starter cache for the warm comparison. `--router-source` can load a
baseline router in an isolated evaluator child. The evaluator records native
first-token timing and backend cached/prefilled token counts as well as reply
time. Local evidence is in `/tmp/marathon-fresh-baseline-warm-20261015`,
`/tmp/marathon-fresh-verified-20261015`, and
`/tmp/marathon-fresh-candidate-20261015`; baseline source and regression logs are
in `/tmp/marathon-fresh-20261015`. No existing user sessions were stopped or
reassigned during these tests.

### Branch cache, streaming, and cached search, October 17

Full-history branches without a saved parent snapshot now retain a verified live
slot from the same conversation and instruction/tool scaffold. The backend still
receives the complete branch history and keeps only its token-exact shared prefix.
Disk checkpoints remain preferred; missing/unknown live state, a failed restore,
changed scaffolds, different conversation keys, and delta-only requests keep their
existing fallback paths. New router processes pick up the change.

Text validation skips Markdown parsing when there is no `<` character. Both tool
markers require that character, so protocol detection, split-marker buffering,
quoted examples, and executable-tool validation are preserved. Text containing
`<` still uses the existing parser; this is an ordinary-prose optimization.

Cached searches can now read a committed, unexpired result without taking the
network-request lock. The optimistic SQLite read runs off the event loop with
zero lock timeout. Missing/busy databases use the existing locked path, which
rechecks the cache and preserves request coalescing, pacing, and engine cooldowns.
Expiry is checked again after the worker-thread read completes.

Synthetic measurements on AiGuy, Python 3.12, and the normally leased Qwen3.8 27B
IQ4_XS worker on GPU 1 (RTX 3090 Ti):

| Check | Before | After |
| --- | ---: | ---: |
| Branch first text, 8,288-token request, median of three per revision | 8.808 s | 0.210 s |
| Branch tokens prefilled / reused | 8,288 / 0 | 23 / 8,265 |
| Process a 32 KiB prose stream in four-character chunks, median of three | 5.115 s | 0.053 s |
| Cached search while an unrelated fetch holds the lock | Test queue timeout at 0.201 s | Hit in 0.00027 s |

The GPU test disabled disk checkpoints, starter caching, and retained-slot
handoff only in its isolated routers. It compared branches against equivalent
full-history requests: forwarded-request and answer hashes matched in every
pair, and all 30 seed, preparation, and measured replies were correct. This
measures the missing-checkpoint condition, not its frequency in normal usage.
The prose replay preserved every output/event hash and measures router CPU work,
not model token-generation speed. The search used a mock blocked fetch and a
0.2-second test timeout; production's queue timeout remains 45 seconds.

A scratch Rust shared library performed the same marker-presence check through
an optimized native `memchr`, including UTF-8 conversion and Python call overhead.
The same 32 KiB replay took 64.5 ms, versus 53.2 ms with the Python fix. Profiling
the fixed replay attributed about 3 ms to all 8,196 marker checks; event handling
and JSON serialization accounted for most remaining work. This small experiment
does not justify a native dependency or establish the speed of a complete router
rewrite. The frontend already uses Rust and local inference runs in llama.cpp;
neither is replaced by rewriting the Python coordinator.

Run the focused regression checks without inference:

```bash
PYTHONPATH=tests .marathon/venv/bin/python -m unittest \
  test_tool_contract test_warmup_cache test_search_coordinator -v
```

Those 37 tests passed. The full suite ran 555 tests: 541 passed and 14 optional
checks skipped. The installed frontend's scripted HTTP research/edit integration
also passed. Python compilation and tracked shell syntax checks passed.
All five live terminal workflows passed with synthetic data and isolated homes:
recall/cache reuse, file edits with tests, background cancellation and recovery,
resume with recall, and headless overrides with shell tools. These smoke routers
also disabled persistent cache access; existing saved conversations were not read.
Local baseline sources, benchmark scripts, JSON measurements, and profiling data
are in `/tmp/marathon-fixes-20261017`; these temporary evidence paths are not
portable repository fixtures. The earlier blocked-search baseline is in
`/tmp/marathon-self-check-8vuylqtg/report.json`. Existing private conversations
were not used for any test.

Keep first-run setup in the README and advanced configuration in [advanced usage](ADVANCED_USAGE.md).
Document actual tested support separately from expected hardware compatibility.
Preserve existing user selections and never change power caps or stop unrelated inference services during setup.
