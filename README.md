# Marathon

Marathon is a one-command, terminal-first local AI runtime. It discovers GGUF
models in the centralized AI directory, starts the model's configured local
backend and Marathon's API router in the foreground, then opens either Codex or
a clean Direct Chat. Shipped profiles currently cover upstream llama.cpp,
DeepSeek V4's optimized long-context llama.cpp fork, and the legacy four-GPU
DwarfStar pipeline.

Planned work that is intentionally not current behavior is tracked in
[`docs/FUTURE_UPDATES.md`](docs/FUTURE_UPDATES.md).

The process showing the Marathon dashboard owns the backend. Exiting Marathon
stops its router and every backend process group and frees the GPUs. This lifecycle
works through an ordinary SSH shell; no exposed daemon or web UI is required.
Marathon also has a native remote-client mode: Codex and its tools run on a
client machine while inference runs on a Linux GPU host through an SSH tunnel.

## Quick Start

```bash
git clone --recurse-submodules https://github.com/thebusdriver37/Marathon.git
cd Marathon
./bin/marathon install        # makes `marathon` available from any repo
marathon setup-deps           # private router/UI Python environment
marathon search up            # optional local web search
marathon doctor
```

Run Marathon from the project directory you want Codex to edit:

```bash
cd /path/to/your/repo
marathon
```

The remembered model, profile, and frontend are preselected. Use the arrow keys
and press Enter to load the highlighted setup. Model selection drills directly
into that model's profiles. When a frontend exits, Marathon returns to its
dashboard with the model still warm. Reopen Codex, enter Direct Chat, switch
models, or quit and unload the backend.

```text
marathon  →  Enter  →  model loads  →  Codex
                                      ↓ exit
                         Marathon dashboard
                     Codex · Direct Chat · switch · quit
```

## Native Mac client with Linux GPUs

Install the same Marathon checkout on the Mac and Linux GPU host. The Linux
host needs the models and Marathon's configured inference backends. The Mac
needs Marathon's Python environment, Codex, and key-based SSH access, but it
does not need local models or llama.cpp.

From the Mac, enter the project that Codex should edit and name the SSH host:

```bash
cd ~/Documents/my-mac-project
marathon remote deforest@gpu-rig
```

The normal arrow-key model/profile dashboard is populated from the Linux host.
After selection, Marathon starts the Linux runtime in the foreground, creates a
loopback-only SSH tunnel, and launches Codex on the Mac. Consequently:

- Codex tools read and edit the Mac project, not the Linux filesystem.
- Mac-global skills, plugins, configuration, and local/repository `AGENTS.md`
  discovery work normally because the Codex process is local.
- Inference workers, the router, GPU telemetry, and models remain on Linux.
- Exiting Codex returns to the Mac dashboard with the remote model warm.
- Quitting Marathon, losing SSH, or closing the client connection stops the
  Linux supervisor and frees its GPUs.

The inference API remains bound to `127.0.0.1` on both ends; no unauthenticated
port is exposed to the LAN or internet. `BatchMode=yes` prevents Marathon from
hanging on an SSH password prompt, so first ensure this succeeds from the Mac:

```bash
ssh -o BatchMode=yes deforest@gpu-rig true
```

Use an entry in `~/.ssh/config` if the host needs a custom port or identity key.
If `marathon` is not on the Linux host's non-interactive SSH `PATH`, point the
Mac client at its absolute remote launcher:

```bash
MARATHON_REMOTE_BIN=/home/deforest/.local/bin/marathon \
  marathon remote deforest@gpu-rig
```

By default Marathon recursively discovers models under:

```text
~/AI/models/gguf/
```

Set `MARATHON_AI_ROOT` to move the entire model, backend, and cache hierarchy,
or `MARATHON_MODELS_DIR` to override only model discovery. Model families and
selectable profiles are defined in `config/runtime_catalog.toml`.

```bash
MARATHON_AI_ROOT=/mnt/local-ai marathon       # move the complete hierarchy
MARATHON_MODELS_DIR=/mnt/models marathon      # override only GGUF discovery
LLAMACPP_BIN=/opt/llama.cpp/llama-server marathon
MARATHON_BACKEND_DS4_DISTRIBUTED=/opt/ds4/ds4-server \
MARATHON_BACKEND_DS4_DISTRIBUTED_WORKER=/opt/ds4/ds4 marathon
```

Specific overrides take precedence over `MARATHON_AI_ROOT`; relative paths in
the runtime catalog are resolved beneath the effective AI root.

llama.cpp profiles may use automatic placement or explicit tensor splits.
DeepSeek V4 Flash defaults to **Stable 64K**, which uses the `ds4-longctx`
branch of `alesha-pro/llama.cpp`. Its constant-shape, causal-indexer, and
MoE-tile switches are catalog data rather than model-name conditionals in Marathon. The
profile uses deterministic sampling, F16 KV, 4,096-token batches, and a
`1,1,0.95,0.9` layer split.

**Experimental MTP 64K** adds the fork's speculative-decoding path. It can
raise decode throughput from roughly 24 to 30 tokens/second on this four-GPU
rig. The former long-continuation crash was traced to `DSV4_SPARSE_FA`, which
poisoned the base model logits before MTP exposed the invalid token index. The
root cause was a stream-K bookkeeping mismatch: the sparse CUDA kernel processed
the selected rows while its result combiner mapped the full KV width. The patched
backend now schedules the selected rows consistently, both DeepSeek profiles
enable sparse prompt attention, and the exact 40K/42K/43K regression sequence
passes with MTP active. The profile remains experimental until it completes
broader real Codex workloads.

**MTP 128K** requests the full 131,072-token window with the fork author's
documented `1,1,1,0.85` four-GPU placement. It keeps the same F16 compressed
KV representation, 4,096/512 prompt batching, sparse attention, greedy
sampling, and MTP decode path. A 120,021-token local validation completed at
389.0 prompt tokens/second and 17.2 decode tokens/second, recovered an exact
needle from the start of the prompt, and stayed below 22.7 GiB on every GPU.
These figures describe the four-GPU development rig, not a portability
guarantee. The 64K profiles remain unchanged.

The optimized backend is intentionally stored outside the repository beneath
the centralized AI root. Build it once on the Linux GPU host:

```bash
git clone --branch ds4-longctx https://github.com/alesha-pro/llama.cpp \
  ~/AI/backends/llama.cpp-ds4-longctx
cmake -S ~/AI/backends/llama.cpp-ds4-longctx \
  -B ~/AI/backends/llama.cpp-ds4-longctx/build-cuda \
  -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build ~/AI/backends/llama.cpp-ds4-longctx/build-cuda \
  --config Release -j"$(nproc)"
```

If the machine has multiple CUDA toolkits, set `CUDACXX` to the intended
`nvcc` before configuring. Place the matching
`DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf` beside the main DeepSeek GGUF, or set
`DSV4_MTP_GGUF` to its absolute path. `marathon doctor` reports a missing fork
or sidecar before a normal run is attempted.

**Legacy DS4 64K** remains available as a rollback profile. It uses one
contiguous layer slice per GPU, three workers, and one HTTP coordinator.
Marathon supervises all four processes as one foreground backend and stops them
together. `MARATHON_DS4_GPUS` overrides its catalog `0,1,2,3` GPU mapping.

## Commands

```bash
marathon                  # dashboard; Enter starts remembered Codex setup
marathon codex            # dashboard with Codex selected
marathon direct           # dashboard with clean Direct Chat selected
marathon remote HOST      # local Codex, remote Linux GPUs over secure SSH
marathon tune             # open Dyno directly
marathon models           # list installed centralized GGUF models
marathon status           # inspect an active foreground runtime
marathon stop             # emergency stop request from another SSH shell
marathon report           # post-mortem for the latest run
marathon report <run-id>  # inspect a particular run
marathon compare A B      # compare two runs or model/profile configurations
marathon doctor           # diagnose setup, GPUs, models, and ports
marathon search up        # optional local SearXNG for Codex web tools
```

The old `marathon backend ...` commands remain temporarily for compatibility,
but the dashboard does not use their detached-process lifecycle.

## Dyno machine tuning

Choose **Tune / benchmark** from the cold dashboard, or run `marathon tune`.
Dyno asks for one priority: Balanced, Fastest responses, Longest context,
Quality / reliability, or Lowest power. It then runs three or four bounded
candidate profiles against the selected model's shipped default and saves the
Pareto-optimal passing result.

Dyno currently tunes llama.cpp profiles only. Architecture-specific distributed
backends such as DS4 use their verified shipped profile and do not show the Tune
menu until backend-aware candidate generation is implemented.

Dyno is deterministic infrastructure rather than an LLM judge. Every candidate
must load its requested context and complete two fixed-seed server requests.
The runner measures prompt and decode throughput, latency, GPU power,
utilization, peak VRAM, temperature, and energy. Quality mode uses higher-precision
KV cache and a deterministic response gate; it does not claim to improve or
measure the underlying quantized weights. Context mode verifies allocation and
executes an 8K-token workload, rather than pretending one short request proves
perfect recall across the entire window.

The servers are children of the foreground Dyno process and are stopped between
trials or on interruption. Shipped profiles are never overwritten. Winners are
stored as machine-local profiles and automatically invalidated when the GPU
identity, model file, or llama.cpp backend changes:

```text
~/.config/marathon/dyno/profiles/<model>.json
~/.local/state/marathon/dyno/<run>/summary.json
~/.local/state/marathon/dyno/<run>/<trial>.log
```

The generated `Dyno · …` profile appears under its model in Marathon's normal
profile drill-down and can be launched like any shipped profile. This keeps the
daily workflow simple while retaining complete trial evidence for later review.

## Codex behavior

Marathon prefers its small patched Codex binary at
`$XDG_DATA_HOME/marathon/bin/codex` and falls back to the installed stock
`codex` command. It leaves `CODEX_HOME` untouched, so global configuration,
installed skills, plugins, session history, and normal global/repository/nested
`AGENTS.md` discovery work as they do in Codex itself. Marathon supplies only
per-invocation overrides for its local provider, selected model, context window,
and generated model catalog.

The selected profile requests the backend's context allocation, but Marathon
does not assume that request became the loaded size. After startup it reads the
backend's reported `n_ctx` or `context_length` and propagates that exact runtime
value to the router,
Codex model catalog, session status, compaction limit, and truncation policy.
The same path therefore handles 64K, 128K, 256K, or another backend-supported
window without a model-specific context constant in the application code.
Marathon also reserves model-scaled headroom before automatic compaction: 12K
minimum, one eighth of the loaded window through 256K, and a 32K ceiling. This
keeps space for hidden chat-template overhead, tool results, and the next model
generation. `MARATHON_CONTEXT_RESERVE_TOKENS` and
`MARATHON_COMPACTION_GUARD_TOKENS` override the two dynamic margins when a
measured profile needs different tuning.

Router-normalized shell/function outputs are bounded to 16,384 characters per
item with both the head and tail preserved. The truncation marker tells the
model to rerun a narrower command. Set `MARATHON_TOOL_OUTPUT_MAX_CHARS` to tune
that limit; this bounds model context, not what Codex displays to the user.
Individual backend responses are also capped at one eighth of the context
window, between 2,048 and 8,192 generated tokens. This prevents an unbounded
reasoning or edit turn from monopolizing the GPUs; large edits should be split
across tool calls. `MARATHON_MAX_OUTPUT_TOKENS` overrides the cap.
Profiles whose backend supports a native thinking budget can set
`tool_thinking_budget`. Marathon applies it only after Codex returns a tool
result; initial user turns remain unrestricted. Set
`MARATHON_ADAPTIVE_THINKING_BUDGET=0` to disable this behavior for an A/B test.
The DS4 backend exposes high or disabled thinking rather than a fixed native
token budget, so its shipped profile relies on the model-agnostic response cap
and stalled-response recovery instead of advertising a budget it cannot honor.
Profiles can separately advertise `parallel_tool_calls` after that capability
has been verified for the model and chat template. It remains off by default;
DeepSeek's raw API can emit concurrent calls to distinct tools, but a real
Codex obstacle run did not batch repeated `exec_command` calls and regressed
completion quality, so the DeepSeek 64K profile deliberately leaves it off.
If a capped agent response contains only unfinished reasoning, Marathon makes
one bounded recovery request requiring a concrete tool action instead of
reporting a false-successful Codex turn. Set
`MARATHON_STALLED_RESPONSE_RECOVERIES=0` to disable that recovery.
Tool calls also have a model-agnostic protocol guard: invalid patch JSON, exact
repetition loops, or arguments over 24,576 characters are aborted and retried
once with a smaller generation budget and instructions to split the edit. A
supervised backend stream is allowed to remain quiet while a structured tool
call is generated; Marathon sends Codex independent progress keepalives and an
actual backend exit closes the stream immediately. This avoids cancel/retry and
checkpoint-rewind loops on slower local models. Reconnects for the same Codex
prompt-cache session still supersede a genuinely abandoned in-flight request.
`MARATHON_TOOL_ARGUMENT_MAX_CHARS` and `MARATHON_TOOL_PROTOCOL_RECOVERIES` tune
the protocol guard and its bounded recovery.
The Marathon Codex patch removes stock Codex's fixed 12K display normalization,
so the visible percentage is the backend-reported active tokens divided by that
loaded window. `marathon build-codex` performs a release build in a temporary
target directory, installs only the resulting binary, and removes build output.
`marathon update-codex` explicitly fetches current upstream, preflights the
patches in a throwaway worktree, runs the focused Codex and complete Marathon
test suites, smoke-checks the CLI, and then atomically promotes the release
binary. The previous binary remains beside it as `codex.previous`. Ordinary
Marathon startup does not fetch, compile, or replace Codex.

## Direct Chat

Direct Chat sends a streaming Chat Completions request through the local
router. It deliberately provides no tools, coding-agent prompt, AGENTS files,
skills, memory, or Hermes harness. Use `/new` to clear the conversation and
`/back` to return to the warm dashboard.

## Runtime storage

- Models and backends: `~/AI/`
- Slot cache and router state: `~/AI/cache/marathon/`
- Remembered selection: `~/.config/marathon/selection.json`
- Dyno tuned profiles: `~/.config/marathon/dyno/profiles/`
- Dyno benchmark evidence: `~/.local/state/marathon/dyno/`
- Per-run telemetry: `~/.local/state/marathon/runs/*.jsonl`
- Compatibility process logs: `~/.local/state/marathon/logs/`
- Live lock and PID metadata: `$XDG_RUNTIME_DIR/marathon/`

## Per-run observability

Every foreground backend launch creates one append-only JSONL flight recorder.
The supervisor, router, Codex importer, Direct Chat, backend output capture,
GPU sampler, host sampler, and optional backend `/metrics` sampler all append
correlated events to that same file. A trace remains readable when Marathon or the machine
stops unexpectedly; a missing `run.completed` event marks an interrupted run.

The default trace is metadata-oriented. It records model/profile arguments,
loaded context, timings, token counters, reasoning effort and exposed reasoning
token counts, tool names and durations, cache/slot behavior, process output,
GPU identity/utilization/VRAM/power/temperature, host load/memory, and errors. It
also follows the system journal for NVIDIA Xids, NVRM failures, and PCIe/AER
alerts while the run is alive. It
does not intentionally copy Codex prompt text, responses, tool arguments, tool
outputs, developer instructions, or hidden model reasoning. Common credential
shapes in operational process output are redacted and long lines are bounded.

```bash
marathon report                 # latest trace
marathon report a82f31          # unique filename/run-id fragment
marathon compare a82f31 b4c901  # side-by-side throughput/resource comparison
```

`marathon report` also reads an open Codex rollout, so token usage, tool calls,
tool failures, and backend throughput remain visible while the foreground
session is still running. It reads on demand; it does not add a monitor process.

No telemetry daemon or database runs in the background, and Marathon does not
automatically delete traces. Set `MARATHON_RUNS_DIR` to place them on another
disk. Sampling defaults to two seconds; `MARATHON_TELEMETRY_INTERVAL=5` reduces
sampling overhead. Backend `/metrics` polling is disabled by default because it
can queue behind long inference requests; opt in with
`MARATHON_BACKEND_METRICS_ENABLED=1`. Set
`MARATHON_TELEMETRY_PROCESS_OUTPUT=0` to omit mirrored backend/router
operational lines. Set `MARATHON_ELECTRICITY_RATE_USD_KWH` to
include an estimated GPU-energy cost in reports; this excludes CPU and PSU
conversion losses and is therefore not a wall-power measurement.

These traces measure real workloads. Repeatable model-quality claims still
require fixed eval tasks, seeds, sampling settings, software versions, and
warm/cold conditions; `marathon compare` deliberately reports measurements
rather than inventing a quality score.

## Diagnostics and Model Checks

Use `marathon doctor` when setting up a new machine or debugging a failed run.
It checks Codex, the backends required by installed models, centralized models,
Marathon's private Python environment, GPU visibility, ports, and optional SearXNG.

## Prompt Cache Snapshots

Linear follow-up turns reuse llama.cpp's live prompt slot, so normal Codex
conversations do not write large prompt snapshots to disk. Optional disk
snapshots can preserve recent in-process branches, but each snapshot can be
hundreds of megabytes at long context and adds synchronous I/O after a turn.
They are therefore disabled by default.

Backends without llama.cpp's slot API, including DS4, receive Marathon's full
response lineage and manage their own live prefix cache. The router never sends
llama.cpp-only slot fields or attempts disk slot snapshots for those backends.

At router startup Marathon deletes stale slot snapshots, because the in-memory
response lineage needed to use them does not survive a router restart. During
runtime it keeps each profile's snapshot directory capped by both count and
bytes so 128K sessions cannot silently fill the disk.

| Var | Default | Purpose |
|---|---|---|
| `MARATHON_SLOT_SAVE_ROOT` | `~/AI/cache/marathon/slots` | Root directory passed to llama.cpp `--slot-save-path` launchers |
| `MARATHON_SLOT_SNAPSHOTS_ENABLED` | `0` | Opt into disk snapshots for recent in-process conversation branches |
| `MARATHON_SLOT_SNAPSHOT_MAX_COUNT` | `16` | Max snapshots retained per model profile during one router process |
| `MARATHON_SLOT_SNAPSHOT_MAX_BYTES` | `32 GiB` | Max snapshot bytes retained per model profile |
| `MARATHON_SLOT_SNAPSHOT_CLEAN_STARTUP` | `1` | Delete stale snapshots on router startup; set `0` only for debugging |

## Web tools (Search + Fetch)

Marathon ships a layered web-tool pipeline for the local model:

- **`web_search`** — backed by a self-hosted SearXNG container, returns
  ranked snippets (title, URL, content) for a natural-language query.
- **`web_fetch`** — fetches a single URL, runs it through `trafilatura`, and
  returns clean Markdown with link preservation. Use this after `web_search`
  whenever the model needs full page content (verbatim quotes, long docs,
  complete article bodies). Replaces ad-hoc `curl` from the model.
- **`web_browse`** — optional Crawl4AI-backed browser rendering for pages where
  `web_fetch` fails or returns mostly empty/navigation content. This path is
  heavier, is exposed only when Crawl4AI is installed, and only runs when the
  model explicitly calls it.

These tools are exposed as one unit: enabling web search also enables fetch and
browse.
The router translates Codex's `web_search` config into the function tools the
local model can actually call, runs the multi-turn tool loop transparently
inside one Codex request, and rewrites the result back into `web_search_call`
ResponseItems (`action.type=search` or `action.type=open_page`) so Codex's TUI
renders native pills.

**Prerequisites:** Docker 20.10+ with the `docker compose` v2 plugin, and
Python 3.10+ on the host.

```bash
./bin/marathon setup-deps       # one-time: create .marathon/venv for the router
./bin/marathon search up        # one-time: pull image, generate secret, start SearXNG
./bin/marathon                  # launch — router wires the web tools
```

The first `search up` writes `docker/searxng/.env` from `.env.example`,
generates a fresh `MARATHON_SEARXNG_SECRET`, and binds the container to
`127.0.0.1:18093`. Edit `.env` to change bind/port or the pinned SearXNG image.

```bash
./bin/marathon search status    # check container state
./bin/marathon search logs      # tail container logs
./bin/marathon search down      # stop and remove the container
```

### Configuration

| Var | Default | Purpose |
|---|---|---|
| `MARATHON_SEARXNG_URL` | `http://127.0.0.1:18093` | Where the router talks to SearXNG |
| `MARATHON_WEB_SEARCH_TIMEOUT` | `15` | Per-search timeout (seconds) |
| `MARATHON_WEB_SEARCH_MAX_RESULTS` | `8` | Results returned to the model per call |
| `MARATHON_WEB_SEARCH_MAX_ITERS` | `5` | Cap on managed tool-call rounds inside one Codex turn |
| `MARATHON_WEB_TOOL_CACHE_MAX_ENTRIES` | `256` | In-memory exact-action replay cache; prevents a reconnect from executing the same search/fetch twice |
| `MARATHON_WEB_SEARCH_MODE` | `cached` | Forwarded to Codex's `web_search` config: `disabled`, `cached`, or `live` |
| `MARATHON_WEB_FETCH_TIMEOUT` | `25` | Per-fetch timeout (seconds) |
| `MARATHON_WEB_FETCH_MAX_CHARS` | `20000` | Default cap on extracted-content chars per fetch |
| `MARATHON_WEB_FETCH_MAX_CHARS_CAP` | `200000` | Hard cap on what the model can request |
| `MARATHON_WEB_FETCH_MAX_BYTES` | `5 MiB` | Hard cap on raw response size |
| `MARATHON_WEB_FETCH_USER_AGENT` | Marathon-identifying Mozilla UA | UA the fetch executor sends |
| `MARATHON_WEB_FETCH_ALLOW_PRIVATE` | unset | Set to `1` only if fetches to loopback/private network URLs are intentional |
| `MARATHON_WEB_BROWSE_ENABLE` | `1` | Set to `0` to hide the optional Crawl4AI-backed `web_browse` tool path |
| `MARATHON_ROUTER_PYTHON` | `.marathon/venv/bin/python3` | Python interpreter the launcher uses for the router |

## Updating Codex

Ordinary Marathon use prefers the tested patched binary and falls back to stock
Codex only when that binary is absent. To fetch current upstream, preflight the
small Marathon patch set, run the regression gates, and atomically install the
result:

```bash
marathon update-codex
```

If a patch fails, Codex changed in the touched area. Rebase or retire that patch
intentionally rather than editing Codex ad hoc. Set
`MARATHON_FORCE_CODEX_REBUILD=1` only when the recorded current commit must be
rebuilt from scratch.
