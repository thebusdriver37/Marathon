# Advanced Usage

This guide covers Marathon features that are intentionally kept out of the beginner README.

## Runtime Lifecycle

Marathon supervises the model backend and API router as children of one foreground process.
Stopping Marathon terminates every owned process group and frees the GPUs.
The same lifecycle works in a local terminal and through an ordinary SSH shell.

The normal `marathon` command starts the remembered model and opens Codex directly.
The `marathon dashboard` command exposes model profiles, alternate frontends, warm-backend reuse, and Dyno tuning.

Legacy `marathon backend ...` commands remain available for compatibility, but the normal runtime does not use their detached-process lifecycle.

## Model and Backend Paths

Marathon uses this default hierarchy:

```text
~/AI/models/gguf/
~/AI/backends/
~/AI/cache/marathon/
```

Use `MARATHON_AI_ROOT` to relocate the complete hierarchy.
Use `MARATHON_MODELS_DIR` to replace model discovery with one folder.
Use `MARATHON_MODEL_DIRS` for an OS-path-separated list of folders.
Use `marathon models add PATH` to persist additional folders without moving their contents.

Specific backend overrides take precedence over `MARATHON_AI_ROOT`:

```bash
MARATHON_AI_ROOT=/mnt/local-ai marathon
MARATHON_MODELS_DIR=/mnt/models marathon
LLAMACPP_BIN=/opt/llama.cpp/llama-server marathon
```

Model families, backend mappings, reasoning levels, and selectable profiles are data in [`config/runtime_catalog.toml`](../config/runtime_catalog.toml).

## llama.cpp Setup

`marathon setup-llama` clones the commit recorded in [`config/llamacpp.ref`](../config/llamacpp.ref) and builds it under the configured AI root.
The source and build output remain outside this repository.

CUDA is selected when `nvcc` is available.
Set `MARATHON_GPU_BACKEND=cpu` or `MARATHON_GPU_BACKEND=cuda` to make the choice explicit.
Set `MARATHON_LLAMACPP_DIR` or `MARATHON_LLAMACPP_BUILD_DIR` to override the source or build location.

The default Qwen 3.8 profile requests 262,144 tokens, automatic GPU layers, Q8 KV cache, and llama.cpp memory fitting.
The exact four-GPU 256K profile remains available through the dashboard for validated machines.

Unrecognized GGUF models use a conservative automatic 32K profile.
Use the dashboard or add a catalog family when a model has known, tested requirements.

## Native Mac Client with Linux GPUs

Install the same Marathon checkout on the Mac and Linux GPU host.
The Linux host needs the GGUF models and inference backends.
The Mac needs Marathon, the desired frontend, and key-based SSH access.

From the Mac, enter the local project and name the GPU host:

```bash
cd ~/Documents/my-project
marathon remote user@gpu-host
```

Codex and its tools run on the Mac, so they edit the Mac project and use the Mac's configuration, skills, plugins, and `AGENTS.md` files.
Inference, models, backend logs, and GPU telemetry remain on Linux.
Marathon connects through a loopback-only SSH tunnel and stops the remote runtime when the client disconnects.

Marathon uses noninteractive SSH, so verify key authentication first:

```bash
ssh -o BatchMode=yes user@gpu-host true
```

If `marathon` is not on the remote noninteractive `PATH`, provide its absolute location:

```bash
MARATHON_REMOTE_BIN=/home/user/.local/bin/marathon \
  marathon remote user@gpu-host
```

## Dyno Tuning

Run `marathon tune` or choose **Tune / benchmark** from the cold dashboard.
Dyno asks whether to prioritize balance, response speed, context length, reliability, or power use.

Each candidate must load and complete fixed-seed requests before it can be selected.
Dyno records prompt and decode throughput, latency, VRAM, utilization, power, temperature, and energy.
It stores passing machine-local profiles without modifying the shipped catalog.

Tuned profiles are invalidated when the model file, GPU identity, or llama.cpp backend changes.

```text
~/.config/marathon/dyno/profiles/<model>.json
~/.local/state/marathon/dyno/<run>/summary.json
~/.local/state/marathon/dyno/<run>/<trial>.log
```

Dyno currently tunes llama.cpp profiles only.
Architecture-specific distributed backends keep their shipped profiles.

## Codex Integration

Marathon prefers its patched Codex binary at `$XDG_DATA_HOME/marathon/bin/codex` and falls back to a stock `codex` command.
It leaves `CODEX_HOME` unchanged, so normal Codex configuration, sessions, skills, plugins, and `AGENTS.md` discovery continue to work.

Marathon supplies invocation-specific settings for the local provider, selected model, loaded context, model catalog, and status line.
After backend startup, Marathon reads the context that was actually loaded and propagates that value to Codex, compaction, and truncation limits.

The patched status line includes:

- Live estimated generation throughput with a `~` prefix.
- Exact completed-turn output tokens divided by active generation time.
- The active reasoning effort.
- Context usage based on the backend's actual loaded window.

Tool execution time is excluded from completed-turn generation throughput.

Use Codex's `/model` menu to change the active reasoning effort without reloading the GGUF or discarding the conversation.
Supported values are defined per model family in the runtime catalog.

Marathon sessions remain in Codex's normal session store but are tagged with the `marathon-local` provider.
The resume picker filters on that provider, and Marathon-branded sessions print `marathon resume <id>` when they exit.
The `marathon resume` and `marathon fork` commands start the remembered backend before opening Codex and stop it again afterward.

Marathon bounds tool outputs and individual model responses to protect the context window from accidental unbounded output.
It also performs one bounded recovery when a response contains only unfinished reasoning or a malformed tool call.

## Alternate Frontends

`marathon direct` opens a streaming chat frontend without coding tools, agent instructions, skills, memory, or `AGENTS.md` files.
Use `/new` to clear its conversation and `/back` to return to the warm dashboard.

`marathon hermes` opens Hermes for profiles that advertise compatible agent context.
Marathon points only that child process at its supervised local API.
Hermes continues to use its normal configuration, memory, skills, tools, and session history.

## DeepSeek and Distributed Profiles

The catalog retains specialized DeepSeek V4 and legacy four-GPU DwarfStar profiles.
These are advanced compatibility paths and are not part of the default Qwen setup.

DeepSeek V4 profiles expect the `ds4-longctx` llama.cpp fork under:

```text
~/AI/backends/llama.cpp-ds4-longctx/build-cuda/bin/llama-server
```

The MTP profiles also require the matching MTP GGUF beside the main model or an explicit `DSV4_MTP_GGUF` path.
Run `marathon doctor` to identify a missing fork, worker, or sidecar before starting the model.

The legacy DwarfStar profile uses three workers and one coordinator across four GPUs.
`MARATHON_DS4_GPUS` overrides its default `0,1,2,3` mapping.

## Runtime Storage and Telemetry

| Data | Default location |
|---|---|
| Model and backend hierarchy | `~/AI/` |
| Slot cache and router state | `~/AI/cache/marathon/` |
| Remembered selection | `~/.config/marathon/selection.json` |
| Registered model folders | `~/.config/marathon/models.json` |
| Dyno profiles | `~/.config/marathon/dyno/profiles/` |
| Dyno evidence | `~/.local/state/marathon/dyno/` |
| Per-run traces | `~/.local/state/marathon/runs/*.jsonl` |
| Compatibility logs | `~/.local/state/marathon/logs/` |
| Live process metadata | `$XDG_RUNTIME_DIR/marathon/` |

Every foreground launch creates an append-only JSONL trace.
The trace records timings, token counters, reasoning effort, tool names and durations, cache behavior, process output, GPU measurements, host measurements, and errors.
Common credential shapes are redacted and long operational lines are bounded.

The trace does not intentionally include prompts, responses, tool arguments, tool outputs, developer instructions, or hidden reasoning.
No telemetry daemon or database runs in the background.

```bash
marathon report
marathon report RUN_ID
marathon compare RUN_A RUN_B
```

Set `MARATHON_RUNS_DIR` to move traces to another disk.
Set `MARATHON_TELEMETRY_INTERVAL=5` to reduce the default two-second sampling frequency.
Set `MARATHON_TELEMETRY_PROCESS_OUTPUT=0` to omit mirrored backend and router lines.

Optional disk prompt snapshots are disabled by default because long-context snapshots can be large.
Their count and total size are bounded when enabled.

## Web Search and Fetch

`marathon search up` starts the optional loopback-only SearXNG service through Docker Compose.
Marathon then exposes search, page fetch, and optional browser-rendering tools to the local model.

```bash
marathon search up
marathon search status
marathon search logs
marathon search down
```

The first start generates a local secret in `docker/searxng/.env`.
Set `MARATHON_SEARXNG_URL` when the service runs somewhere else.
Set `MARATHON_WEB_SEARCH_MODE=disabled`, `cached`, or `live` to control Codex's web-search mode.
Set `MARATHON_WEB_BROWSE_ENABLE=0` to hide the optional browser-rendering path.

Fetch blocks loopback and private-network targets unless `MARATHON_WEB_FETCH_ALLOW_PRIVATE=1` is explicitly set.

## Updating Codex

Run this command to fetch upstream Codex, preflight Marathon's patches, run regression tests, build a release binary, and install it atomically:

```bash
marathon update-codex
```

The previous binary remains available as `codex.previous`.
Ordinary Marathon startup does not fetch, compile, or replace an existing Codex binary.

`marathon build-codex` builds the currently pinned submodule and installs the patched binary.
Temporary Cargo build output is removed after the build so the repository does not accumulate a large target directory.

## Diagnostics

`marathon doctor` checks the launcher, Python environment, Codex, configured backends, model folders, required sidecars, GPU visibility, active ports, web search, and storage use.

Use these commands when diagnosing an active or completed run:

```bash
marathon status
marathon report
marathon stop
```
