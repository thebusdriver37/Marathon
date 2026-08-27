# Advanced Usage

This guide covers Marathon features that are intentionally kept out of the beginner README.

## Runtime Lifecycle

Marathon supervises the model backend and API router as children of one foreground process.
Stopping Marathon terminates every owned process group and frees the GPUs.
The same lifecycle works in a local terminal and through an ordinary SSH shell.

The normal `marathon` command starts the remembered model and opens Codex directly.
The `marathon dashboard` command exposes model profiles, alternate frontends, warm-backend reuse, and Dyno tuning.
`marathon exec PROMPT` uses the same supervised lifecycle for headless and CI work, including starting the remembered model and cleaning it up afterward.
`marathon codex -- CODEX_ARGS` forwards Codex flags without bypassing supervision.

## Named Instances

The default instance preserves Marathon's original paths and configured ports.
A named instance uses the same model catalog and runtime profiles but owns independent mutable state.

Configure local GPU policy in `~/.config/marathon/catalog.toml`:

```toml
[instances.gpu23]
gpus = [2, 3]
```

If the default profile is pinned to GPUs 0 and 1, the following commands run two copies of that profile concurrently:

```bash
marathon
marathon --instance gpu23
```

The instance GPU list overrides the selected profile's `gpus` value without duplicating the profile.
This makes one tested model and profile reusable across several GPU groups.
A new named instance initially copies the default remembered model and profile choice, then writes its own selection file.

Marathon derives backend and router ports from a stable hash of the instance name.
Use explicit values when a machine has a fixed port policy or a derived port conflicts with another service:

```toml
[instances.gpu23]
gpus = [2, 3]
llama_port = 18082
router_port = 28111
```

Before launching any child process, Marathon checks every requested listener and selected GPU.
An occupied resource produces an error containing the port or GPU, PID, process name, and command.
Marathon never stops or evicts an occupying external process.

Instance identity applies to `dashboard`, `codex`, `exec`, `hermes`, `direct`, `remote`, `setup`, `status`, `stop`, `report`, `compare`, `resume`, and `fork`.
For example, `marathon --instance gpu23 resume` sees only that instance's Codex sessions and starts only that instance's backend.

Named writable data uses `instances/NAME/` below each normal root.
This includes the runtime lock and session, logs, traces, generated model catalog, router state, slot prompt cache, remembered selection, and Marathon Codex home.
The default instance never moves, so existing installs and scripts retain their original behavior.

## Prompt Prefix Cache

Marathon enables llama.cpp prompt caching and keeps completed prompt prefixes available when a new conversation starts.
llama.cpp compares the complete token stream and reuses only the exact common prefix, so edits to the system prompt, tools, plugins, skills, project context, or `AGENTS.md` automatically reprocess the changed suffix.
No manual invalidation or cache clearing is required.

The default host-memory cache is 8 GiB.
Set `MARATHON_PROMPT_CACHE_RAM_MIB` or override `prompt_cache_ram_mib` in the personal catalog settings to change the limit.
Exiting Marathon frees the GPUs and memory cache.

Marathon also saves the stable system-and-tools prefix under `~/AI/cache/marathon/slots/`.
After a cold backend start, the first conversation restores an exact matching disk snapshot instead of processing that prefix again.
The cache fingerprint includes the model, projector, backend binary and arguments, instructions, and tools, so incompatible changes build a new snapshot automatically.
Marathon retains up to eight starter snapshots within a 16 GiB default limit.
The optional `MARATHON_STARTER_CACHE_MAX_COUNT` and `MARATHON_STARTER_CACHE_MAX_BYTES` environment variables adjust those limits.

Rolling conversation checkpoints are separate from the starter cache and are enabled by default for llama.cpp backends with slot support.
Marathon waits for 60 seconds of conversation inactivity before saving, which keeps the large disk write off the response path.
The first checkpoint starts at 16,384 context tokens, and another save is needed only after at least 4,096 tokens of growth or a clean shutdown.
Each conversation atomically replaces its previous checkpoint instead of accumulating one file per response.
Each Marathon instance keeps at most two recent conversations, while all instances share a hard 32 GiB ceiling.
An inactive checkpoint expires 48 hours after its last save or restore.
Cleanup runs at startup, after saves, and periodically while Marathon remains open.
Checkpoint metadata stores only hashed conversation and response identities, compatibility fingerprints, token counts, sizes, and timestamps.
The binary checkpoint contains model KV state derived from the conversation and is created with owner-only permissions.
Set `MARATHON_SLOT_SNAPSHOTS_ENABLED=0` or `slot_snapshots_enabled = false` to disable rolling checkpoints without deleting Codex session history.

The following environment variables have matching keys in the personal catalog's `[settings]` table:

- `MARATHON_SLOT_SNAPSHOT_MAX_COUNT` controls recent conversations per instance and defaults to `2`.
- `MARATHON_SLOT_SNAPSHOT_MAX_BYTES` controls the shared byte budget, is hard-capped at 32 GiB, and defaults to that ceiling.
- `MARATHON_SLOT_SNAPSHOT_TTL_SECONDS` controls inactivity expiry and defaults to `172800`.
- `MARATHON_SLOT_SNAPSHOT_IDLE_SECONDS` controls the background-save delay and defaults to `60`.
- `MARATHON_SLOT_SNAPSHOT_MIN_TOKENS` controls the first-save threshold and defaults to `16384`.
- `MARATHON_SLOT_SNAPSHOT_MIN_TOKEN_GROWTH` controls rolling-save frequency and defaults to `4096`.

If a checkpoint is missing, expired, too large, corrupt, or incompatible, Marathon safely falls back to the starter cache and normal prompt replay.

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

Marathon reads the embedded GGUF name, architecture, and trained context from the first shard and caches that inspection by file identity.
This lets renamed models match architecture families without repeatedly scanning unchanged files.
Unrecognized GGUF models still use a conservative automatic 32K profile.
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
Marathon gives Codex a separate writable home at `.marathon/codex-home` inside the Marathon installation.
Configuration, sessions, logs, history, memory, and SQLite state therefore cannot modify or appear in stock Codex.

Marathon refreshes a generated `marathon-shared` profile from the user's normal Codex configuration before each launch.
It excludes the model, provider, reasoning effort, context, and catalog keys that Marathon owns.
Authentication, `AGENTS.md`, hooks, rules, skills, and plugins are explicitly linked into the isolated home so the same user tools remain available.
Project `.codex/config.toml` files continue to apply through normal Codex configuration precedence.

On the first isolated launch, existing `marathon-local` rollout files are moved from the stock Codex session tree into Marathon's session tree.
Set `MARATHON_CODEX_HOME` to relocate the isolated home or `MARATHON_STOCK_CODEX_HOME` when the normal Codex home is not `~/.codex`.
`MARATHON_USE_USER_CONFIG=1` is an explicit compatibility escape hatch that disables this isolation.
`MARATHON_CODEX_BIN` selects both the patched Codex install destination and the executable Marathon launches.

Marathon supplies invocation-specific settings for the local provider, selected model, model catalog, and status line.
After backend startup, Marathon reads the context that was actually loaded and records that value with the active model's compaction and truncation limits in the generated catalog.
This keeps context limits correct when `/model` switches between deployments with different windows.

The patched status line includes:

- Live estimated generation throughput with a `~` prefix.
- Exact completed-turn output tokens divided by active generation time.
- The active reasoning effort.
- Context usage based on the backend's actual loaded window.

Tool execution time is excluded from completed-turn generation throughput.

Use Codex's `/model` menu to change the active reasoning effort without reloading the GGUF or discarding the conversation.
Supported values are defined per model family in the runtime catalog.

## External OpenAI-Compatible Models

Marathon can include optional Responses API models in the same Codex `/model` menu as the active local model.
External models are configured only in the machine-local catalog, so endpoint addresses and deployment-specific model names do not enter the public repository.
Marathon never starts, stops, or unloads these services, and switching to one leaves the supervised local backend warm.

Add an entry to `~/.config/marathon/catalog.toml`:

```toml
[[external_models]]
id = "remote-coder"
model = "provider-model-id"
display_name = "Remote Coder"
description = "Private OpenAI-compatible coding model"
base_url = "https://inference.example.net/v1"
api_key_env = "MARATHON_REMOTE_API_KEY"
context = 131072
temperature = 0.0
```

Export the referenced credential before starting Marathon:

```bash
export MARATHON_REMOTE_API_KEY="your-private-key"
marathon
```

Do not place the key itself in the catalog.
To avoid exporting the key globally, set `api_key_file` to a private file containing either the bare key or the named `api_key_env=value` entry.
The endpoint must expose `/v1/models` and the Responses API, including function tool calls for agentic coding.
The optional `auto_compact_token_limit`, `truncation_limit`, `supports_parallel_tool_calls`, and `input_modalities` fields override Marathon's conservative defaults when the deployment has been validated for them.
Set `enabled = false` to retain an entry without showing it in the model menu.

Marathon sessions remain in Marathon's isolated Codex session store and are tagged with the `marathon-local` provider.
The resume picker also filters on that provider, and Marathon-branded sessions print `marathon resume <id>` when they exit.
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

The catalog retains specialized DeepSeek V4 profiles.
These are advanced paths and are not part of the default Qwen setup.

DeepSeek V4 profiles expect the `ds4-longctx` llama.cpp fork under:

```text
~/AI/backends/llama.cpp-ds4-longctx/build-cuda/bin/llama-server
```

The MTP profiles also require the matching MTP GGUF beside the main model or an explicit `DSV4_MTP_GGUF` path.
Run `marathon doctor` to identify a missing fork, worker, or sidecar before starting the model.

The DwarfStar safe profile uses three workers and one coordinator across four GPUs.
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
| Marathon Codex home | `<marathon-install>/.marathon/codex-home/` |
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
marathon search check
marathon search logs
marathon search down
```

The first start generates a local secret in `docker/searxng/.env`.
`search up`, `search restart`, and `search status` include a real search probe so upstream engine failures are visible immediately.
The probe queries Google CSE directly and fails if only fallback providers are working.
The bundled Google CSE engine uses SearXNG's public endpoint and has no user API key or numeric quota endpoint to inspect.
Marathon therefore verifies live Google CSE contribution and surfaces 429, suspension, and fallback details in search tool output.
The bundled configuration keeps only Google CSE, Bing, and Wikipedia, with Google CSE weighted first for technical and documentation queries.
The search tool accepts an optional `time_range` of `day`, `week`, `month`, or `year`.
Result deduplication ignores fragments, common tracking parameters, scheme and `www` aliases, query ordering, and trailing slashes while retaining semantic query parameters.
Set `MARATHON_SEARXNG_URL` when the service runs somewhere else.
Set `MARATHON_WEB_SEARCH_MODE=disabled`, `cached`, or `live` to control Codex's web-search mode.
Set `MARATHON_WEB_SEARCH_RETRIES` from `0` through `3` to control retries for transient SearXNG transport and server failures.
Set `MARATHON_WEB_BROWSE_ENABLE=0` to hide the optional browser-rendering path.

Fetch blocks loopback and private-network targets unless `MARATHON_WEB_FETCH_ALLOW_PRIVATE=1` is explicitly set.

## Updating Codex

Run this command to fetch the stable Codex tag recorded in `config/codex.ref`, preflight Marathon's patches, run regression tests, build a release binary, and install it atomically:

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
