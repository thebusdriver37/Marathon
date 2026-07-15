# Marathon

Marathon is a one-command, terminal-first local AI runtime. It discovers GGUF
models in the centralized AI directory, starts llama.cpp and Marathon's local
API router in the foreground, then opens either Codex or a clean Direct Chat.

The process showing the Marathon dashboard owns the backend. Exiting Marathon
stops its router and llama.cpp process groups and frees the GPUs. This lifecycle
works through an ordinary SSH shell; no exposed daemon or web UI is required.

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
```

Specific overrides take precedence over `MARATHON_AI_ROOT`; relative paths in
the runtime catalog are resolved beneath the effective AI root.

## Commands

```bash
marathon                  # dashboard; Enter starts remembered Codex setup
marathon codex            # dashboard with Codex selected
marathon direct           # dashboard with clean Direct Chat selected
marathon models           # list installed centralized GGUF models
marathon status           # inspect an active foreground runtime
marathon stop             # emergency stop request from another SSH shell
marathon doctor           # diagnose setup, GPUs, models, and ports
marathon search up        # optional local SearXNG for Codex web tools
```

The old `marathon backend ...` commands remain temporarily for compatibility,
but the dashboard does not use their detached-process lifecycle.

## Codex behavior

Marathon launches the installed stock `codex` command and leaves `CODEX_HOME`
untouched. Global configuration, installed skills, plugins, session history,
and normal global/repository/nested `AGENTS.md` discovery therefore work as
they do in Codex itself. Marathon supplies only per-invocation overrides for
its local provider, selected model, context window, and generated model catalog.

## Direct Chat

Direct Chat sends a streaming Chat Completions request through the local
router. It deliberately provides no tools, coding-agent prompt, AGENTS files,
skills, memory, or Hermes harness. Use `/new` to clear the conversation and
`/back` to return to the warm dashboard.

## Runtime storage

- Models and backends: `~/AI/`
- Slot cache and router state: `~/AI/cache/marathon/`
- Remembered selection: `~/.config/marathon/selection.json`
- Logs: `~/.local/state/marathon/logs/`
- Live lock and PID metadata: `$XDG_RUNTIME_DIR/marathon/`

## Diagnostics and Model Checks

Use `marathon doctor` when setting up a new machine or debugging a failed run.
It checks stock Codex, the centralized llama.cpp backend and models, Marathon's
private Python environment, GPU visibility, ports, and optional SearXNG.

## Prompt Cache Snapshots

Linear follow-up turns reuse llama.cpp's live prompt slot, so normal Codex
conversations do not write large prompt snapshots to disk. Optional disk
snapshots can preserve recent in-process branches, but each snapshot can be
hundreds of megabytes at long context and adds synchronous I/O after a turn.
They are therefore disabled by default.

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
| `MARATHON_WEB_SEARCH_MODE` | `cached` | Forwarded to Codex's `web_search` config: `disabled`, `cached`, or `live` |
| `MARATHON_WEB_FETCH_TIMEOUT` | `25` | Per-fetch timeout (seconds) |
| `MARATHON_WEB_FETCH_MAX_CHARS` | `20000` | Default cap on extracted-content chars per fetch |
| `MARATHON_WEB_FETCH_MAX_CHARS_CAP` | `200000` | Hard cap on what the model can request |
| `MARATHON_WEB_FETCH_MAX_BYTES` | `5 MiB` | Hard cap on raw response size |
| `MARATHON_WEB_FETCH_USER_AGENT` | Marathon-identifying Mozilla UA | UA the fetch executor sends |
| `MARATHON_WEB_FETCH_ALLOW_PRIVATE` | unset | Set to `1` only if fetches to loopback/private network URLs are intentional |
| `MARATHON_WEB_BROWSE_ENABLE` | `1` | Set to `0` to hide the optional Crawl4AI-backed `web_browse` tool path |
| `MARATHON_ROUTER_PYTHON` | `.marathon/venv/bin/python3` | Python interpreter the launcher uses for the router |

## Experimental Codex fork

Ordinary Marathon use prefers the installed stock Codex binary. The submodule
and patch workflow remain available for experiments that demonstrate a real
upstream gap; they are not required to start Marathon.

To update that experimental fork:

```bash
git -C codex fetch origin
git -C codex checkout main
git -C codex pull --ff-only
./scripts/apply_codex_patches.sh
./scripts/build_codex.sh
```

If a patch fails, Codex changed in the touched area. Rebase the patch intentionally instead of editing Codex ad hoc.
