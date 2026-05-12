# Marathon

Marathon is a local-model runtime layer for running OpenAI Codex against llama.cpp backends.

The design goal is to keep Codex upstream-friendly:

- `codex/` is the official OpenAI Codex repo as a git submodule.
- `patches/codex/` contains the small local patches Marathon needs.
- `scripts/` owns model launchers, routing, benchmarks, and llama.cpp setup.
- The router owns local model metadata and exports it to Codex at launch time.

## Quick Start

```bash
git clone --recurse-submodules <marathon-repo-url>
cd Marathon
./bin/marathon install        # makes `marathon` available from any repo
git submodule update --init --recursive
marathon build
marathon setup-llama
marathon setup-deps           # creates .marathon/venv for router Python deps
marathon search up            # optional: brings up SearXNG for web_search/web_fetch tools
marathon backend start 128k-single
marathon doctor               # verify setup, backend health, GPU, models, ports, and cache usage
```

After `marathon install`, run Marathon from the project directory you want
Codex to edit:

```bash
cd /path/to/your/repo
marathon
```

If your shell cannot find `marathon`, make sure `~/.local/bin` is on `PATH`.
The install command prints the exact `export PATH=...` line when needed.

By default Marathon looks for models under `~/models`.

Expected 27B path:

```text
~/models/Qwen3.6-27B-GGUF/qwen3.6-27b-q4_k_m.gguf
```

Expected 35B A3B path:

```text
~/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf
```

Expected Qwopus path:

```text
~/models/Qwopus3.6-35B-A3B-v1-GGUF/Qwopus3.6-35B-A3B-v1-Q4_K_M.gguf
```

Expected Gemma 4 26B A4B path:

```text
~/models/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-Q4_K_M.gguf
```

You can override paths:

```bash
QWEN36_27B_GGUF=/path/to/model.gguf marathon backend start 128k-single
QWEN36_35B_A3B_GGUF=/path/to/model.gguf marathon backend start a3b
QWOPUS36_35B_A3B_GGUF=/path/to/model.gguf marathon backend start qwopus
GEMMA4_26B_A4B_GGUF=/path/to/model.gguf marathon backend start gemma
```

### Custom GGUF Models

Marathon can also run any local GGUF model through llama.cpp without editing
the repo:

```bash
marathon backend start custom /path/to/model.gguf
```

Optional metadata and llama.cpp tuning:

```bash
MARATHON_MODEL_SLUG=local-coder \
MARATHON_MODEL_DISPLAY_NAME="Local Coder" \
MARATHON_MODEL_CONTEXT=32768 \
MARATHON_LLAMACPP_ARGS="--jinja" \
marathon backend start custom /path/to/model.gguf
```

You can also set `MARATHON_MODEL_PATH=/path/to/model.gguf` and run
`marathon backend start custom`. Custom models still use llama.cpp and
Marathon's slot-save path, so `previous_response_id` resume behavior continues
to work the same way as the built-in Qwen profiles.

## Commands

```bash
marathon backend start 128k-single   # start router + llama.cpp backend
marathon backend start qwopus         # start Qwopus3.6 35B A3B v1
marathon backend start gemma          # start Gemma 4 26B A4B IT
marathon backend start custom ./model.gguf
marathon backend status              # show backend health
marathon backend logs -f             # follow router logs
marathon backend stop                # stop router + llama.cpp backend
marathon doctor                      # diagnose setup and runtime health
marathon eval coding                 # run a temporary coding gauntlet against the active model
marathon                             # launch the TUI from the current repo
marathon exec "..."                  # headless Codex exec from the current repo
marathon search up                   # bring up the SearXNG container
marathon sweep-128k                  # focused config sweep
```

The backend is intentionally explicit. Start it once, leave it running, then
open Marathon from any repo with `marathon`. The frontend does not start or
switch model servers on its own.

## Diagnostics and Model Checks

Use `marathon doctor` when setting up a new machine or debugging a failed run.
It checks the patched Codex binary, llama.cpp server binary, router Python venv,
Python, Docker, GPU visibility, known model paths, router/backend health,
SearXNG reachability, and local snapshot/log disk usage.

Use `marathon eval coding` to sanity-check a model before recommending it. The
eval creates a temporary repository under `/tmp`, asks the active Marathon model
to fix a small Python package, runs the included unit tests, and prints the
result, duration, token count when available, logs, and diffstat. It is designed
to catch the practical failure modes that matter for agentic coding: no edits,
broken shell commands, corrupted files, and tests that do not go green.

## Prompt Cache Snapshots

Marathon uses llama.cpp slot save/restore to emulate Codex's
`previous_response_id` behavior locally. These snapshots live under
`.marathon/llama-slots/` and are an ephemeral speed cache, not durable chat
history. Linear follow-up turns reuse the live llama.cpp slot directly; disk
snapshots are kept only as bounded fallback cache for recent branches.

At router startup Marathon deletes stale slot snapshots, because the in-memory
response lineage needed to use them does not survive a router restart. During
runtime it keeps each profile's snapshot directory capped by both count and
bytes so 128K sessions cannot silently fill the disk.

| Var | Default | Purpose |
|---|---|---|
| `MARATHON_SLOT_SAVE_ROOT` | `.marathon/llama-slots` | Root directory passed to llama.cpp `--slot-save-path` launchers |
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
./bin/marathon backend start 128k-single
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

## Updating Codex

```bash
git -C codex fetch origin
git -C codex checkout main
git -C codex pull --ff-only
./scripts/apply_codex_patches.sh
./scripts/build_codex.sh
```

If a patch fails, Codex changed in the touched area. Rebase the patch intentionally instead of editing Codex ad hoc.
