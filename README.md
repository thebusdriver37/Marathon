# Marathon

[![CI](https://github.com/thebusdriver37/Marathon/actions/workflows/ci.yml/badge.svg)](https://github.com/thebusdriver37/Marathon/actions/workflows/ci.yml)

Marathon is a terminal-first runtime for using Codex with local GGUF models.
It finds or downloads a model, starts the correct local backend, opens Codex, and frees the GPUs when you exit.

Qwen 3.8 27B is the recommended starting model.
Existing GGUF files and other Hugging Face repositories are also supported.

## Quick Start

```bash
git clone https://github.com/thebusdriver37/Marathon.git
cd Marathon
./bin/marathon
```

The first launch prepares Marathon's private Python environment and walks you through model setup.
No manual configuration file is required.
Marathon also installs its launcher at `~/.local/bin/marathon` when possible and prints exact PATH instructions when the folder is not already available in the shell.

You can:

- Use a GGUF model Marathon already found.
- Add an existing model folder without moving or copying anything.
- Download Qwen 3.8 27B and choose a quant.
- Download a GGUF from another Hugging Face `owner/repository`.

If llama.cpp or Codex is unavailable, Marathon offers to build the pinned compatible version.

After setup, run Marathon from the project you want Codex to edit:

```bash
cd /path/to/your/project
marathon
```

Marathon loads the remembered model and opens Codex directly.
Its writable Codex configuration and sessions stay separate from stock Codex while personal instructions, skills, plugins, hooks, and rules remain available.

## Choosing a Quant

For Qwen 3.8 27B:

| Quant | Approximate download | Best for |
|---|---:|---|
| `Q4_K_M` | 15.9 GiB | Most computers and the recommended first choice |
| `Q5_K_M` | 18.5 GiB | More quality when memory allows |
| `Q6_K` | 21.3 GiB | Higher quality with more memory use |
| `Q8_0` | 27.1 GiB | Maximum practical GGUF quality on large-memory systems |

Marathon requests up to 256K context for Qwen 3.8.
The automatic profile lets llama.cpp choose GPU placement and reduce context when the available memory cannot hold the full request.
Unknown model families use a conservative 32K default.

Downloads are resumable and pinned to the exact Hugging Face repository revision selected during setup.
Marathon records the repository, filename, revision, size, and expected SHA-256 beside the downloaded GGUF.
When a repository includes a compatible vision projector, Marathon downloads it beside the model and enables screenshot and image inspection automatically.
Existing models get the same behavior when their matching `mmproj` or `vision-f16` GGUF is stored beside the main model.
Marathon saves the token-exact system-and-tools prefix to disk after processing it once.
Later cold starts restore unchanged system instructions, tools, and `AGENTS.md` content after the model loads, while prompt changes automatically build a new cache.
No manual cache management is required.

## Reasoning Control

Open Codex's `/model` menu to change the reasoning level for the active session.

Qwen 3.8 supports:

- `None` for direct answers without model reasoning.
- `Low` for quick tasks.
- `Medium` for normal coding work.
- `XHigh` for difficult debugging and planning.

Changing the reasoning level does not reload the model or discard the active conversation.
Use `None` for routine work and raise it only when the task benefits from deeper reasoning.

## Existing Model Folders

Register as many existing model folders as needed:

```bash
marathon models add /mnt/models
marathon models add ~/Downloads/gguf
```

Marathon scans registered folders in place and does not copy their files.
Registered folders are stored in `~/.config/marathon/models.json`.

For temporary overrides, use an OS-path-separated list:

```bash
MARATHON_MODEL_DIRS=/mnt/models:/data/gguf marathon
```

`MARATHON_MODELS_DIR` remains available as a single-folder compatibility override.

## Common Commands

| Command | Purpose |
|---|---|
| `marathon` | Load the remembered model and open Codex |
| `marathon setup` | Find, add, or download a model |
| `marathon dashboard` | Open advanced model, profile, frontend, and tuning controls |
| `marathon models` | List discovered GGUF models |
| `marathon models add PATH` | Register an existing model folder |
| `marathon direct` | Open the clean Direct Chat frontend |
| `marathon hermes` | Open Hermes for compatible profiles |
| `marathon resume [ID]` | Resume a Marathon session or open its session picker |
| `marathon fork [ID]` | Fork a Marathon session or open its session picker |
| `marathon tune` | Benchmark machine-specific llama.cpp profiles with Dyno |
| `marathon doctor` | Check dependencies, models, backends, GPUs, and ports |
| `marathon status` | Inspect the active foreground runtime |
| `marathon stop` | Ask an active runtime to stop and free the GPUs |
| `marathon report` | Summarize the latest telemetry trace |
| `marathon --help` | Show every command and environment override |

## Advanced Dashboard

The normal `marathon` command skips Marathon's menu and opens Codex immediately.

Use the dashboard when you want to:

- Select a different installed model.
- Choose an exact runtime profile.
- Keep a loaded model warm while reopening frontends.
- Use Direct Chat or Hermes.
- Run Dyno tuning.

```bash
marathon dashboard
```

## Requirements

Marathon creates and maintains its own Python environment automatically.

A clean source installation needs:

- Python 3.10 or newer.
- Git.
- CMake and a C++ compiler for llama.cpp.
- Rust and Cargo for Marathon's patched Codex build.
- Enough disk space for the selected GGUF plus at least 2 GiB of download headroom.

CUDA acceleration is enabled automatically when `nvcc` is available.
Without CUDA, the pinned llama.cpp setup builds a CPU backend.

Prebuilt Marathon binaries are not published yet, so a completely fresh machine may need the build tools above.

## Optional Web Tools

Marathon can provide local web search and page fetching through SearXNG:

```bash
marathon search up
marathon search check
marathon
```

This optional feature requires Docker with the `docker compose` plugin.
The service binds to loopback by default.
Startup waits for a functional upstream search instead of treating the container health endpoint as proof that search works.
The probe requires Google CSE itself to return results, so a Bing-only fallback now fails `marathon search check` instead of looking healthy.
SearXNG's Google CSE adapter uses a public endpoint rather than a user API key, so Marathon verifies live provider availability and reports rate limiting instead of claiming to read a key quota counter.
Agents can request `day`, `week`, `month`, or `year` freshness filters, and tracking variants of the same result URL are collapsed automatically.

## Troubleshooting

Start with:

```bash
marathon doctor
```

Marathon reports missing dependencies, model folders, backend binaries, GPU visibility, port conflicts, and cache usage.

Useful storage locations:

| Data | Default location |
|---|---|
| Models and backends | `~/AI/` |
| Registered model folders | `~/.config/marathon/models.json` |
| Remembered selection | `~/.config/marathon/selection.json` |
| Personal catalog overrides | `~/.config/marathon/catalog.toml` |
| Runtime traces | `~/.local/state/marathon/runs/` |
| Runtime logs | `~/.local/state/marathon/logs/` |
| Persistent prompt cache | `~/AI/cache/marathon/slots/` |

Set `MARATHON_AI_ROOT` to move the complete models, backends, and cache hierarchy.

Personal runtime profiles and catalog edits belong in `~/.config/marathon/catalog.toml`.
That file is merged over the repository's `config/runtime_catalog.toml`, so machine-specific families, backends, and profiles never need to be committed.
Optional OpenAI-compatible external models can also be declared there and will appear in Codex's `/model` menu without changing the repository.
Set `MARATHON_USER_CATALOG` to point the override elsewhere.

## Runtime Safety

The foreground Marathon process owns its router and backend processes.
Exiting Marathon stops those processes and frees the GPUs.
Inference endpoints bind to loopback by default and are not exposed to the local network.

Runtime traces contain operational metadata such as timings, token counts, GPU measurements, tool names, and errors.
They do not intentionally record prompts, responses, tool arguments, tool outputs, or hidden reasoning.

## More Documentation

- [Advanced usage and architecture](docs/ADVANCED_USAGE.md)
- [Planned improvements](docs/FUTURE_UPDATES.md)
- [Runtime profiles](config/runtime_catalog.toml)

## Development

Run the test suite with Marathon's private environment:

```bash
./bin/marathon setup-deps
.marathon/venv/bin/python -m unittest discover -s tests -v
```

GitHub Actions runs shell syntax checks, Python compilation, and the complete unit test suite for every push and pull request.
