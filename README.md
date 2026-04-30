# Marathon

Marathon is a local-model runtime layer for running OpenAI Codex against llama.cpp backends.

The design goal is to keep Codex upstream-friendly:

- `codex/` is the official OpenAI Codex repo as a git submodule.
- `patches/codex/` contains the small local patches Marathon needs.
- `scripts/` owns model launchers, routing, benchmarks, and llama.cpp setup.
- `config/qwen_models.json` owns local model metadata outside Codex.

## Quick Start

```bash
git submodule update --init --recursive
./bin/marathon build
./bin/marathon setup-llama
./bin/marathon 128k --no-alt-screen
```

By default Marathon looks for models under `~/models`.

Expected 27B path:

```text
~/models/Qwen3.6-27B-GGUF/qwen3.6-27b-q4_k_m.gguf
```

Expected 35B A3B path:

```text
~/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf
```

You can override paths:

```bash
QWEN36_27B_GGUF=/path/to/model.gguf ./bin/marathon 128k
QWEN36_35B_A3B_GGUF=/path/to/model.gguf ./bin/marathon a3b
```

## Commands

```bash
./bin/marathon 128k          # default long-context Qwen3.6 27B profile
./bin/marathon fast          # faster 32K Qwen3.6 27B profile
./bin/marathon a3b           # Qwen3.6 35B A3B profile
./bin/marathon exec "..."    # headless Codex exec
./bin/marathon sweep-128k    # focused config sweep
```

## Updating Codex

```bash
git -C codex fetch origin
git -C codex checkout main
git -C codex pull --ff-only
./scripts/apply_codex_patches.sh
./scripts/build_codex.sh
```

If a patch fails, Codex changed in the touched area. Rebase the patch intentionally instead of editing Codex ad hoc.
