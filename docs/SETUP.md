# Setup help

Start with the quick-start command in the [README](../README.md).
This page is for troubleshooting, not a second installation procedure.

## First launch

Marathon prepares its private Python environment, offers model selection, and offers to build missing inference and frontend binaries.
Downloads and builds can take a while; subsequent launches reuse them.
Prebuilt installers are not published yet, so the current installation requires build tools.
Source builds also need the platform's OpenSSL development headers and libraries.
On Ubuntu, install `build-essential cmake pkg-config libssl-dev bubblewrap`; install Rust with rustup and the CUDA toolkit separately for NVIDIA acceleration.
Linux also needs working user namespaces for the tool sandbox; restrictive containers may block them even with bubblewrap installed.
Marathon reports sandbox problems without disabling the sandbox.
Marathon bootstraps its own pinned, checksum-verified pip when the system Python lacks it, including on Ubuntu without `python3-venv`.
Normal installation does not require `just`, `cargo-nextest`, or running the developer test suite.
Backend builds omit llama.cpp's optional browser UI and its unrelated asset downloads.

The general Qwen download offers several quantizations.
Q4_K_M is the general starting point; larger quantizations need more memory and storage.
Downloads record the selected repository revision and expected checksum beside the model.
The optional tuned bundle additionally verifies the actual SHA-256 of every downloaded file.

Existing models can be registered without copying them:

```bash
marathon models add /path/to/models
```

## NVIDIA setup

`nvidia-smi` should recognize your GPU, and `nvcc --version` should recognize the CUDA compiler for a source build.
A working GPU driver alone does not include the compiler.
If NVIDIA tools are present but the compiler is missing, Marathon stops with an explanation instead of silently building a CPU-only runtime.
An explicitly selected CPU build remains available through `MARATHON_GPU_BACKEND=cpu marathon setup-llama`.

The optional 196K preset requires Linux, an RTX 3090 or 3090 Ti, and at least 23,500 MiB of free VRAM.
Desktop applications can consume enough VRAM to make that preset unavailable.
Use Automatic instead, or close applications yourself; Marathon never evicts another workload.
The preset selects one eligible card and leaves power settings unchanged.
The 3090-specific source build targets CUDA architecture 8.6.
Other GPUs use the general runtime, subject to their driver and compiler compatibility.

## Storage and memory

The tuned bundle downloads approximately 16.2 GiB: target weights, draft weights, and the vision projector.
Setup also reserves 2 GiB of download headroom.
Build output and optional prompt caches need additional space.
The frontend is a large Rust build; allow tens of gigabytes of additional disk space for its reusable `.marathon/codex-target` build cache.
Long-context inference needs host RAM as well as VRAM; the vision projector runs on the CPU in the tuned preset.
Model file size alone is not a memory-fit guarantee.

| Data | Default location |
| --- | --- |
| Models, backends, and caches | `~/AI/` |
| Personal configuration | `~/.config/marathon/` |
| Operational logs and reports | `~/.local/state/marathon/` |
| Isolated Codex sessions | `.marathon/codex-home/` inside the checkout |

Set `MARATHON_AI_ROOT` before setup to use another disk.
Leave your personal paths and credentials out of the repository.

## If something fails

Run `marathon doctor` and address the missing prerequisite it reports.
The launcher configures PATH for Bash, Zsh, and Fish when needed, preserving a backup of existing startup files.
Open a new terminal afterward; the shell that launched setup can still use `./bin/marathon` from the checkout.
Set `MARATHON_CONFIGURE_SHELL=0` before installation to keep shell startup files untouched.
An unavailable GPU or occupied port is reported without stopping its owner.
No model is launched after a tuned-bundle checksum failure.
An interrupted download can be retried by reopening setup.

When reporting a problem, include the OS, GPU, driver version, selected model/profile, and redacted error.
Do not attach credentials or private session files.
