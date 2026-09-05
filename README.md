# Marathon

A local coding agent in your terminal.
Marathon connects a customized Codex frontend to GGUF models running on your own hardware.

## Get started

With the prerequisites below installed, clone Marathon and start setup in one command:

```bash
git clone https://github.com/thebusdriver37/Marathon.git && cd Marathon && ./bin/marathon
```

The setup menu lets you download a model or use one you already have.
Marathon remembers your choice, starts inference when needed, and releases its GPUs when you exit.
No configuration editing is required for the normal setup.

After setup, open a new terminal and run `marathon` from the project you want to work on.

## What you need

- Linux is the tested local setup platform.
- Python 3.10+, Git, and enough storage for your model.
- For the current source installation: CMake, a C++ compiler, Rust/Cargo, pkg-config, and OpenSSL development libraries.
- On Linux: bubblewrap for sandboxed tools.
- For NVIDIA acceleration: a working driver and CUDA toolkit with `nvcc`.

**Prebuilt installers are not published yet.**
Setup can build the missing runtimes, but it cannot install system drivers or eliminate the source-build requirements.
See [setup help](docs/SETUP.md) if a prerequisite is missing.

## Models and hardware

Qwen 3.8 27B is the suggested starting point.
You can also select another GGUF or Hugging Face repository.
The general Automatic profile fits context and GPU placement to available memory.

On an eligible, mostly free RTX 3090 or 3090 Ti, setup also offers an optional bundle:

- Qwen 3.8 27B Uncensored IQ4_XS and its DFlash2 drafter.
- 196K requested context, Q8 target KV cache, and CPU-backed vision.
- The custom runtime patches and portable profile included in this repository.

The bundle pins and verifies its model downloads.
The original deployment was tested on 3090-class hardware; performance of newly rebuilt binaries still requires release validation.
Other hardware keeps the general setup path, without untested 3090-specific tuning.
No power limits are changed automatically.
See [runtime details and reproducibility](docs/RUNTIME.md).

## Local by default

Inference runs locally unless you explicitly configure a remote model.
The hardened frontend disables background cloud telemetry, announcements, and update checks.
Downloading models and source code requires internet access; explicitly enabled web tools can also use the internet.
This is application-level hardening, not an operating-system network sandbox.

## Everyday commands

| Command | What it does |
| --- | --- |
| `marathon` | Open your remembered model |
| `marathon setup` | Add or change models |
| `marathon resume` | Continue a saved conversation |
| `marathon dashboard` | Open advanced controls |
| `marathon doctor` | Diagnose setup problems |
| `marathon report` | Show measured performance |

The footer reports backend decode speed separately from first-output latency.
Speed depends on context, workload, hardware, and speculative acceptance; there is no single guaranteed tok/s number.

## More

- [Setup and troubleshooting](docs/SETUP.md)
- [Advanced usage](docs/ADVANCED_USAGE.md): instances, caching, tools, remote hosts, and tuning
- [Runtime packaging](docs/RUNTIME.md): source patches, model pins, and release checks
- [Development](docs/DEVELOPMENT.md)
