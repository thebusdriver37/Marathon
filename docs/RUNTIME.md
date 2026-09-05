# Runtime packaging

Marathon keeps setup, portable profiles, model manifests, and native runtime patches in one repository.
Users do not need a second fork or the maintainer's GPU-control configuration.
Machine-local broker routing remains outside the public defaults.

## Two paths

| Path | Source | Intended use |
| --- | --- | --- |
| General | Upstream commit in `config/llamacpp.ref` | Automatic fitting and other models/hardware |
| Optional Qwen bundle | Base in `config/llamacpp-qwen38.ref` plus `patches/llama.cpp/qwen38/` | Exact IQ4_XS bundle on eligible 3090-class cards |

The optional path never replaces the general backend.
Its model, drafter, and projector repositories, revisions, sizes, and SHA-256 values are tracked in `config/model_bundles.toml`.
Its launch settings are in `config/runtime_catalog.toml`, with no personal home paths or fixed physical GPU numbers.
The profile is shown only for its target filename and checks the complete bundle and hardware before launch.
Only the setup download verifies full file hashes; manual replacement of installed files is outside that verification.

The setup menu handles the bundle download and offers the appropriate build.
Maintainers can build its backend directly:

```bash
./bin/marathon setup-llama qwen38
```

That command requires a CUDA toolkit capable of compiling SM86.
The original runtime was built with CUDA 12.8.
The optional build omits llama.cpp's browser UI and disables its build-time UI downloads; Marathon uses its own terminal frontend.
It uses a separate backend directory and a patch-identity-specific worktree.
Re-running patch setup checks the complete patch stack and preserves local edits instead of force-removing worktrees.
Old worktrees are retained for manual review, not automatically deleted.

## Deployed source provenance

The base is upstream `9723942adc518b43c4b95dc4dce6906903eb5e09`.
The first seven patches reproduce the deployed private branch through `ca461b488709b74fb7a59211fa4c75b8643e5601`:

1. IQ4_XS crossover tuning: `8d812be82`.
2. SM86 speculative-decode optimizations: `8f02ccee1`.
3. Fused DFlash KV injection: `e64dfa178`.
4. Q8 attention V-tile swizzling: `bbede57ec`.
5. Fused Q8 V decode attention: `d43b67033`.
6. Disable speculation for media input: `f79bbc685`.
7. Persist speculative slot state: `ca461b488`.

Three following patches preserve the deployed cache preference, opt-in scheduler reuse, and recurrent rewind-checkpoint serialization.
The profile enables scheduler reuse through `LLAMA_REUSE_SCHEDULER=1`.
Rejected J32, asynchronous-copy, shared-store, and short-context attention experiments are not included.

These are llama.cpp source changes, not a redistribution of model weights or personal caches.
The upstream license remains with the fetched source.
Review the licenses of each model repository separately before redistributing weights.

## Reproducibility boundary

The original production image was `a6b27f3fd9c59d2756e447d8cc35576a541fb1db61624f18681199e297b97561`, a local Docker image ID rather than a published registry digest.
Its CUDA library SHA-256 was `1509d63eed59f6d918971e39b1a2698a69733d6ffc0c3d3398458534865720d5`.
The scheduler-reuse library hash was `27e04097441046435180630b6c13daec608f9311311fbf6d658d0943a8ff709a`.
The server implementation hash was `8414d2e55ae5d54e140afe043aae2ae3d8444195c343402521b28c45c59f374e`.

The original machine used 250 W for RTX 3090 and 275 W for the inference RTX 3090 Ti.
Those power caps are provenance, not installation actions or requirements.
Per-session inference used one GPU, not a combined multi-card model.

A prior fresh CUDA rebuild passed output checks but lost decode speed.
Consequently, matching source is not a promise of a byte-identical binary or the original throughput.
The packaged source must pass the release checks below before it is advertised as a performance-equivalent prebuilt runtime.

## Before publishing prebuilt downloads

- Build in a pinned toolchain and record the source, patch, binary, CUDA, and platform identities.
- Include required runtime libraries and licenses, never credentials, models, or conversation snapshots.
- Verify matching greedy output and draft acceptance against the deployed baseline.
- Benchmark cold prefill and sustained decode at several occupied context lengths.
- Verify cold-start, cache restore, media handling, shutdown, and frontend network hardening.
- Install on a clean host without the maintainer's personal configuration or local Docker images.
- Publish immutable, checksum-verified artifacts and wire setup to those exact artifacts.

Prebuilt downloads are not published or automatically fetched by the current source setup.
Do not add a download button pointing at an unvalidated or nonexistent release.
