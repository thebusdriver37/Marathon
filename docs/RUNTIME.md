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
The production runtime was built with CUDA 12.4 for SM86 (CUDA architecture 86, GGML_NATIVE=OFF).
The optional build omits llama.cpp's browser UI and disables its build-time UI downloads; Marathon uses its own terminal frontend.
It uses a separate backend directory and a patch-identity-specific worktree.
Re-running patch setup checks the complete patch stack and preserves local edits instead of force-removing worktrees.
Old worktrees are retained for manual review, not automatically deleted.

## Deployed source provenance

The base is upstream llama.cpp v1.0.0, `b1b770d90d94cad9e9ea5b993bdbc383a522fb28`.
The eight patches reproduce, file for file, the source of the production Marathon worker image of 2026-11-14:

1. `001-media-speculation-guard.patch`: no speculative drafting for requests with images on unsupported paths.
2. `002-iq4-xs-verify-kernel.patch`: IQ4_XS matrix kernel for 2-8 row verify batches (Marlin-style, SM86).
3. `003-sm86-q4-attention.patch`: integer q4_0 KV flash attention for long-context decode on SM86.
4. `004-sm86-gdn-geometry.patch`: gated delta-net kernel geometry tuned for SM86.
5. `005-kernel-regression-cases.patch`: `test-backend-ops` cases for the kernels above.
6. `006-wider-q4-batches.patch`: the q4_0 attention path for wider query batches.
7. `007-checkpoint-buffer-reuse.patch`: recycles exclusively owned recurrent-checkpoint buffers.
8. `008-tree-verify.patch`: tree verification for the DFlash2 drafter.

The kernels are opt-in by environment; the Qwen backend in `config/runtime_catalog.toml` sets the same values as the
production image: `GGML_CUDA_IQ4XS_MARLIN=1`, `GGML_CUDA_MARATHON_Q4_INT8=1`, `GGML_CUDA_MARATHON_GDN_TPC=16`,
`GGML_CUDA_MARATHON_Q4_MIN_CONTEXT=16384`, `GGML_CUDA_MARATHON_Q4_MAX_QUERIES=256`, `LLAMA_REUSE_CHECKPOINT_BUFFERS=1`.

### Tree verification

With `LLAMA_SPEC_TREE_NODES=7` and `LLAMA_SPEC_TREE_TEMP=2.0` the DFlash2 drafter's candidate lattice becomes a
best-first tree of 7 guesses (scores divided by the temperature first, because the drafter is overconfident) instead of
one chain of 7, verified in the same 8-row target batch, so a round costs the same. The target samples at the root and
follows the child that drafted that token; the accepted path is kept in the KV cache and recurrent state, every other
row is dropped. Sampling is unchanged. Unset or 0 keeps the chain. It is used only with one slot (`--parallel 1`);
n-gram drafts still use the chain path.

Measured on RTX 3090s at 250 W with the C3 drafter, on held-out sessions: +5.1% tokens per round on identical text,
42.1 ms per round against 43.0 ms for the chain; restored slots continue identically to a cold prefill; no change in
decode speed at 180K context. With greedy sampling the text can differ from the chain's after some tokens: the KV
cells of a tree-built conversation sit in a different order, which changes floating-point summation order (mean
absolute log-probability difference 0.003 against a cold prefill).

Old snapshots: the drafter state is saved with each slot, so a Marathon pool that changes drafter or turns the tree on
should use a new `cache_id`.

For a llama-swap pool, Marathon starts the selected worker and local router concurrently, then opens Codex while the worker finishes loading.
The router serializes the readiness check so an immediate first prompt waits for that same load safely.
In two cold paired runs on the local pool, this moved the Codex Ready screen from an average 11.12 seconds after runtime start to 1.24 seconds and reduced the immediate first reply from 16.39 seconds to 14.50 seconds.

These are llama.cpp source changes, not a redistribution of model weights or personal caches.
The upstream license remains with the fetched source.
Review the licenses of each model repository separately before redistributing weights.

## CPU-checked DFlash host-staging candidate

GPU follow-up on 2026-10-31 is complete: **do not promote this candidate**. Warm decode throughput averaged 0.49% lower across 8K, 32K, 128K and 186K occupied context, with matching token sequences. Original application/lifecycle checks and a frozen-source application rerun passed. The patch remains experimental; no production runtime or broker configuration changed, and no further testing is planned for this candidate. See [the result record](host-staging-gpu-20261031.json) for scope, exclusions, source consistency and privacy notes.

`scripts/experiments/dflash-host-staging.patch` is an experimental follow-up to
the 14 default Qwen patches. It is outside `patches/llama.cpp/qwen38/`, so setup
and ordinary builds do not select it. No production runtime, model, power
setting, broker, or existing worker was changed during this review.

The patch reuses DFlash's per-sequence indexing vectors, uses the injection
batch's position storage for the small encoder call, and sizes the small
injection batch to `n_ubatch` instead of `n_batch`. Every injection chunk is
already bounded by `n_ubatch`. The encoded embedding allocation therefore
shrinks from `n_batch * n_embd_dec * sizeof(float)` to
`n_ubatch * n_embd_dec * sizeof(float)`, with corresponding reductions to the
position and sequence metadata. This is host memory, not a KV-cache or VRAM
reduction. The patch changes no CUDA kernel, arithmetic, draft policy, context
length, or cache serialization.

This follows the allocation and scheduling review prompted by
[Elpis](https://github.com/gildrb/elpis/tree/13e8b593ea4c65f605b5fe7ac7d5ebb7a24f3c19).
Elpis's power-of-two scratch allocation is specific to its ExLlamaV3 path;
Marathon does not have that same allocation to patch.

CPU validation on 2026-10-29 used the pinned llama.cpp base plus all 14 default
patches, fetched into a new temporary source checkout. The candidate passed
C++17 syntax checking and 6,400 paired host-staging and draft-batch cases under
AddressSanitizer and UndefinedBehaviorSanitizer. The checker extracts the real
before/after method bodies and compares encoder/decoder inputs through API
doubles. Cases exercise repeated calls, multiple sequences, microbatch tails,
the fused-injection threshold, M-RoPE position jumps, image guards, and encoder
and decoder failures. It neither loads a model nor links a GPU backend.

Reproduce against a source checkout containing the default patches, without the
candidate applied:

```bash
python3 scripts/experiments/check_dflash_host_staging.py \
  --source /path/to/patched-llama.cpp
```

The source argument is read-only. The checker applies the candidate and compiles
one test executable in its own temporary directory. It prints source and patch
hashes with the result. It requires Git and a C++17 compiler with ASan/UBSan.
These checks establish host-side equivalence within the fixture; they do not
establish model-output equality, GPU correctness, or a throughput improvement.
Before promotion, compare a separately built candidate with the current runtime
on synthetic text, tool continuations, media, and checkpoint restore, at both
short and long occupied context, using identical hardware/power and model files.

Tree verification remains unimplemented. The pinned DFlash interface emits a
linear token list. Supporting Elpis-style branches requires a tree descriptor
through target attention, parent-specific recurrent/GDN state, accepted-path
KV/state commit and rewind, and compatible media/checkpoint behavior. Changing
only the selector would not implement tree verification. Elpis's tree guarantees
are for its greedy path; Marathon's nonzero-temperature sampling also requires
its own design and validation. No tree or new attention-kernel option is enabled
by this candidate.

## Reproducibility boundary

The original production image was `a6b27f3fd9c59d2756e447d8cc35576a541fb1db61624f18681199e297b97561`, a local Docker image ID rather than a published registry digest.
Its CUDA library SHA-256 was `1509d63eed59f6d918971e39b1a2698a69733d6ffc0c3d3398458534865720d5`.
The scheduler-reuse library hash was `27e04097441046435180630b6c13daec608f9311311fbf6d658d0943a8ff709a`.
The server implementation hash was `8414d2e55ae5d54e140afe043aae2ae3d8444195c343402521b28c45c59f374e`.

The original machine used 250 W for RTX 3090 and 275 W for the inference RTX 3090 Ti.
Those power caps are provenance, not installation actions or requirements.
Per-session inference used one GPU, not a combined multi-card model.

The local media candidate image is `3dadf9fed7adc27a4264c23c177e68c5dcb2a269cd88360aa3e5b41185e1e193`.
Its CUDA library SHA-256 is `913386bd1442b39013b891c9d5cc9751360d5f460a3817ac3d2be3d15eff18bc`.
On one RTX 3090 at 250 W, with 196K allocated context, temperature zero, and seed 424242, representative backend decode timings were:

| Test | Production tok/s | Candidate tok/s |
| --- | ---: | ---: |
| Short text, 114 output tokens | 70-71 | 71-72 |
| One image, 160 output tokens | 33 | 51-53 |
| Two images, 384 output tokens | 33 | 56 |
| 80K occupied context plus image, 384 output tokens | 19 | 21 |

Those historical exact-arithmetic measurements used an image window of three; lowering it to one reached about 23 tok/s in the 80K test.
Cold prefill for that 80K prompt remained about 108 seconds, so no prefill improvement is claimed.
These fixture results are not universal throughput guarantees or a claim of 90-100 tok/s with images.

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

## Media decode repeatability check

Reserve an idle worker exclusively before running this check; it replaces that worker's in-memory cache.
The script does not manage workers, change routing, or save persistent snapshots.
It requires the patched runtime, its slots endpoint, and a loopback server or broker upstream URL without `/v1`.
Credentials come only from the named environment variable, never command-line key values.

```bash
python3 scripts/experiments/check_media_decode.py \
  --base-url http://127.0.0.1:8080 \
  --model qwen3.8-27b-uncensored \
  --image /path/to/test-image.jpeg
```

Repeat `--image` for multi-image checks and use `--prompt-file` for occupied-context tests.
The default 768-token output exercises cache boundaries as well as short-answer equality.
The checker compares raw token IDs, complete messages, and stop reasons between cold and warm requests at the same window, including the follow-up.
It defaults to six proposals; use `--window` for another configured limit.
It does not assert equivalence to zero-draft arithmetic or replace quality evaluation and baseline-versus-candidate tests.
It requires speculation to activate and prints hashes and backend timings without logging response text or token IDs.
Also test saved-slot recovery through the actual application's checkpoint workflow before deploying a new runtime.
