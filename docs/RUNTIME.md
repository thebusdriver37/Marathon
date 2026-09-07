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

Patch `011-media-single-column-verification.patch` introduced image-position repairs and an exact-arithmetic verification path for the single-GPU SM86, 27B IQ4_XS, Q8-KV configuration.
The tested image was selected for the machine-local production pool on 2026-09-05, with a new cache identity and separate snapshot directory.
Unsupported configurations retain the media speculation guard.
The loader supplies missing temporal-position metadata for the legacy DFlash2 file in memory, without changing its weights or requiring another model download.
Draft generation starts at the model's next position, not the number of prompt tokens; these diverge after images.
The target processes every image embedding; only the drafter omits raw image rows and prefill history outside every draft layer's sliding window.
Patch `013-batched-media-default.patch` supersedes the exact-arithmetic policy: text and images use normal batched verification and the configured draft window, six in the local profile.
The tested padded-KV-boundary and cache-layout guards remain for both text and media.
There is no user-facing slow/fast mode or runtime diagnostic control-file requirement.
`speculative.n_max` can lower the per-request limit, including zero for a reference run, but cannot exceed the configured limit.
Changing the drafter's position layout requires a new external backend `cache_id` and a coordinated router restart; do not reuse old draft snapshots under the old identity.

The upstream [image-position report](https://github.com/ggml-org/llama.cpp/issues/27408) and [quantized CUDA batch-numerics report](https://github.com/ggml-org/llama.cpp/issues/27407) describe separate failure modes.
The local metadata-only test accepted 12 of 861 draft tokens and decoded at about 25 tok/s.
Correcting the draft start position restored about 67 tok/s, but changed the greedy output, so that configuration was initially rejected under an exact-output requirement.
The current default does not require batched speculation to reproduce zero-draft wording bit for bit; cleanup regressions are compared against the known-working batched runtime instead.

Patch `012-media-attention-reuse.patch` reuses K/V work across two verification queries while preserving the original single-query reduction partition.
It passed bitwise attention checks, image and follow-up token checks, multiple images, 80K occupied context, and saved-slot recovery on the tested SM86 configuration.
A reconstructed 17.5K-context screenshot conversation improved from about 40 to 44 tok/s, with unchanged tokens and draft acceptance; ordinary text and prefill were essentially unchanged.
This is a partial attention improvement, not elimination of the media decode slowdown, and it has not been deployed to the local production pool.

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
