# RTX 3090 kernel optimization trials, 2026-09-08

Seven scheduling, tile-size, and prefetch variants were screened after the [profiling session](GPU_PROFILING_2026-09-08.md).
None established an improvement worth deploying.
Production inference configuration, binaries, model files, power limits, and clocks were not changed.

The later [multi-prompt acceptance evaluation](SPECULATIVE_ACCEPTANCE_EVALUATION.md) tested the possibility that better speculative acceptance could outweigh slower matrix operations.
That broader evaluation found lower aggregate acceptance and slower responses for both promising scheduling variants.

## Isolation and comparison

GPU 2 was occupied when the investigation began, so trials used the free RTX 3090 on GPU 3 at its existing 250 W limit.
Every scratch run acquired Marathon's existing worker-3 pool lease and checked for conflicting broker workers.
Scratch workers used a loopback endpoint and separate temporary slot storage.
The broker configuration was never reloaded, and active workers on GPUs 1 and 2 were not restarted by this investigation.

The production image remained `sha256:443d87c87fe673faf379b29dd11bfa22e03c76e4d2fe14e9ce9503c810daf34c`.
Model quantization, 196K context capacity, KV types, draft window, and sampling settings matched the profiling record.
Inference measurements replayed real conversation text through the native chat endpoint, with seed 424242, temperature 1, and 256 generated tokens.
Each measured conversation group included four cached repeats after cold prefill.
This tests backend inference, not a fresh Marathon UI or compaction end-to-end suite.

An initial rebuilt CUDA library was about 3% slower with its experiment disabled.
Its saved sources lacked the production L1 preference, and its other rebuilt kernels also differed from the deployed artifact.
That build was excluded from candidate qualification.
Subsequent trials preloaded only the experimental IQ4 translation unit and resolved its dependencies against the production CUDA backend.
This retained production kernels for the other quantization types and attention.
The preload control reproduced the original output hashes and acceptance counts, with conversation decode within 0.2% of the initial production run.

## Conversation screening

| Variant | Cached decode, tokens/s | Cached request, ms | Accepted / drafted | Output matches production |
| --- | ---: | ---: | ---: | --- |
| Production, initial | 54.96 | 4974.5 | 158 / 577 | Yes |
| Production, final return control | 54.92 | 4968.5 | 158 / 577 | Yes |
| Isolated preload, experiment off | 54.90 | 4968.0 | 158 / 577 | Yes |
| Prefer shared memory | 53.15 | 5129.5 | 158 / 577 | Yes |
| Double blocks, prefer shared memory | 66.00 | 4190.5 | 179 / 449 | No |
| Double blocks, retain L1 preference | 67.94 | 4075.5 | 179 / 449 | No |
| Force tiling, omit Stream-K fixup | 70.45 | 3959.5 | 188 / 397 | No |

The faster-looking rows changed the reduction partition, continuation, and speculative acceptance.
Their throughput is a measurement of different generated workloads and does not establish a kernel speedup.
The two double-block variants generated the same text and acceptance as each other; shared-memory preference was slower in that paired comparison too.
The existing L1 preference remains the better measured choice.
The final production replay reproduced all initial output hashes, cache counts, generated-token counts, and draft acceptance counts across the warmup, needle, and conversation requests.
Conversation decode changed by less than 0.1%, and needle decode changed from 87.22 to 86.81 tokens/s, a difference below 0.5%.
This return control found no material end-to-end benchmark drift.

## Fixed-work matrix tests

The existing llama.cpp `test-backend-ops` executable measured CUDA-graph execution on seven matrix shapes taken from the model's IQ4_XS tensor metadata.
Each shape used seven input columns to exercise the J8 path without changing generated text or speculative acceptance.
The shapes included the two large feed-forward directions, recurrent projections, and smaller or fallback matrices.
Repeated single-operator timings isolate matrix work, but do not reproduce the complete model's cache behavior or establish sub-percent differences.

| Variant | K=5120, M=17408 | K=17408, M=5120 | K=6144, M=5120 |
| --- | ---: | ---: | ---: |
| Production | 106.07 us | 109.97 us | 40.76 us |
| Isolated preload control | 108.25 us | 112.09 us | 41.82 us |
| Double blocks, prefer shared memory | 120.89 us | 121.98 us | 47.00 us |
| Double blocks, retain L1 preference | 111.80 us | 113.57 us | 43.29 us |
| Force tiling | 108.19 us | 138.14 us | 51.60 us |
| 64-row tile, 128 threads | 109.77 us | 111.30 us | 41.44 us |
| 64-row tile, 128 threads, double blocks | 112.14 us | 115.89 us | 46.57 us |
| Prefetch next quant block into L1 | 112.44 us | 115.15 us | 43.52 us |

Lower time is better.
No candidate improved the dominant shapes convincingly against both controls.
The smaller tile helped some small shapes modestly but did not deliver a broad matrix improvement.
Doubling the grid made both the original and smaller tile slower on these major shapes.

The first 64-row prototype retained 256 threads and failed with an illegal memory access in its isolated test process.
Reducing the thread count to 128 made the tile layout viable; all seven CPU-reference numerical comparisons then passed.
This failed prototype was never run in a production worker or retained as a candidate.

The prefetch variant fetched the next IQ4_XS block while the current tile was decoded and used, without changing arithmetic.
All seven CPU-reference numerical comparisons passed, but the matrix timings regressed.
The experiment followed NVIDIA's documented [PTX prefetch instruction](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prefetch-prefetchu).
Shared-memory experiments were motivated by NVIDIA's documented [Ampere cache and occupancy tradeoffs](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html).
Neither mechanism guarantees a speedup merely because it increases nominal concurrency or begins a memory request earlier.

## Decision and limits

Keep the existing production kernel configuration.
These experiments reject several plausible local changes; they do not prove that IQ4_XS is optimally implemented or that future kernels cannot improve it.
Long-context attention, a larger kernel rewrite, and alternative speculative policies were not newly benchmarked in this round.
The earlier 5-15% improvement estimate was a research target, not a measured opportunity that these trials recovered.

The original broker configuration has SHA-256 `9da896acf2d0686bd7e523b895a44a4ba05686d5249245e36f74ca81e4dd534d`.
Existing source edits in the llama.cpp checkout were preserved; experimental edits were confined to temporary source copies.
No candidate image was installed and no kernel overrides were added to the broker.
Final broker health returned HTTP 200, the test lease was released, and no scratch container remained running.
GPU 3 returned to 15 MiB used with dynamic idle clocks of 210 MHz core and 405 MHz memory at its unchanged 250 W limit.

## Reproduction artifacts

Temporary evidence is in `/tmp/marathon-kernels-20260908` and is subject to normal temporary-file cleanup.
It includes exact request manifests and responses, `summary.json`, matrix test cases and logs, build commands, experimental libraries, and source patches.
The three principal patches are `iq4-j8-scheduling.patch`, `iq4-small-tile.patch`, and `iq4-prefetch.patch`.
The scheduling patch is relative to the existing host source with its production L1 preference.
The small-tile and prefetch patches describe separate experiments, not a combined shipping change.
Private conversation responses and experimental binaries are excluded from this repository.

The isolated scheduling preload has SHA-256 `5c11fb47e5da688421794a12e75604384865882de7709e2bc1924a240e79b044`.
`bench.mjs` is a temporary copy of the existing GPU benchmark with an explicit kernel-preload mount and artifact hash added.
`run.py` holds the existing Marathon pool lease around each inference benchmark.
`ops.py` applies the same lease and conflict checks around standalone matrix tests.
Scratch containers are stopped and removed by their respective runners.
