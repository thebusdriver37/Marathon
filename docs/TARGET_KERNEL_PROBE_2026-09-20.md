# IQ4 decode counter profile and activation-buffer probe

The isolated two-buffer candidate passed numerical checks but did not establish a useful matrix-speed improvement.
Keep the production kernel.
The [preceding source investigation](TARGET_KERNEL_RESEARCH_2026-09-20.md) identified this mechanism; this probe tests only that mechanism, not the separate compressed-Q8 attention-staging proposal.

## Isolation and runtime reproduction

The probe used the free RTX 3090 on physical GPU 3 at its unchanged 250 W limit, under the existing Marathon pool lease and conflict monitor.
GPU 1's live Marathon worker was not restarted or modified.
The production image was `sha256:fc98366ec06248a2b9d4df9d37e4fe86d6419fc79598e0d40fb0c63554c038a1`.
The runtime profile retained the Swift/uncensored merged IQ4_XS target, Marathon R32 drafter, Q8/Q8 target KV, six-token draft window, lookup policy, scheduler reuse, and 196000-token context capacity.
It replayed the previously authorized local 4K token suffix and generated 64 tokens without executing recorded tools.
This reproduces backend inference through the native completion endpoint, not a complete Marathon client session.

Nsight Compute 2025.3.1.4 ran in the isolated container with profiling capabilities and only GPU 3 exposed.
Clock control was disabled, preserving the existing power policy.
The first name filter matched nothing; a broader diagnostic capture then selected startup kernels, which are excluded from the decode findings below.
The final capture combined the exact numeric kernel specialization with the `target.decode.rows=7` NVTX range and selected three actual target-verification launches after prompt processing.
All counter-run completion timings are instrumented and are excluded from speed claims.

## Decode hardware counters

| Sample | Duration under profiling | DRAM throughput, percent of peak | Barrier cycles per issue | Average warp cycles per issue |
| --- | ---: | ---: | ---: | ---: |
| 0 | 27.808 us | 75.76% | 0.578 | 6.327 |
| 1 | 67.936 us | 81.50% | 0.714 | 6.068 |
| 2 | 68.576 us | 80.75% | 0.741 | 6.087 |

These IQ4_XS J8 launches used 82 blocks, 256 threads per block, 92 registers per thread, and 40992 bytes of dynamic shared memory per block.
The barrier-to-warp-latency ratios are approximately 9.1%, 11.8%, and 12.2%.
Long-scoreboard dependencies account for approximately 31-32% of cycles between issued instructions in these samples.
This supports testing synchronization reduction while indicating that memory dependencies remain a larger cost.
Neither ratio is a fraction of request wall time or an achievable speedup estimate.
The samples do not prove a global bandwidth ceiling or characterize every matrix shape.

## Isolated candidate and numerical gate

The candidate stores both activation half-tiles in shared memory, preserving the two dot-product calls and their order while reducing four block barriers to two per loop iteration.
Only IQ4_XS J8 changes; row tiles, thread count, Stream-K partition, L1 preference, and other kernel specializations retain their control implementation.
The additional buffer costs 2048 bytes of shared memory per block.
Compiler resource inspection reports 94 instead of 92 registers for the nonfallback J8 specialization and 140 instead of 144 for the fallback specialization, with no local-memory allocation or stack use reported for either build.

Both candidate and unchanged control were built with the same CUDA 12.8 toolchain and flags, including line information, as isolated IQ4 translation-unit libraries.
They resolve remaining dependencies against the production CUDA backend, avoiding a whole-backend rebuild.
The existing model-derived seven-shape `test-backend-ops` fixture exercised seven input columns.
Both libraries passed all seven CPU-reference numerical checks.
These tolerance checks are not proof of bit-identical outputs or unchanged task quality.

## Fixed-work timing gate

Two performance repetitions per variant used the order production, control, candidate, candidate, control, production.
Each test repeatedly executes the same matrix operation through CUDA graphs, excluding changes in generated text and speculative acceptance as explanations for timing differences.
Values below are arithmetic means of the two runs in microseconds per operation.

| Matrix K, M, input columns | Production | Control | Two-buffer candidate | Candidate latency change versus control |
| --- | ---: | ---: | ---: | ---: |
| 5120, 17408, 7 | 110.355 | 109.030 | 108.755 | -0.25% |
| 17408, 5120, 7 | 113.175 | 111.440 | 111.685 | +0.22% |
| 6144, 5120, 7 | 41.865 | 41.870 | 41.310 | -1.34% |
| 5120, 6144, 7 | 42.025 | 41.680 | 41.555 | -0.30% |
| 5120, 48, 7 | 22.610 | 22.510 | 22.330 | -0.80% |
| 5120, 1024, 7 | 15.030 | 14.995 | 15.095 | +0.67% |
| 5120, 12288, 7 | 80.960 | 79.425 | 79.925 | +0.63% |

Negative percentages mean lower latency.
The dominant feed-forward directions are effectively tied, with changes in opposite directions.
Repeat-to-repeat variation reaches several percent, so the small favorable entries are not established gains.
The production/control differences likewise do not establish that rebuilding the unchanged IQ4 library improves serving.
This bounded screen cannot rule out a tiny gain, but provides no convincing reason to advance this candidate to full conversation testing.

## Decision, quality limits, and final state

Jev reviewed only aggregate counters, numerical outcomes, timing results, and limitations, and selected stopping this candidate over expanding to serving tests or more matrix repetitions.
That is an advisory judgment rather than an independent benchmark or calibrated success probability.
No private conversation text, token arrays, or session paths were sent to Jev.

No candidate conversation-speed or task-quality benchmark was run because the fixed-work performance gate did not establish a gain.
The test confirms numerical viability, not a quality-qualified production optimization.
It leaves compressed-Q8 attention staging as a separate untested avenue and does not justify repeating the previously rejected grid, tile, or prefetch changes.
That separate avenue was subsequently evaluated in the [Q8 staging probe](Q8_STAGING_PROBE_2026-09-20.md).

The experiment containers exited, GPU 3 returned to 15 MiB used, and the lease was released.
GPU power limits remained 230/275/250/250 W, the broker health endpoint returned HTTP 200, and the GPU-control repository remained clean.
No production library or configuration was deployed.

Private and binary reproduction artifacts are retained outside Git in `.marathon/diagnostics/target-kernel-probe-20260920/`, with directory mode 0700.
They include the successful capture command, the excluded startup capture, final decode counters, isolated patch and source copies, build logs, numerical and performance logs, aggregate results, Jev exchange, and final artifact hashes/state.
The small helper scripts are scoped to this diagnostic directory rather than installed as another runtime system.
