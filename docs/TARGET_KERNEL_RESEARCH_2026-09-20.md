# Target-kernel research, 2026-09-20

Two source-level mechanisms merit further investigation, but this research establishes no additional decode speed gain.
Jev recommends sampling hardware counters for the dominant IQ4_XS J8 decode kernel before implementing a narrowly isolated activation-buffer experiment.
No GPU worker, production configuration, model, power limit, or kernel was changed during this investigation.

## Evidence and exclusions

The [production runtime profile](MARATHON_RUNTIME_PROFILE_2026-09-20.md) assigns 2,665 ms of 4,212 ms summed GPU kernel duration to quantized matrix multiplication at 75K occupied tokens.
Attention accounts for 720 ms, approximately 17%, versus approximately 2.7% at 4K.
These are summed kernel durations, not request-wall-time fractions or proof of a particular resource bottleneck.

The [September 8 trials](GPU_KERNEL_TRIALS_2026-09-08.md) already rejected doubled grids, forced tiling, smaller row tiles, shared-memory preference, and next-weight-block L1 prefetch on dominant fixed-work matrix shapes.
Some conversation runs appeared faster only while changing generated text and speculative acceptance.
Repeating those changes unchanged is not justified by finding another repository that uses them.
The [September 16 counters and trials](GPU_WARM_PREFILL_2026-09-16.md) concern prefill J128 and attention, not the J8 decode bottleneck.

Current source already contains L1 cache preference for the relevant matrix specializations and native fused Q8 K/V attention at long context.
The production trace confirms the long-context fused path; headline improvements from introducing fusion elsewhere cannot be counted again here.

## Candidate 1: two activation buffers inside IQ4_XS J8

Public implementation: [patch 0008](https://github.com/0x7067/qwen38-27b-rtx3090-llamacpp/blob/8c851eb3e479061f6ffbe9690148e80305164081/benchmarks/engine-trial-2026-09-02/llamacpp/patches-rebased/0008-mmq-smalln-grid-plus-l1-pipeline.patch).
Repository HEAD observed during inspection: `8c851eb3e479061f6ffbe9690148e80305164081`.

Our current `mul_mat_q_process_tile` loads one activation half-tile, synchronizes, computes, synchronizes, overwrites that buffer with the second half, synchronizes, computes, and synchronizes again.
The public Y-buffer variant stores both activation half-tiles simultaneously and calls the same two dot-product operations in the same order with two barriers rather than four per loop iteration.
This differs from the previously rejected prefetch of the next quantized weight block.

Port only that buffer/barrier mechanism for IQ4_XS J8 if counters support it.
Keep current row tile, thread count, Stream-K partition, L1 preference, and other quantization kernels unchanged.
The complete public patch bundles grid and tile changes, including unconditional tile changes even when its environment switch is off.
It is therefore unsuitable as a clean on/off experiment here.

Additional shared memory can reduce residency or worsen performance, and source-level arithmetic order does not guarantee compiled numerical equivalence.
The inspected rebase notes establish successful compilation but do not establish a standalone speedup for this component on our model.

## Candidate 2: stage compressed Q8 attention data ahead of use

Public implementation: [llamAmpere attention source](https://github.com/JakeATX/llamAmpere/blob/2cb16936b5d081a92f1d0369954561efb6d1e2c7/ggml/src/ggml-cuda/fattn-mma-f16.cuh).
Repository HEAD observed during inspection: `2cb16936b5d081a92f1d0369954561efb6d1e2c7`.

The fork stages raw compressed Q8 K/V tiles into shared memory with asynchronous copies before dequantization.
Our specialized Q8 loaders read compressed data directly from global memory.
Existing asynchronous copies elsewhere in our attention code, for FP16 data or masks, do not implement this specific compressed-Q8 staging mechanism.

This is a more involved port because layouts and attention implementations differ.
It is principally a long-context candidate given the measured attention shares.
The fork's [release analysis](https://github.com/JakeATX/llamAmpere/blob/2cb16936b5d081a92f1d0369954561efb6d1e2c7/docs/llamampere-v0.3/ARTICLE.md) uses different weights/cache configurations and a 3090 Ti at 350 W for headline comparisons, outside our power constraint.
It also reports a regression from deeper raw KV staging in its turbo attention path.
Neither its positive aggregate results nor that negative variant directly establishes the outcome of a one-stage Q8/Q8 port here.

## Jev review and next probe

Jev 1.13.0 received only curated aggregate measurements, constraints, prior negative results, and public mechanism descriptions.
No conversation text, token arrays, or private session paths were sent.
The request and response are retained in the ignored `.marathon/diagnostics/target-kernel-research-20260920/` directory.
Jev selected `counters_first`, ahead of immediately implementing the Y-buffer or attention port.
Its reported confidence is an advisory model judgment, not a calibrated probability of success.

The smallest useful next probe is a leased, isolated GPU run sampling representative IQ4_XS J8 decode kernels with Nsight Compute, using the existing production-matched runtime harness.
Measure barrier stalls, load dependencies, shared-memory use, registers, and residency on the dominant feed-forward shapes.
Do not infer available speedup from low occupancy alone, and exclude profiler timings from throughput comparisons.

If the measurements support barrier reduction, reuse the previous isolated IQ4 translation-unit preload approach with an unchanged control build.
First compare CPU-reference numerical error and fixed-work timings for the seven existing model-derived matrix shapes at seven input columns.
Only a convincing improvement advances to repeated production-matched cached completions at short and 75K occupied context, followed by near-196K capacity validation and broader task-quality checks before promotion.
Retain the current target weights, Q8/Q8 cache, draft settings, power policy, and reduction partition throughout that comparison.
Whole-backend rebuilds previously changed unrelated kernels and regressed the control, so they should not serve as an uncontrolled candidate comparison.

Follow-up: the [decode counter profile and isolated Y-buffer experiment](TARGET_KERNEL_PROBE_2026-09-20.md) are complete.
The Y-buffer passed numerical checks but did not establish a useful fixed-work gain, so it was not promoted to serving tests or production.
The subsequent [Q8 staging probe](Q8_STAGING_PROBE_2026-09-20.md) found a compact candidate with a small measured serving improvement and exact tested output parity.
It remains experimental and is not deployed.
