# Warm-prefix GPU profiling and kernel trials, 2026-09-16

## Decision

Keep the deployed CUDA backend, Q8_0 target KV cache, and 256-token microbatch.
Six isolated kernel changes slowed the 60K-cached-prefix plus 20K-new-token workload.
Nsight Compute counters are available through a temporary GPU 3 container with NVIDIA Nsight Compute 2025.3.1.4; no driver configuration change or user-run `sudo` command is needed.
The only retained speed lever is a higher GPU power limit: 275 W reduced warm time by 5.3% on average, and 300 W reduced it by 8.8%, against a matched 250 W control.
This is a speed-versus-energy policy choice because prior production sweeps identify 250 W as the card's efficiency knee.

## Workload and controls

The input is generated deterministic text and contains no Marathon usage trace or private session content.
The target is Qwen3.8 27B IQ4_XS with DFlash2, a requested 196K context, Q8_0 target KV, Q4_0 draft KV, logical batch 1024, and microbatch 256 on an RTX 3090.
One request processes a cold 60,000-token prefix, a continuation adds 20,000 tokens with `cache_n=60000`, and an exact replay has `cache_n=79996` and processes four tokens.
The deployed baseline processed the warm append in 33.584 seconds, or 595.5 new tokens per second; an identical-source rebuilt backend processed it in 33.865 seconds.
This agreement supports using the rebuilt backend as the control for source trials.
Each source trial swapped only the CUDA backend library inside an isolated container and used the GPU 3 lease.
No registered worker, serving image, broker configuration, or model file was changed.

The warm-suffix Nsight Systems trace assigns 46.86% of summed GPU kernel duration to FlashAttention (14.494 seconds), 29.73% to IQ4_XS J128 matrix multiplication (9.195 seconds), 8.23% to gated delta network, and 4.31% to Q5_K J128 matrix multiplication.
These percentages describe kernel time during the 20K continuation, not request-wall-time shares.
The exact cached replay took 126 ms in the uninstrumented baseline, so rebuilding the already-cached prefix is not the dominant warm cost.

## Hardware counters

The R580 driver has `RmProfilingAdminOnly: 1`.
Granting `cap_perfmon` to the old Nsight Compute 2024.1 executable did not enable a usable target profile, and that temporary file capability was removed.
NVIDIA's checksum-verified Nsight Compute 2025.3.1.4, run in a short-lived GPU 3 container with profiling capabilities, captured and replayed kernels successfully.
The counters below are single kernel instances, so they diagnose resource limits without estimating a full-request speedup.

| Kernel instance | Grid, threads per block | Registers per thread | Dynamic shared memory per block | Achieved occupancy | DRAM throughput | Duration |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Q8 FlashAttention near 80K tokens | 128 blocks, 128 threads | 255 | 34.43 KiB | 13.14% | 4.03% of peak | 13.58 ms |
| IQ4_XS J128, late cold fill | 80 blocks, 256 threads | 254 | 57.86 KiB | 16.67% | 14.29% of peak | 177.34 us |
| IQ4_XS J64 candidate, 1K context | 82 blocks, 256 threads | 168 | 48.38 KiB | 16.66% | 41.68% of peak | 221.79 us |

FlashAttention's grid provides only 0.78 waves per SM on 82 SMs, and its maximum feasible occupancy is 16.67% because of register and shared-memory usage.
Its measured SM and memory throughput are each 33.10% of peak, so a simple claim that DRAM bandwidth is saturated is unsupported.
The profiled J128 instance provides 0.98 waves per SM, with at most one 256-thread block resident per SM because of both its 254 registers per thread and shared-memory use.
J128 recorded zero local-memory spills, 18.84% L1 hit rate, 83.89% L2 hit rate, and 39.13% integer tensor-pipe activity.
Its instruction and resource pressure merit a focused experiment, but occupancy alone is not a speedup prediction.
Nsight Compute's launch skip counted additional CUDA graph launches, so this J128 instance was captured during late cold fill; it has the same J128 kernel specialization as the warm continuation.
The separate J64 capture reduced registers per thread from 254 to 168, but both register allocation and shared memory still limit its sampled launch to one block per SM.
The J64 instance used a 1K synthetic context, while the J128 instance came from late cold fill of the 60K prefix; their individual durations and DRAM percentages are not a controlled comparison.
The unchanged 16.7% occupancy does not by itself explain the whole-request J64 slowdown, but it rules out an occupancy increase in this sampled shape.
The attention launch can already fit two 128-thread blocks per SM by both register and shared-memory limits, while its 128-block grid supplies only 0.78 waves of that capacity across 82 SMs.
Reducing registers without also changing the 34.43 KiB shared-memory allocation or the number of independent query tiles cannot raise its block residency.
Instrumented request timings were excluded from performance comparisons.

The gated-delta kernel accounts for 8.23% of warm-suffix GPU time and provides a separate optimization target after the rejected attention scheduling trials.
A representative 256-token `gated_delta_net_cuda<128, false, true>` launch used 48 registers per thread, no dynamic shared memory, 70.91% achieved occupancy, and 1.87 waves per SM.
Its load/store pipeline reached 87.73% of peak while DRAM reached only 9.46%, with a 94.82% L2 hit rate and no local-memory spills.
This supports reducing redundant cached loads or load/store instructions inside the kernel rather than increasing its occupancy.

## Matched source trials

All source trials were built from the same saved source and CUDA 12.4 toolchain as their 75% stream-K threshold / J128 control.
The benchmark sent identical synthetic token arrays to the same deployed server binary with the candidate CUDA backend loaded in a container.

| CUDA backend | Cold 60K prompt | Warm 20K append | Change from matched control | Sampled output |
| --- | ---: | ---: | ---: | --- |
| Rebuilt control | 77.323 s | 33.865 s | Reference | Matches deployed baseline |
| Stream-K threshold 75% to 80% | 84.385 s | 39.849 s | +17.7% warm time | Changed the 2K output; 80K sample matched |
| IQ4_XS prefill tile cap J128 to J64 | 82.072 s | 35.533 s | +4.9% warm time | Sampled 2K and 80K outputs matched |
| Q8 attention query tile 64 to 32 | 86.599 s | 41.987 s | +24.0% warm time | Sampled 60K and 80K outputs matched |
| Q8 attention block 128 to 256 threads | 83.732 s | 40.298 s | +19.0% warm time | Sampled 60K and 80K outputs matched |
| Gated-delta shared Q/K staging | 78.541 s | 34.342 s | +1.4% warm time | Sampled 60K and 80K outputs matched |
| Q8 key copy as aligned 16-bit units | 81.847 s | 37.522 s | +10.4% warm time | Sampled 60K and 80K outputs matched |

At 80K, raising the attention stream-K threshold routes a 128-block grid into a 164-block split-K grid with an extra partial-result combination.
The measured regression rejects this dispatch change and its short-output mismatch independently rules out adopting it without deeper validation.
The J64 experiment changes only SM86 IQ4_XS operations with at least 128 input columns and leaves decode shapes on their existing path.
Its smaller tile did not improve end-to-end warm prefill and is also rejected.
The 32-query attention tile doubled the independent output tiles in long-context Q8 prefill but slowed the warm request by 24.0%; matching cache-hit counts and sampled outputs exclude a cache miss or obvious generated-output change.
This trial rejects smaller query tiles as the simple fix for the attention grid's unused block slots.
Doubling each attention block to eight warps preserved the 64-query tile but slowed the warm request by 19.0%, so added intra-block latency hiding is also rejected.
Loading each gated-delta Q/K row once per four-warp block reduced redundant global loads but added two block barriers per recurrent token; the full warm request slowed by 1.4%, so this implementation is rejected.
Copying Q8 key data as aligned 16-bit units removed the pack instruction highlighted by stall sampling but doubled shared-store operations; the warm request slowed by 10.4%, so this load change is rejected.
These one-token response hashes check for obvious changed behavior; they are not a general model-quality evaluation.

## Power-limit result

The exact same rebuilt control backend was also measured on physical GPU 3 at its configured 250 W limit and at temporary 275 W and 300 W limits.
Every raised-power run restored 250 W before releasing the GPU lease.
An initial privileged-container wrapper exposed all physical GPUs and addressed index 0 instead of the requested card, temporarily changing GPU 0 from 230 W to 250 W during failed model starts.
No inference completed in those attempts; GPU 0 was restored to 230 W, all four limits were verified, and successful tests selected physical GPU 3 explicitly.

| Power limit | Cold 60K prompt | Warm 20K append | Warm change from 250 W | Busy peak temperature |
| --- | ---: | ---: | ---: | ---: |
| 250 W | 77.391 s | 33.988 s | Reference | Not sampled |
| 275 W, run 1 | 72.324 s | 32.127 s | -5.5% | Not sampled |
| 275 W, run 2 | 72.434 s | 32.246 s | -5.1% | 66 C |
| 300 W | 69.271 s | 30.996 s | -8.8% | 67 C |

The two 275 W runs average 5.3% less warm time and 6.5% less cold-fill time than the matched 250 W run.
All four runs reported the expected cache counts and identical sampled outputs.
The GPU-control history independently identifies 250 W as this card's energy-efficiency knee, so 275 W is a measured speed-versus-power choice rather than a free software gain.
The 300 W point improves warm time by another 3.7% over the 275 W mean and raises measured busy power from a 270.81 W mean at 275 W to 294.46 W.
Its speed gain is real, but its marginal energy efficiency is worse than the 250-to-275 W step.

## Next step and artifacts

A future source experiment would need a larger design change, such as a chunked gated-delta implementation or a Q8 key loader that removes the observed dependent load-and-pack sequence without adding query duplication or block barriers.
The bounded scheduling, tiling, and shared-staging variants tested here do not justify another nearby parameter sweep.
Candidate promotion requires repeated uninstrumented request times, correct cache-hit counts, fixed-work numerical comparison, and broader output or task-quality checks.
No source candidate from this session meets those gates.

The reproducible synthetic harness is `.marathon/drafter-training/profile_warm_prompt.py`.
Ignored local artifacts include the Nsight Systems trace and Nsight Compute reports under `.marathon/drafter-training/prompt-profile-*`, the exact trial patches and build logs under `.marathon/drafter-training/kernel-trials/`, and temporary build products under `/tmp/marathon-llama-streamk80-*`.
The GPU 3 experiment containers exited and released the lease.
The prepared TypeSafe/Jev request at `.marathon/drafter-training/jev-warm-prefill-request.json` contains only aggregate synthetic measurements and has not been sent because `TYPESAFE_API_KEY` is absent from this agent process and the user service environment.
