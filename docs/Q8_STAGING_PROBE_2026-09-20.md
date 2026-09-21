# Q8 attention staging: bounded kernel and serving experiment

The compact staging candidate reduced attention-operation latency by approximately 2.5-3.2% at 75K/128K and improved measured cached decode throughput by approximately 1.5%/1.8% in the bounded serving comparison.
All tested compact-kernel outputs and complete serving continuations matched their controls exactly.
This is a promising small candidate, not a broadly qualified production optimization.
Production was unchanged during the experiments described below.
After the matched Marathon-session comparison, the user authorized production deployment on 2026-09-20.
The three registered Marathon workers now inherit image `sha256:8af4fa77e493f6765b7c66d4f6cbfb673e0add0919b470c11526e08d54365e04`, packaging the exact tested compact library.
The deployment record and rollback are maintained in GPU-control at `runtime/qwen38-q8-staging-deployment.md`.

The reusable [candidate patch](../scripts/experiments/q8-compact-staging-sm86.patch) is preserved without model weights, compiled libraries, or private conversations in version control.

## Hardware, model, and isolation

Tests used the free RTX 3090 on physical GPU 3 at its unchanged 250 W power limit through the existing experiment lease.
The live Marathon worker on GPU 1 was not restarted, modified, or stopped.
The production image was `sha256:fc98366ec06248a2b9d4df9d37e4fe86d6419fc79598e0d40fb0c63554c038a1`.
Serving retained the Swift/uncensored merged IQ4_XS target, its 24 query heads and four KV heads with 256-wide heads, the Marathon R32 DFlash2 drafter, six-token draft window, lookup settings, Q8/Q8 target cache, Q4 draft cache, and 196000-token context capacity.
No model quantization, context-capacity, or power-setting change was made.

Control and candidate each preload only the explicit `ggml_cuda_flash_attn_ext_mma_f16_case<256,256,8,8>` translation unit, retaining the production backend for other kernels.
Both builds use the same CUDA 12.8 toolchain and flags.
The source control header SHA-256 is `fcb5fd7c1d5dfcd32fadbfb3cb5b903f97c78d21ce02f80c1f3fb3ff89cc192c`; use that recorded value when reproducing against another checkout rather than assuming an arbitrary upstream version is equivalent.

## Mechanism and diagnosed first attempt

The adaptation follows the compressed-row asynchronous-staging mechanism in [llamAmpere](https://github.com/JakeATX/llamAmpere/blob/2cb16936b5d081a92f1d0369954561efb6d1e2c7/ggml/src/ggml-cuda/fattn-mma-f16.cuh), while retaining our existing native Q8 K layout, V dequantization, swizzle, and arithmetic.
It copies packed 272-byte Q8 rows into shared memory before the existing loaders consume them.

The first port allocated separate raw K and V buffers, adding 17408 bytes per block.
Nsight Compute confirmed that dynamic shared memory rose from 34432 to 51840 bytes, and the launch changed from 164 to 80 blocks.
That altered the Stream-K reduction partition, produced small numerical differences, and slowed long-context fixed-work timing by approximately 3-4% against the first control run.
It was rejected without a serving trial.

The corrected compact version reuses one raw buffer, adding 8704 bytes per block.
It consumes staged V, synchronizes before reusing the buffer for the next K tile, consumes that K tile, and reuses the same buffer for the next V tile while the current value multiplication proceeds.
Nsight Compute confirmed 43136 bytes of dynamic shared memory and the restored 164-block launch with 128 threads and 255 registers per thread.
The original reduction partition and tested numerical outputs are restored.

The staging path is restricted to the current 256-wide, 8-by-8, native Q8/Q8 specialization.
It does not introduce a new target-weight or KV representation, and it does not change the standard larger-prefill path that uses converted V data.

## Fixed-work numerical and timing checks

The deterministic synthetic fixture uses the model's actual head geometry, causal masks, and seven or eight query tokens.
The 75000-token case allocates 75008 padded KV rows and masks the padded positions.
The other occupied lengths are 16384 and 128000.
Each timing run performs warmup followed by three groups of 200 CUDA-graph evaluations per case.
The table averages the per-case median times from two separate runs of each build.

| Occupied KV tokens | Queries | Control, us | Compact staging, us | Latency change |
| --- | ---: | ---: | ---: | ---: |
| 16384 | 7 | 151.462 | 133.720 | -11.71% |
| 16384 | 8 | 155.969 | 132.776 | -14.87% |
| 75000 | 7 | 519.116 | 505.528 | -2.62% |
| 75000 | 8 | 522.115 | 508.826 | -2.55% |
| 128000 | 7 | 847.554 | 821.152 | -3.11% |
| 128000 | 8 | 855.955 | 828.333 | -3.23% |

All six compact output arrays were bit-identical to the unchanged control in both runs.
The first control also matched the production library bit-for-bit on all six cases.
This compares against production numerics, not an independent full-precision attention oracle.
Profiler timings were excluded from the performance table.

## Complete runtime comparison

The serving fixture at 75K uses the previously authorized reconstructed conversation token suffix.
The 128K fixture prepends 53000 deterministic synthetic-record tokens to the same 75K suffix.
It does not read another production conversation or execute recorded tools.
These are native backend completion replays, not a new Marathon UI or broad coding-task evaluation.

Four process runs used the order control A, compact A, compact B, control B.
Each process measured four cached 256-token greedy completions at each context after an unmeasured warmup.
Cold prefill and test-snapshot restore are excluded from measured completion latency.
Snapshots were generated by the first control in a dedicated test directory, with draft and recurrent checkpoint state retained for the following runs.
Every measured request reused all but four input tokens.

| Context | Control A median tok/s | Compact A | Compact B | Control B | Change using means of run medians |
| --- | ---: | ---: | ---: | ---: | ---: |
| 75000 | 56.097 | 57.102 | 55.948 | 55.267 | +1.51% |
| 128000 | 55.241 | 55.989 | 55.890 | 54.703 | +1.76% |

Corresponding average request-wall medians fell from 4.741 to 4.680 seconds at 75K and from 4.842 to 4.756 seconds at 128K.
All 32 measured completions have identical token and content hashes within their context.
Accepted/drafted counts also match: 168/508 at 75K and 176/466 at 128K.
The measured difference therefore does not come from easier generated text or improved draft acceptance.

There is visible timing drift across process runs, and the candidate's 75K repeats vary.
Two related fixtures and two runs per build support a promising small result, not a precise general improvement estimate or universal quality equivalence.
The larger attention-only gain at 16K was not separately tested in complete serving requests.
The 196K capacity was allocated successfully, but near-196K occupied-context and broader task qualification remain undone.

## User-supplied fast-long-context repository

Inspected [satellitedown/fast-long-context](https://github.com/satellitedown/fast-long-context/tree/b8095a5b3a14babcd7c8b1d66209b6f1cdb2c849) at commit `b8095a5b3a14babcd7c8b1d66209b6f1cdb2c849`.
Its [manifest](https://github.com/satellitedown/fast-long-context/blob/b8095a5b3a14babcd7c8b1d66209b6f1cdb2c849/runtime-manifest.json) specifies a different NVFP4 abliterated target, SGLang 0.5.20, and FlashInfer 0.6.18; its published recipe targets a 32 GB RTX 5090.
The [launcher](https://github.com/satellitedown/fast-long-context/blob/b8095a5b3a14babcd7c8b1d66209b6f1cdb2c849/scripts/serve.sh) uses NVFP4 target KV, FP8 draft settings, and DFlash block size eight.
Its alternative FP8-cache profile requests only 131072 context, below our minimum.

The patch enables fixed-length causal speculative verification in native FP4 attention and gathers/dequantizes cached rows directly into a shared FP8 prefill workspace to avoid large temporary tensors.
These are useful implementation ideas, but neither is a drop-in replacement for our already-fused Q8 attention and unchanged IQ4_XS target.
The installer and model downloads were not run because they would not reproduce the requested experiment under its existing constraints.

The [published measurements](https://github.com/satellitedown/fast-long-context/blob/b8095a5b3a14babcd7c8b1d66209b6f1cdb2c849/results/nvfp4.json) use two repetitions, a synthetic coding task, temperature zero, and thinking disabled.
The repository supplies numerical checks and limited long-context recall evidence, but explicitly does not claim broad accuracy equivalence.
Its 240-300 tok/s headline therefore is not evidence that this recipe yields those speeds on our 3090 with the current model and cache.

## Decision and retained evidence

Jev reviewed only aggregate measurements, mechanisms, constraints, and the external repository's documented configuration.
It selected preserving the compact patch as a promising small candidate while leaving production unchanged until broader and near-196K occupied-context qualification.
Its answer is advisory and not an independent measurement or calibrated probability.
No private conversation text or token arrays were sent to Jev.

The experiment workers exited, GPU 3 returned to 15 MiB used, and the lease was released.
The broker health endpoint returned HTTP 200, the GPU-control repository remained clean, and power limits remained 230/275/250/250 W.
The eight generated snapshot files were removed after testing, recovering approximately 7.54 GiB without touching production snapshots.

Private fixtures, output hashes, numerical arrays, builds, source copies, compiler/resource records, commands, timing logs, final state, and Jev exchange remain in the ignored mode-0700 directory `.marathon/diagnostics/q8-staging-probe-20260920/`.
The small reusable compact patch is separately preserved in the repository; compiled libraries, private fixtures, and snapshots are not added to Git.

## Follow-up: identical real Marathon sessions

The compact candidate also improved speed in matched sessions through the actual Marathon launcher, hardened frontend, router, and production-matched worker.
This follow-up used a synthetic archive and the same cache-design question, without reading private conversations.
All runs used physical GPU 3 at 250 W, the unchanged merged target and drafter, Q8/Q8 cache, and 196000-token capacity.
The local experiment adapter forwarded the private test-pool endpoint to the isolated worker; central configuration and live production sessions were untouched.

Each complete prompt occupied 75212 tokens, including Marathon instructions and tool definitions.
Sampling was pinned to temperature zero, seed 8123, and medium reasoning.
The question requested approximately 500 words explaining a bounded LRU cache with TTL, concurrency, boundary tests, and a short Python example.
The output allowance was 4096 tokens so reasoning and the full answer could finish in one request.

The process order was control A, compact A, compact B, control B.
Each process ran one cold warmup and two fresh measured Marathon sessions with identical prompt text and workspace.
All eight measured requests reused 75208 prompt tokens and processed the remaining four.
Their normalized request bodies matched after excluding session identifiers and client metadata.
The cold warmups are excluded from the following table.

| Metric | Control mean | Compact mean | Change |
| --- | ---: | ---: | ---: |
| Backend decode, tok/s | 69.294 | 70.257 | +1.39% |
| Backend decode time, seconds | 37.969 | 37.449 | -0.521 seconds |
| Inference request wall time, seconds | 38.184 | 37.660 | -0.524 seconds |
| Whole Marathon CLI wall time, seconds | 39.797 | 39.316 | -1.21% |
| Time to first visible prose, seconds | 27.632 | 27.254 | -0.378 seconds |

Individual measured speeds were 69.064/69.126 for control A, 70.036/70.134 for compact A, 70.453/70.405 for compact B, and 69.594/69.392 for control B.
Both order comparisons favor the candidate, although the process runs show some timing drift.
This supports approximately a 1.4% gain on this matched Marathon workload, not a universal speedup estimate.
The comparison uses fresh headless Marathon sessions with a long prompt, not interactive terminal rendering or a resumed production conversation.

All eight measured sessions completed in one request without executing tools.
Their complete visible answers were byte-identical, with 2632 generated tokens and identical draft counts of 1947 accepted out of 4433 generated.
The generated cache example passed six local behavioral checks: exact expiration boundary, LRU eviction, replacement and TTL refresh, stored `None`, zero TTL, and zero capacity.
This is evidence against a quality regression on this prompt, not a broad accuracy qualification.
The shared answer itself has a prose error suggesting a plain Python dictionary offers `move_to_end`; its executable example correctly uses `OrderedDict`.
The example uses lazy expiration and was not subjected to concurrency stress testing.

An initial routing attempt stopped before inference because the temporary catalog omitted the remembered local profile.
The test was corrected to use Marathon's current pool-routing path with a separate experiment pool.
A subsequent 2048-token-cap pilot truncated the answer and induced extra continuation requests; that pilot was stopped and excluded before the final 4096-token comparison.
Its evidence is retained separately so it cannot be mistaken for a measured final run.

All experiment workers exited and GPU 3 returned to 15 MiB used.
The broker remained healthy and the central GPU-control configuration remained unchanged.
No compiled library was deployed to production.
The test created no slot snapshots or model copies.
The runner, request bodies, session evidence, timing summary, code checks, and cleanup record are retained under the ignored `marathon-ab` subdirectory of the existing diagnostic directory.
