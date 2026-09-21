# Optimization Benchmark Audit: Four Completed Experiments

Date: 2026-09-20.
Scope: read-only audit of the four rejected optimization experiments.
No code was modified, no GPU tests were run, and nothing was deployed.
Evidence roots: `.marathon/optimization-queue/workspaces/{recurrent,copy-gather,iq4-loads,q8-residual}` and `.marathon/diagnostics/runtime-profile-20260920`.

## Production Reference (Verified)

Model: `Swift-Qwen3.8-27B-Uncensored-Merge-IQ4_XS.gguf` (gguf v3, architecture `qwen35`).
Verified gguf metadata:

- `qwen35.ssm.time_step_rank = 48` (this is `num_v_heads` per `src/models/qwen35.cpp:349`).
- `qwen35.ssm.inner_size = 6144` = 48 heads x 128 head dim.
- `qwen35.ssm.state_size = 128`, `qwen35.ssm.group_count = 16`, `qwen35.ssm.conv_kernel = 4`.
- `qwen35.block_count = 65` = 64 base layers + 1 nextn (MTP) block; `qwen35.full_attention_interval = 4` gives 16 full-attention and 48 SSM layers.
- `qwen35.embedding_length = 5120`, `qwen35.context_length = 262144`.

Verified launch geometry from the 75K nsys trace (`cuda-75000.sqlite`, kernel table join on demangled name):

- `gated_delta_net_cuda<128,0,1>`: grid (48, 1, 32), block (32, 4, 1), 4224 instances, mean 31.99 us, total 135.13 ms.
- `concat_non_cont<unsigned int, 0>`: grid (10240, 1, 1), block (256, 1, 1), 4176 instances, mean 16.12 us, total 67.32 ms.

The launch code `gated_delta_net.cu:183` is `grid_dims(H, n_seqs, ceil(S_v / num_warps))`.
The production grid (48, 1, 32) therefore means H = 48, n_seqs = 1, num_warps = 4.
The instance counts divide cleanly by 48: 4224 = 48 x 88 decode steps, 4176 = 48 x 87 decode steps.
This confirms 48 SSM layers (not 49 as assumed in two result files) and shows the concat launches one fewer step than the gated-delta-net launches.

## Task 1: recurrent (launch-bounds 2 to 4)

Fixture vs production:

- Kernel: same template instance `<128,false,true>` in fixture and production. Verified.
- S_v = 128: matches. T (n_tokens) = 1: matches. K = 2: matches. f32: matches.
- Head count H: fixture 32, production 48. Verified mismatch (probe.cu:16 vs gguf + trace grid).
- State size: fixture 128 x 128 x 32 x 4 B = 2.0 MB; production 3.0 MB. Both fit the 3090's 6 MB L2.
- Dependencies: fixture is a standalone single-op ggml graph; production sits between q/k/v matmuls, conv, and the other 63 layers' kernels per step.
- Timing method: fixture uses wall-clock over a 2048-iteration async loop (probe.cu:44-46); production figure is CUPTI kernel duration from nsys. Not the same measurement.

Measured gap: 8.9 us fixture mean vs 32.0 us production mean, a 3.6x ratio.

Verified facts:

1. The fixture runs 33% fewer blocks (32 vs 48) and 33% less state traffic per launch.
2. The fixture's 2.0 MB state stays L2-resident across 2048 back-to-back iterations of the same kernel.
3. In production, every other kernel of the step runs between two consecutive uses of the same state.
4. The two timing methods differ (wall throughput vs CUPTI duration).

Hypotheses (unverified):

- Cold-L2 hypothesis: the 3.0 MB state is partially evicted by inter-step kernels, so production pays DRAM round trips the fixture never pays.
- Dependency-latency hypothesis: the production kernel starts soon after a `mul_mat_q` producer and its latency is exposed; the fixture's steady-state loop hides launch and producer latency in the async queue.
- Scaling estimate: applying only the verified head-count ratio, 8.9 x 48/32 = 13.35 us, which still leaves ~2.4x unexplained, so head count alone does not account for the gap.

Does a mismatch invalidate the rejection? No.
The rejection rests on candidate versus control inside the same fixture, with a demonstrated 1.5-2% in-fixture noise band and a sign flip between rounds (-0.16% vs +1.45%).
The H=32 vs H=48 mismatch explains why absolute fixture times do not transfer to production; it does not weaken the relative candidate-versus-control comparison, because both builds ran the same fixture.
Verdict: rejection stands.

Smallest corrective test (if this line of work resumes): change probe.cu:16 from H=32 to H=48 (one line, keeps bit-identical dumps comparable) and rerun the same 2-round gate.
Optionally add a >6 MB L2-clear write between iterations to approximate cold state.
That single fixture fix bounds both verified and hypothesized contributors at once.

## Task 2: copy-gather (concat fast path)

Fixture vs production:

- Layout: probe reproduces the production shapes and strides exactly: pool [30720, 1] f32 (row stride 122880 B) viewed as [3, 10240, 1], transposed qkv [1, 10240, 1] with nb[0] = 40960 B, output [4, 10240] (160 KB), concat dim 0, n_tokens = 1. Verified in probe.cpp against the trace.
- Kernel: same `concat_non_cont<unsigned int, 0>` on the control path; production grid (10240, 1, 1) x 256 reproduced. Verified.
- Candidate: 160-block grid-stride shadow of `ggml_cuda_op_concat`; per-element addressing identical, bit-identical FNV-1a hash (16352693202928018) in both order pairs. Verified.
- Dependencies: production concat consumes the transposed qkv produced by the preceding `mul_mat_q`; the probe reuses fixed source tensors across iterations, so the sources are L2-warm.
- Timing method: probe divides saturated iteration time (80 MB add + 32 concats per iteration) by 32, a derived per-concat cost; production is CUPTI kernel duration. Not the same measurement.

Measured gap: 8.49-8.77 us per concat in the probe vs 16.12 us production mean, a ~1.85x ratio.

Verified facts:

1. The probe's per-concat number is derived by division, not measured as kernel duration, so it mixes concat kernel time with inter-kernel gaps and the dominant add kernel's tail.
2. The probe keeps the 160 KB output and its sources resident in L2 across all iterations; in production each layer's qkv arrives freshly from the previous matmul.
3. The worker's own result file notes the probe is launch-gap limited and that the 10240-block grid cost is "not quantified here".
4. The production grid launches 10240 blocks of 256 threads to move 40,960 elements, i.e. 4 elements per block; most of the 2.6M threads are idle.

Hypotheses (unverified):

- The production 16.12 us is inflated by cold sources (qkv just written by `mul_mat_q`, conv-state rows strided 122880 B apart across the pool) and by the 10240-block launch tail, neither present in the probe.
- The probe's derived ~8.5 us is an underestimate of the pure kernel time, because the add kernel's execution overlaps concat's launch gap in the saturated stream.

Does a mismatch invalidate the rejection? Not the neutral verdict, but it does make the screen weaker than it looks.
The candidate was only shown neutral, not shown fast. A kernel that is neutral in a warm, gap-limited probe can still win 16.12 us x 4176 = 67.3 ms (1.6% of traced GPU time) in production if the production time is dominated by cold-source and launch-tail effects.
The layout-exact probe is the strongest fidelity of the four tasks, and correctness is bit-identical, so the patch stays a live candidate. It should be read as "inconclusive, retest in production-matched conditions", which is exactly how the result file classifies it.
Verdict: rejection stands as recorded (inconclusive), with the caveat that the probe cannot rule out a production win.

Smallest corrective test: profile one production-equivalent decode step with the candidate lib preloaded (nsys kernel table, single 75K request, GPU 3 gate) and compare `concat_non_cont` mean duration directly against the recorded 16.12 us.
No fixture change needed; the patch and lib are already preserved in the workspace.

## Task 3: iq4-loads (Q8_0 sram stride 76 to 72)

Fixture vs production:

- Kernel: `mul_mat_q<23,8,0>` (IQ4_XS, J=8), the dominant production kernel. Verified identity via the 2026-09-20 probe's 7-shape matrix.
- Shapes: n = 7 token decode shapes drawn from the production profile (e.g. `iq4_5120_17408_7`). These match production operand shapes, not a full serving context.
- Strides/tiles: grid, tile sizes, L1 prefetch, and double-buffering unchanged; only the Q8_0 sram row stride changes (76 to 72 ints), with the matching static assert relaxed.
- Dependencies: standalone `test-backend-ops` runs; no producer/consumer graph.
- Timing: two perf rounds per variant, mean of rounds; 7/7 correctness shapes pass for all three variants.

Assessment: this is the best-fidelity task of the four. The candidate ran the real deployed kernel on real production shapes and was 0.5-1.2% slower than control on all 7 shapes (+3.2% on the tiny shape), inside the 1-3% production-versus-control noise band.
The NCU-estimated 20.76% conflict-speedup was an upper bound; the kernel is DRAM/long-scoreboard dominated (75-81% of peak), so removing shared-store conflicts did not move kernel time.
Verdict: rejection is well supported. No fidelity mismatch invalidates it.
Note: the stride change also affects other Q8_0-layout quant types, which were not benchmarked; this matters only if the candidate is ever revived.

## Task 4: q8-residual (L2 prefetch of next KV tile)

Fixture vs production:

- Kernel: the deployed compact Q8 attention kernel, control = image default with `/app/libq8-compact.so` preloaded. Verified identity via the pinned baseline image.
- Fixture: synthetic attention-probe, 24 heads, 256-wide, causal, 16384/75000/128000 occupied tokens, 7-8 queries. Head count (24) matches the gguf (`qwen35.attention.head_count = 24`), but the probe is synthetic data, not a production prompt, and 75000/128000 occupied tokens approximate rather than reproduce a real 75K request.
- Launch dimensions: 164 blocks, 128 threads, unchanged shared memory, same Q8/Q8 8x8 specialization. Verified unchanged by the patch (prefetch instructions only).
- Dependencies: single attention op in a CUDA graph; production attention is preceded by rope/quant ops and interleaved with the SSM path.
- Timing: 12 warmup iterations, then 3 repeats of 200 CUDA-graph evaluations per case; medians reported.

Assessment: the candidate is bit-identical (24/24 arrays) and the effect flips sign between the ab pair (-0.5 to -1.7% for the candidate at 75K/128K) and the cd pair (+0.3 to +1.0% for the control), inside the 1.2-3.1% between-process control drift.
This is a correctly executed screen: two order pairs, two separate gate runs, sign flip, rejection.
Residual fidelity gap: synthetic occupancy (dense causal masks over 75000/128000 tokens) versus production's real KV distribution, and the absence of interleaved SSM kernels between attention launches.
Verdict: rejection stands; the mismatch is unlikely to flip the sign-flipped result, but a production-matched A/B would be required before claiming anything.
Smallest corrective test (if revived): one production-matched A/B with the candidate lib preloaded under the nsys gate, comparing flash-attention kernel duration at the 75K profile point.

## Summary

| Task | Fidelity of fixture to production | Verdict on rejection | Key mismatch |
|---|---|---|---|
| recurrent | Low: H=32 vs 48, standalone op, warm L2, wall-clock timing | Stands | H=32 vs 48; gap is not a relative-comparison problem |
| copy-gather | High: exact shapes/strides/grid; warm sources, derived timing | Stands as inconclusive | Probe cannot rule out a production win; retest in production |
| iq4-loads | Highest: real kernel, real shapes | Stands | None material |
| q8-residual | Medium: real kernel, synthetic occupancy | Stands | Synthetic data; sign flip dominates |

The two priority discrepancies both have the same structure: the fixtures understate absolute kernel time (3.6x for the recurrent kernel, 1.85x for the concat kernel) because they run isolated, warm, back-to-back, and they measure throughput rather than CUPTI duration.
None of the four rejections is invalidated by a fixture mismatch.
The one finding most likely to change under a production-matched test is the copy-gather concat patch, which is neutral in a warm probe against a 1.6%-of-GPU-time production kernel.

## Evidence Paths

- `docs/OPTIMIZATION_WORKERS.md` (rules, result contract, pinned image).
- `.marathon/optimization-queue/workspaces/recurrent/result.md`, `probe.cu`, `probe.py`, `candidate.patch`.
- `.marathon/optimization-queue/workspaces/copy-gather/result.md`, `probe.cpp`, `concat-patch.cu`.
- `.marathon/optimization-queue/workspaces/iq4-loads/result.md`.
- `.marathon/optimization-queue/workspaces/q8-residual/result.md`.
- `.marathon/diagnostics/runtime-profile-20260920/cuda-75000-stats.csv` and `cuda-75000.sqlite` (durations, counts, grids).
- `/home/deforest/AI/experiments/swift-uncensored/Swift-Qwen3.8-27B-Uncensored-Merge-IQ4_XS.gguf` (model metadata).
- `.marathon/llama.cpp-iq4-xs-source/src/models/qwen35.cpp` (head count semantics) and the recurrent workspace `source/gated_delta_net.cu:183` (grid mapping).
