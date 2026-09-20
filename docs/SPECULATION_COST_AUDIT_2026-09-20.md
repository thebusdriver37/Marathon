# Speculation cost audit, September 20, 2026

The existing production configuration reached 194.12 tokens/second over a 512-token streaming window on a synthetic copy-and-edit task.
This is a measured baseline peak, not a new optimization or a demonstrated 200 tokens/second result.
Seven proposals still fails with CUDA out of memory on the current deployment, narrowing the next investigation to its draft attention execution and memory lifetime.

## Constraints and isolation

All measurements used GPU 1, a single RTX 3090 Ti at its existing 275 W limit, the existing merged IQ4_XS target, Q8_0 target KV, Q4_0 draft KV, and 196000 configured context capacity.
These short synthetic prompts do not measure decode at 196000 occupied tokens.
The central configuration was unchanged, the selected worker was initially unloaded and exclusively leased, and diagnostic containers used fresh empty slot directories.
No production conversation or slot snapshot was read or sent to Jev.
Jev received curated numerical evidence and source findings only.
Other active workers were left alone.

Runtime image: `sha256:765a84864664d953cc274adb0fdb161b793269857e4602d228460a790919cea8`.
Draft: `Qwen3.8-27B-DFlash2-Marathon-R32-Q4_K_M.gguf`, SHA256 `e096aa09c26d5096b63b1a4d0400258819980b16250ef5c5d08fcf830e6bb6a6`.
Full command and configuration hashes are saved with the experiments.

## Paired diagnostic requests

Each case used greedy sampling with medium reasoning, an eight-token warmup, and request proposal widths 6, 0, 0, 6 with a fixed 768-token cap.
Zero proposals keeps the drafter loaded and is not a clean standalone target baseline.
The incomplete code and prose outputs are throughput diagnostics, not successful-task comparisons.

| Synthetic task | Six proposals, tokens/s | Zero proposals, tokens/s | Approximate useful tokens per verification round |
|---|---:|---:|---:|
| Copy/edit | 185.62, 181.99 | 35.07, 34.89 | 6.79 |
| New code | 129.36, 128.45 | 35.37, 35.23 | 5.22 |
| Novel prose | 69.21, 68.95 | 35.13, 35.15 | 2.81 |

All four copy outputs had matching hashes.
Code and prose matched within each repeated setting but differed between speculative and zero-proposal settings, limiting direct speedup interpretation.
Total decode time divided by verification rounds was approximately 37 ms for copy and 40-41 ms for code and prose.

## Completed copy task and peak definition

The task returns a 100-record Python dictionary with exactly one quota changed.
Both naturally completed outputs passed AST parsing and exact dictionary comparison, with identical output hashes.
Each generated 3048 tokens from a roughly 3040-token prompt.
The peak measurement required at least 512 consecutive tokens and two seconds, fixed before measuring.
Streaming token IDs matched cumulative token counts.

| Run | Whole decode, tokens/s | Best qualifying window, tokens/s | Window seconds | Full request seconds |
|---|---:|---:|---:|---:|
| Cold prompt | 186.37 | 194.12 | 2.638 | 19.75 |
| Warm prompt | 183.29 | 191.68 | 2.671 | 16.83 |

This highly reusable output strongly favors ngram speculation and does not establish ordinary prose or long-context performance near 200 tokens/second.

## Phase timing

A diagnostic copy of the same worker enabled existing verbosity-four speculative timers.
Per-request differences in cumulative draft generation counters were approximately 59-61 ms out of 4117-4125 ms for copy, 728-740 ms out of 5826-5869 ms for code, and 1347-1357 ms out of 10848-10940 ms for prose.
Draft generation therefore accounted for about 1.5% of copy decode and 12.5% of code/prose decode.
One copy request used 101 successful ngram rounds and 12 DFlash rounds.
The residual includes target verification, draft feature injection, synchronization, and scheduling; these host timers are not a separate GPU kernel profile.
The small timing differences from ordinary verbosity are not optimization gains.

Source inspection found that request-level proposal limits truncate after the configured DFlash block is computed.
Confidence filtering likewise occurs after that forward pass.
This does not invalidate historical startup-width sweeps, which changed the configured block itself.

## Seven-proposal feasibility retest

After receiving the new phase evidence, Jev preferred one isolated seven-proposal feasibility test over deeper instrumentation, with choice weights 0.64 versus 0.36.
These are advisory model outputs, not probabilities of engineering success.
The diagnostic configuration changed both the startup draft maximum and ngram proposal length from six to seven, preserving the other constraints.
The model loaded, but the first synthetic READY request failed before a benchmark output.

The reported CUDA out-of-memory site was `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)` in `ggml_cuda_flash_attn_ext_mma_f16_case`.
The stack passes through `common_speculative_impl_draft_dflash::draft` and `llama_decode`.
This locates the observed failure in draft attention execution; it does not prove that target verification buffers or shared memory capacity alone caused it.
Kernel loading, persistent allocations, and allocation lifetime remain to be separated.
The failed container was removed and the GPU released.

With this failure evidence, Jev selected diagnosis of seven-token draft attention and memory lifetime as the next investigation over broad sweeps or more of the same training.
The engineering objective is to fit an additional proposal without reducing context, target precision, or cache precision, then measure whether the extra accepted tokens outweigh round cost.
No speed benefit from seven proposals has yet been measured.

## Reproducibility

Harness: `scripts/evals/speculation_cost_probe.py`.
Jev requests and responses are retained beside the relevant diagnostic artifacts.
The harness was exercised end to end in paired, copy-peak, phase-cost, and wide-copy modes, including cleanup after the seven-proposal failure.

- `.marathon/diagnostics/speculation-cost-20260920`: paired requests and metrics.
- `.marathon/diagnostics/speculation-copy-peak-20260920`: complete synthetic streams, exact checks, peak summaries.
- `.marathon/diagnostics/speculation-phase-cost-20260920`: trace log, phase summary, Jev decision.
- `.marathon/diagnostics/speculation-wide-copy-20260920`: exact command, failure log, updated Jev decision.
