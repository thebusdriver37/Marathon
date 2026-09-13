# llamAmpere follow-up evaluation

Date: 2026-09-11.

## Result

There are small optimization signals, but this evaluation does not establish a drop-in production upgrade.
The packed vocabulary shortlist usually improved matched-output decoding by about 2-3%, but exceeded VRAM at the current context allocation and diverged on one long-context greedy case.
The actual short-context Q8 implementation improved its own off/on comparison by 1.3-2.3%, with a smaller and less conclusive gain against the untouched production binary.
The proposed frequent graph-recapture mechanism was not supported by the traces.
Keep production unchanged on this evidence.

## Scope and controls

This evaluates the pasted llamAmpere report against Marathon's actual inference runtime.
Production configuration, model files, and source code were not changed.
Experiments used the two benchmark workers handed over with the request, with exclusive Marathon backend leases.
Existing unrelated workloads and pre-existing source edits were preserved.

Runtime image: `sha256:443d87c87fe673faf379b29dd11bfa22e03c76e4d2fe14e9ce9503c810daf34c`.
Target: `Qwen3.8-27B-Uncensored-IQ4_XS.gguf`.
Draft: `Qwen3.8-27B-DFlash2-Q4_K_M.gguf`, maximum six draft tokens.
Target KV remained Q8, draft KV Q4, batch 1024, microbatch 256.
Medium reasoning and `Minimize thinking` were held fixed.
Sampled requests used temperature 1, top-k 20, top-p 0.95, min-p 0.05, and seeds 6100 and 731.
Greedy requests used temperature zero.

Fixtures contain actual Marathon Python source plus coding or architecture requests.
Each measurement generates up to 512 tokens, generally ending during reasoning.
These are controlled inference measurements, not completed coding-task quality evaluations.
Timing uses the server's generated-token rate; cold prefill and cached decode are recorded separately.
Same-card comparisons avoid treating GPU-to-GPU variation as an optimization.
Raw requests, responses, recipes, logs, profiler traces, and candidate artifacts are retained under `.marathon/diagnostics/llamampere-followup-20260911/`.

## The original Q8 flag experiment did not activate the proposed code

The production CUDA library does not contain `GGML_CUDA_Q8_V_SHORT`.
The flag exists in a pre-existing, uncommitted local header edit, but that edit was not compiled into the tested image.
The production CUDA library SHA-256 is `913386bd1442b39013b891c9d5cc9751360d5f460a3817ac3d2be3d15eff18bc`.
Consequently the earlier report's 1.3% difference cannot establish a gain from that flag.
Setting it in production configuration alone would not activate the optimization.

The candidate implementation enables direct Q8 V loads below 16,384 KV positions for supported small query batches.
The existing implementation already uses direct Q8 V loads in its supported long-context decode path.
The flag checks for presence, so setting it to `0` would still enable the candidate path; the off condition must omit it.

For a real test, the 16 affected 256/256 attention template instantiations were compiled into an isolated preload library using the existing CUDA build environment.
Dynamic linker binding logs confirmed that all 16 dispatch functions in the production CUDA library resolved to the candidate library.
This avoids rebuilding unrelated dirty source files or replacing production binaries.
The test sequence on GPU 2 was untouched production, candidate off, candidate on, candidate off, candidate on, then untouched production again.
The first candidate-off stage additionally recorded startup linker bindings; its ordinary repeat provides a check without linker diagnostics.
The 17,070-token case is a negative control because the flag should not change attention selection there.

Across 72 requests including warm-ups, each measured sampling case produced the same output in all six stages.
The table averages the three per-sampling percentage differences, using the two off, two on, and two production measurements for each case.

| Actual input | Candidate on versus off | Candidate on versus production | Production before-to-after drift |
|---|---:|---:|---:|
| 7,073 | +1.34% | +0.44% | -2.37% |
| 14,069 | +2.29% | +1.66% | -1.45% |
| 17,070 | +0.14% | +0.13% | -0.62% |

The repeatable off/on signal is small and confined to the expected short-context path.
Production drift is large relative to the net improvement, especially on the shortest fixture.
This is evidence worth retaining, but not a demonstrated universal improvement or a reason to ship the preload library.
The comparison does not establish completed-task quality, all attention shapes, or all GPU architectures.

## Packed DFlash2 vocabulary shortlist

A candidate GGUF was built using the author's 65,536-token shortlist.
All 81 original draft tensors were copied byte-for-byte.
The candidate adds an I64 `d2t` tensor and selected Q6_K rows from the target's output head, without requantization.
The target and original draft tokenizer vocabularies were checked for exact equality.
The new output head contains 275,251,200 bytes, approximately 262.5 MiB.
The original model files were preserved.

At the production 196,000-token requested allocation, the candidate failed to load with an out-of-memory error while allocating a 343.63 MiB compute buffer.
The following comparison therefore gives both baseline and candidate the same 131,072-token allocation on GPU 3.
Baseline was run before and after the candidate; percentages compare with the mean of those two baselines.
Warm-up requests are excluded.

| Actual input | Sampling | Baseline tok/s | Shortlist tok/s | Change | Same output as both baselines |
|---|---|---:|---:|---:|---|
| 7,073 | Seed 6100 | 96.24 | 98.09 | +1.91% | Yes |
| 7,073 | Seed 731 | 80.84 | 83.29 | +3.03% | Yes |
| 7,073 | Greedy | 90.39 | 93.17 | +3.08% | Yes |
| 14,069 | Seed 6100 | 81.82 | 84.03 | +2.71% | Yes |
| 14,069 | Seed 731 | 66.20 | 67.41 | +1.82% | Yes |
| 14,069 | Greedy | 87.59 | 86.61 | -1.12% | Yes |
| 64,075 | Seed 6100 | 68.24 | 69.88 | +2.40% | Yes |
| 64,075 | Seed 731 | 66.62 | 68.38 | +2.64% | Yes |
| 64,075 | Greedy | 71.22 | 72.15 | +1.30% | No |

This is a small speed signal with a material memory regression, not a production-ready replacement.
Matching these short outputs does not establish broad vocabulary coverage, quality equivalence, or multilingual behavior.
In the longer greedy case, both baseline runs produced the same output and draft counts, while the shortlist diverged partway through reasoning.
That row is not an equivalent-work speed comparison.
The cause of divergence was not established; it should not be described as proof of degraded task quality, but it prevents claiming verified greedy equivalence.
The author's indexed-row MTP implementation avoids a separate packed head; a comparable DFlash2 implementation could be worth investigating if subsequent tests justify it.

## Graph-cache hypothesis

The cache is pointer-keyed, but a graph UID mismatch does not automatically trigger capture.
The runtime compares node properties, shapes, strides, and data pointers before deciding whether an update is required.
Partial draft acceptance alone does not establish that this comparison fails every round.

Fresh Nsight traces captured cached decoding on realistic source-plus-request fixtures with natural partial acceptance.
Draft acceptance was approximately 47.8% and 42.6%, respectively.

| Actual input | Graph launches | Captures | Executable updates | Update API time | Graph destruction API time |
|---|---:|---:|---:|---:|---:|
| 14,069 | 519 | 4 | 4 | 5.01 ms | 2.98 ms |
| 64,075 | 556 | 9 | 9 | 8.69 ms | 4.59 ms |

Neither trace contained `cudaMallocHost` or `cudaFreeHost` calls.
These counts do not support the report's claim of a recapture on every acceptance-width bounce.
API durations alone do not measure all CPU graph construction or GPU idle time, so they do not prove zero remaining graph overhead.
No shape-keyed cache patch was implemented, and no 5% gain has been demonstrated here.
The existing ten-second idle eviction policy could still affect pauses between tool turns; these uninterrupted traces do not evaluate that scenario.

## Other claims remain unproven

The report's 100% draft acceptance fixture is not representative of these natural coding requests.
Its 113-142 tok/s figures cannot establish that DFlash2 universally beats the article's MTP measurements or preserves better quality.
Hardware, input length, generated length, output distribution, sampler, and acceptance all matter.

Shared target/draft scratch, a reserved converted-KV pool, Turbo3 attention, and GDN changes were not ported or benchmarked in this evaluation.
The proposed 2.8 GB memory saving must not be treated as additive or verified for this runtime.
Its supported long-context decode already bypasses full K/V conversion through direct Q8 attention paths.

## Recommended next engineering target

If pursuing another runtime optimization, prioritize a memory-neutral indexed-row DFlash2 shortlist experiment over a speculative graph-cache rewrite.
First investigate the observed greedy divergence and validate coverage on completed coding, tool-use, multilingual, and unusual-token tasks.
Then require a same-card improvement at the existing 196K allocation before promotion.
For Q8, a clean integrated build followed by longer randomized comparisons would determine whether the small net gain survives without the preload experiment's build differences.
Neither path justifies reducing reasoning effort or changing the already-tested thinking instruction.

## Final state

Both benchmark containers were stopped and removed after testing, and GPUs 2 and 3 were free.
The unrelated production worker on GPU 1 was left running.
The central configuration SHA-256 remained `9da896acf2d0686bd7e523b895a44a4ba05686d5249245e36f74ca81e4dd534d`.
Only this report was added to tracked project scope; experimental models, the preload library, recipes, and raw results remain in the ignored diagnostics directory for reproduction.
No production model, configuration, runtime binary, or source edit was deployed.

## External references

- [llamAmpere Qwen Ampere documentation](https://raw.githubusercontent.com/JakeATX/llamAmpere/main/QWEN_AMPERE.md).
- [llamAmpere v0.2 writeup](https://raw.githubusercontent.com/JakeATX/llamAmpere/main/docs/llamampere-v0.2/ARTICLE.md).
- [MTP vocabulary shortlist design](https://raw.githubusercontent.com/JakeATX/llamAmpere/main/docs/mtp-vocabulary-shortlist.md).
- [65,536-token shortlist used by this experiment](https://raw.githubusercontent.com/JakeATX/llamAmpere/main/docs/mtp-vocab/atx_65536.txt).
