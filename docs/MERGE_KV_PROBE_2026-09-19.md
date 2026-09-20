# Merged-model KV-cache probe

## Decision

Keep the current Q8-key/Q8-value cache for local use.
Q4/Q4 saved approximately 3 GiB of GPU memory and passed this small accuracy screen, but did not demonstrate a consistent speed improvement.
No production configuration or personal model selection was changed.

## Method

The installed Marathon CLI executed all scored tasks, including its real instructions, tool schemas, router and tool execution.
An isolated proxy selected the inference worker, pinned seeds and captured requests and timing events; it did not implement a substitute agent loop.
The GPU-management skill guided worker isolation and preservation of active workloads.
The control used the registered production GPU 1 worker; candidates used a private broker only after unloading that control.
The user's active Bonsai worker on GPU 3 was not interrupted.

- Hardware: the same RTX 3090 Ti, GPU 1, with its existing 275 W power limit.
- Target: Swift Qwen3.8 27B Uncensored Merge IQ4_XS, unchanged weights.
- Drafter: installed Marathon R32 Q4_K_M V1, unchanged; this was not V2.
- Runtime: `sha256:765a84864664d953cc274adb0fdb161b793269857e4602d228460a790919cea8`.
- Allocated context: 196,000 tokens; medium reasoning remained enabled in captured requests.
- Sampling: temperature 1.0; retrieval seeds 41 and 73.
- Only target KV types changed, apart from isolated container names, ports, slot directories and workspace paths.

Each completed variant ran four retrieval sessions, two prose sessions and two tool tasks.
Retrieval requested three exact random codes planted at approximately 10%, 50% and 90% of a synthetic archive, with actual occupied input lengths around 32,725 and 131,033 tokens including Marathon's instructions.
Those percentages refer to the archive, not the entire system-plus-user prompt.
Q4/Q4 had four additional input tokens in retrieval/prose because its isolated workspace path differed.
Seeds repeat the same archive at each length, so twelve checked values are not twelve independent tasks.
There was no context truncation reported in completed retrieval runs.

## Results

| Measurement | Q8/Q8 control | Q4/Q4 candidate |
| --- | ---: | ---: |
| Sampled GPU memory during inference | 23,980 MiB | 20,916 MiB |
| Exact retrieval values, 32K, two seeds | 6/6 | 6/6 |
| Exact retrieval values, 128K, two seeds | 6/6 | 6/6 |
| Arithmetic with actual command execution | Pass | Pass |
| Parser with independent invalid-input checks | Pass | Pass |
| 32K retrieval decode, seeds 41 / 73 | 82.58 / 97.02 tok/s | 76.51 / 86.37 tok/s |
| 128K retrieval decode, seeds 41 / 73 | 74.65 / 74.19 tok/s | 58.87 / 63.93 tok/s |
| 32K prose decode | 51.92 tok/s | 67.72 tok/s |
| 32K prose CLI completion | 21.05 s | 35.88 s |
| 128K prose decode | 65.23 tok/s | 43.69 tok/s |
| 128K prose CLI completion | 43.70 s | 50.18 s |

Decode rates are the server's native evaluation timings, including generated thinking tokens, not final-prose-only speed.
The prose requests reused the archive prefix cached by preceding retrieval runs.
The 32K prose candidate generated 2,293 output tokens versus 990 for the control, illustrating why higher decode throughput can still yield a slower answer.
At 128K, the candidate generated 2,081 output tokens versus 2,689 for the control, but decoded more slowly.
Prose was manually inspected for coherence, not independently scored for literary quality or exact 400-word compliance.
Word counts were 487/404 for control and 421/382 for candidate at 32K/128K respectively.

Cold 32K prompt processing was 876.60 tok/s for Q8/Q8 and 865.16 tok/s for Q4/Q4.
The 128K requests reused 7,773 prefix tokens and processed their remaining approximately 123K tokens at 613.78 versus 596.80 tok/s.
Candidate first-request CLI time includes worker startup while the control was preloaded for tokenization, so those cold CLI totals are not a fair model-speed comparison.
The parser completed faster with Q4/Q4 in this attempt, 15.29 versus 19.92 seconds, using fewer inference turns; arithmetic was approximately six seconds for both.
These samples do not establish an across-workload speed ranking or a causal reasoning-quality change.

## Mixed Q8-key/Q4-value attempt

The mixed candidate reduced sampled GPU memory to 21,594 MiB, but prompt processing became dramatically slower.
It processed only 7,773 tokens in 77.84 seconds, compared with the control's complete 32,725-token prompt in 37.33 seconds.
The run was deliberately interrupted before completing its first answer rather than spending much longer on the 128K case.
No accuracy result exists for the mixed candidate; cancellation is not a model failure.

The local runtime source's CUDA flash-attention selector rejects unequal K/V types when `GGML_CUDA_FA_ALL_QUANTS` is disabled, and the local build cache records that flag as OFF.
This is a plausible explanation for an inefficient fallback, not a profiler-confirmed attribution to the exact loaded binary.
The same source also contains a small-batch SM86 optimization explicitly restricted to Q8/Q8, another reason not to assume reduced precision improves speed on this runtime.
No runtime rebuild or kernel modification was attempted.

## Limits and evidence

Passing these tests does not prove lossless quantization, broad coding/reasoning equivalence or reliable retrieval at 196K/260K occupied context.
This was one machine, one runtime, one archive per length, two retrieval seeds and two small tool tasks.
It was a bounded screening experiment, not a promotion-quality statistical benchmark.
Preserving weights and medium reasoning avoids two major confounds, but sampling, differing generated text, workspace paths and sequential execution still limit causal speed claims.

The reusable runner is `scripts/evals/merge_kv_probe.py`.
Local evidence includes captured requests, actual CLI events, answers, independent parser checks, protocol files and native server logs:

- `.marathon/diagnostics/merge-kv-20260919-run2/`: completed control and interrupted mixed-format candidate.
- `.marathon/diagnostics/merge-kv-20260919-q4q4/`: completed Q4/Q4 comparison.
- `.marathon/diagnostics/merge-kv-20260919/`: initial proxy-discovery setup failure, with no model-quality inference.

The production configuration hash before testing and at completion was `a4c63f48d5a3d26d220f81de9e52b7664895eb75c20959c7b08deef3ff6a203d`.
Experiment workers were unloaded and their GPU reservation released.
No training data, model weights or existing user artifacts were deleted.
