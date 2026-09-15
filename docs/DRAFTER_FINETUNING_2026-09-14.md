# Custom DFlash2 fine-tuning, 2026-09-14

The initial targeted pilots did not improve accepted block length, and their exported candidate was 1.22% slower in aggregate request latency.
A broader rank-16 LoRA run improved decode throughput by 5.9% on matched synthetic tool prompts, but showed no gain on plain coding or completed Marathon tasks.
A later rank-32 mixed-workload checkpoint improved held-out agent-state accepted block length by 26.02% and matched serving decode throughput by 27.15%.
That rank-32 checkpoint is the selected Marathon drafter.
The gain is concentrated in structured agent and tool continuations and does not establish a universal speedup or a global Pareto optimum.

## What was trained

The starting checkpoint was `incoai/Qwen3.8-27B-DFlash2` at revision `dedf8df68adfb1afeaf7b7480c0a0243108177b4`.
Its downloaded weight file was verified against SHA-256 `67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c`.
All 81 tensors loaded without missing, unexpected, or mismatched keys.
The model contains 1,924,404,480 parameters.

Two runs trained the three selector tensors, totaling 128,450,560 parameters.
Two runs trained the feature projection matrix, totaling 131,072,000 parameters.
These are targeted full-matrix updates, not full-model training or LoRA.
All other draft parameters and the entire target model stayed frozen.

Each mode used learning rates of `1e-4` and `1e-5`, AdamW, weight decay `0.01`, gradient clipping at `1.0`, and 108 optimizer steps.
The frozen backbone used BF16; trainable parameters and Adam states used FP32 with BF16 autocast.
Training used SDPA, 16 sampled anchors per example, eight-token blocks, and two-block objective chunks.
The projector runs used activation checkpointing.
The implementation uses the DFlash2 training code from pinned SpecForge revision `3d64e7a61f5fcc7f7d78ba6164c881f831943947`.

Peak PyTorch allocated memory was 10.49 GiB for selector training and 10.54 GiB for projector training.
These figures exclude CUDA context and allocator-reserved memory and do not describe full-model training requirements.
The runs used GPUs 2 and 3 independently.

## Exact-target feature capture

A small native extractor links to libraries copied from the deployed runtime image, `sha256:765a84864664d953cc274adb0fdb161b793269857e4602d228460a790919cea8`.
It loads the deployed uncensored IQ4_XS target directly, with Q8 target KV and flash attention.
The extractor captures inputs to layers `[6, 20, 34, 48, 62]`, corresponding to post-layer outputs `[5, 19, 33, 47, 61]` in the training configuration.
Each token therefore has 25,600 captured feature values.
All captured values were checked for finiteness.
An independently repeated first example produced identical token IDs and bit-identical captured features.

The frozen output projection and token embeddings were dequantized from the same deployed GGUF and converted to BF16 for training.
This uses the actual quantized target's features and weights, but the training arithmetic is not bit-identical to llama.cpp's quantized matrix operations.
The draft backbone starts from the official BF16 checkpoint; this pilot does not implement quantization-aware training of the draft.

## Data and validation boundaries

The pilot contains 24 synthetic coding prompts rendered through the current runtime's chat template.
The target generated their continuations greedily, with a 384-token generation limit per example and thinking disabled.
Training used 18 examples containing 6,608 generated tokens.
Validation used six separate examples containing 2,304 generated tokens.
Prompt tokens were excluded from the training loss.
These short examples establish pipeline feasibility and screen candidate updates; they are not a representative corpus of full Marathon tool trajectories.

Validation sampled the same 16 anchors per example with fixed seeds, for 96 blocks per evaluation.
The reported accepted length includes the anchor and up to seven proposed tokens, so its maximum is eight.
Production uses six proposals, and the offline metric must not be interpreted as production tokens per second.
Repeated evaluation and learning-rate selection use this validation set; the later serving fixtures are separate.

## Offline results

| Trainable component | Learning rate | Stock | Step 36 | Step 72 | Step 108 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Selector | 0.0001 | 6.1771 | 6.0000 | 6.0729 | 6.0000 |
| Selector | 0.00001 | 6.1771 | 6.1354 | 6.0833 | 6.0833 |
| Feature projection | 0.0001 | 6.1771 | 6.1146 | 6.0729 | 6.0729 |
| Feature projection | 0.00001 | 6.1771 | 6.1771 | 6.1771 | 6.1771 |

The low-rate projector run reduced validation token cross-entropy from 1.1675 to 1.1567 by step 108, but accepted block length did not increase.
Lower training or validation loss alone is therefore insufficient evidence of a serving gain.
The midpoint checkpoint from that tied run was selected for the serving check.

## Quantized export and serving protocol

The candidate export changes only `fc.weight`.
All other tensor bytes are identical to the production drafter, and all tensor shapes and quantization types are preserved.
The export copies the production GGUF metadata, including its rotary and convolution settings.
The candidate is 1,143,006,752 bytes with SHA-256 `80922f565345126b515568382b44aea6894b4950009a5fa738438c24cd209ced`.
No trained checkpoint has been promoted to production.

The serving comparison uses the deployed runtime libraries and an isolated worker on GPU 1 after its registered worker unloaded.
It retains the 196,000-token context allocation, target and draft KV types, six proposals, CPU vision projector, batch sizes, sampling options, and power limits.
The ordering is stock, candidate, candidate, stock on the same GPU.
Marathon's worker lease prevents another Marathon process from claiming that worker, and a monitor terminates the experiment if a conflicting registered workload appears.
The central llama-swap configuration is not changed.

Two saved Marathon request fixtures test 512-token generation at temperatures zero and one, with fixed seeds and warm-prefix reuse.
Those capped fixtures measure decoding and request latency, not completed agent tasks.
Separate Python execution, SQL execution, structured extraction, and tool-call checks evaluate completed responses at both temperatures.
The response cap for those correctness checks is 2,048 tokens.

## Completed serving results

All four stages completed, for 52 requests: 24 timed warm-prefix performance probes, eight warmups, and 20 correctness checks.
The warmups are excluded from the performance totals below.

| Measurement | Stock | Trained projector |
| --- | ---: | ---: |
| Timed performance requests | 12 | 12 |
| Generated performance tokens | 6,144 | 6,144 |
| Total request time | 74.1546 s | 75.0618 s |
| Total decode time | 70.7675 s | 71.7455 s |
| Generated tokens / decode second | 86.8195 | 85.6361 |
| Accepted / proposed draft tokens | 4,486 / 9,728 | 4,470 / 9,818 |
| Draft acceptance rate | 46.1143% | 45.5286% |
| Correctness checks passed | 10 / 10 | 10 / 10 |

The candidate increased request time by 1.22% and decode time by 1.38%, reducing decode throughput by 1.36%.
Five of the six fixture/seed/temperature combinations were slower on average; the sixth differed by less than 0.04%.
All four repeated outputs were identical within each performance combination, so the fixed-output performance comparison is not explained by different answer lengths.
These measurements provide no evidence of the desired 10% gain.
The sample is small and does not establish precise population-level performance bounds.

All ten completed correctness responses had identical final answers or tool-call contents after ignoring generated tool-call IDs.
One greedy debugging response had different reasoning text and generated 937 tokens instead of stock's 1,053, while returning the identical correct function.
Therefore these results do not prove bit-for-bit generation preservation or universal losslessness.
The cause of that reasoning divergence was not isolated by this experiment.
The ten passing checks also do not establish unchanged accuracy across Marathon's full workload distribution.

The benchmark exited successfully and released its worker lease.
Final GPU inspection showed GPUs 1, 2, and 3 idle with 24, 15, and 15 MiB used respectively.

## Reproducibility and retained artifacts

The isolated workspace is `.marathon/drafter-training/`.
It contains the pinned source, dependency lock, verified stock weights, native capture program, training and export programs, GPU lease wrapper, benchmark runner, captured data, checkpoints, and raw results.
`manifest.json`, `code-hashes.json`, `capture-validation.json`, `training-summary.json`, and the export manifest record provenance and checks.
`projector-serving-abba/summary.json` contains aggregate and paired results, and its sibling response files and server logs retain the underlying evidence.
Production application code, model files, broker configuration, GPU assignments, and power limits were not modified.

The experiment found no useful speedup or quality improvement and does not establish a global Pareto optimum.
A larger training effort would need representative Marathon trajectories and a separate final test set; the current results cannot establish whether that would succeed.

## Broader synthetic tool training

The second round uses only synthetic conversation content, tool arguments, and tool results.
Private Marathon conversation messages, argument values, and tool output contents are excluded.
Runtime telemetry was restricted to numerical performance fields and tool names.
The metadata sample contains 1,895 responses, including 1,645 responses containing tool calls.
Context length differs substantially between the tool and text subsets, so their aggregate acceptance rates do not establish a causal tool-format penalty.

The long synthetic corpus contains 40 training cases and 12 validation cases from disjoint task families.
Prompts contain approximately 10,900 to 11,000 tokens, with 16,639 generated training tokens and 4,146 generated validation tokens.
The target captures retain the deployed IQ4_XS model and runtime.
BF16 feature files use memory mapping to limit duplicate host memory.

LoRA trains 8,355,840 parameters across 36 attention, MLP, and feature projection matrices, using rank 16, alpha 32, learning rate 0.0001, and 240 steps.
Both runs use the same data and fixed validation anchors.
The ordinary token loss and prefix-weighted D-PACE loss are compared below.
Peak allocated GPU memory was 11,832,630,272 bytes for these runs.

| Objective | Stock | Step 80 | Step 160 | Step 240 |
| --- | ---: | ---: | ---: | ---: |
| Ordinary DFlash | 5.738715 | 6.053819 | 6.092014 | 6.145833 |
| Prefix-weighted D-PACE | 5.738715 | 6.037326 | 6.142361 | 6.190104 |

These accepted lengths include the anchor and up to seven proposals.
They measure BF16 offline draft behavior, not quantized serving throughput or task accuracy.
The best D-PACE checkpoint is selected for quantized serving tests, with a separate six-proposal offline check to match Marathon's production window.
The later serving tasks were not used to select the checkpoint.

## Other second-round screens

The calibrated Q3 drafter from `andrew-paul/Qwen3.8-27B-DFlash2-Q3_K_M-GGUF`, revision `0f4bd6266e7deae539079e8bf0650ca5487711f1`, was verified against SHA-256 `c46514c25f79a5663526bcefe5846881328f60b4434c0fad22ab039e466a8e65`.
It saves 226,303,744 file bytes compared with stock Q4.
The completed ABBA comparison measured 81.6633 tokens/s for stock and 81.6487 tokens/s for Q3, with Q3 request time 0.66% worse.
Both variants passed all 10 separate correctness checks with identical final answers.
Concurrent work on another GPU limits interpretation of tiny timing differences, but this screen supplies no evidence of a substantial speed gain.
A previous attempt failed from a CUDA graph allocation OOM in the stock control before Q3 loaded; the completed comparison separates performance and quality into fresh workers.

A selector lookahead policy sweep improved the best offline accepted length by only 0.65% across eight settings.
This small validation-selected change does not justify modifying the serving engine at this stage.
Evidence remains under `.marathon/drafter-training/`, including scripts, immutable source checkpoints, synthetic captures, evaluation files, export manifests, and serving logs.

## Broader checkpoint serving results

The selected D-PACE step-240 export is 1,143,006,752 bytes, SHA-256 `49f3c8d37eac64d08dd50647921894e7dd621840ca53ab42d32fe95cb8a963e2`.
It changes the 36 adapted matrices and preserves all other tensor bytes, shapes, quantization types, and production metadata.
The six-proposal offline check measured accepted length 5.317708 for stock BF16 and 5.627604 for the adapted BF16 checkpoint, an improvement of 5.83%.

Three serving comparisons use GPU 2, the deployed runtime, 196K context, and stock/trained/trained/stock ordering.
Other experiment GPUs were idle during these timing comparisons.
No central inference configuration or production model was changed.

| Workload | Stock decode tokens/s | Trained decode tokens/s | Trained total wall-time change |
| --- | ---: | ---: | ---: |
| Plain coding, 12 timed 512-token probes per variant | 82.9683 | 82.6358 | +0.40% |
| Completed Marathon tasks, four per variant | 83.1563 | 82.7743 | +28.31% |
| Matched synthetic tool prompts, eight per variant | 98.2962 | 104.1081 | -1.50% |

The agent comparison invokes the actual Marathon router and installed hardened frontend, with isolated synthetic workspaces and histories.
It reuses the independent `recovery` and `retry` task oracles in `scripts/evals/prompt_audit.py`.
All eight tasks passed and preserved protected files.
Stock generated 10,360 output tokens using 24 tool calls in 168.2868 seconds; the trained drafter generated 12,947 tokens using 30 tool calls in 215.9312 seconds.
The sample is small and trajectories differ, so its observed 28.31% wall-time regression is not a precise estimate of general production impact.
It does rule out claiming a demonstrated end-to-end win from these runs.
The initial per-task logger incorrectly looked for `completion_tokens`; the final summary recomputes output counts from the router's `output_tokens` field and excludes its separate warmup event.
Timing and oracle checks were unaffected.

The matched tool comparison uses one case from each of the four held-out families, with approximately 11K context and greedy generation to EOS.
All 20 requests, including four warmups, ended at EOS without truncation.
The trained checkpoint improved aggregate decode throughput by 5.91%, but output count increased from 2,272 to 2,364 tokens, leaving only a 1.50% wall-time reduction.
Two prompts, `transpose_0` and `rotate_0`, produced identical output across all four stages.
On this identical-output subset, both variants generated 626 tokens; total wall time decreased from 8.6092 to 8.0659 seconds, or 6.31%.
This subset is a diagnostic comparison, not an independent broad-workload validation set.

The result is a narrow serving gain that does not yet generalize to completed Marathon tasks.
The checkpoint is retained for further investigation and is not promoted.

## Quantization diagnostic

The final diagnostic imports all 81 tensors from each serving GGUF into the same offline evaluator, dequantizing them to BF16 and preserving the fixed captured target features and validation anchors.
The deployed converter was checked for layout transformations; this checkpoint uses the default NeoX rotary layout and does not require the converter's optional interleaved-rotary permutation.
With six proposals, the stock Q4 weights scored 5.288194 accepted tokens per block and the trained Q4 weights scored 5.625000, a 6.37% improvement.
The corresponding pre-quantization results were 5.317708 and 5.627604.
Quantization therefore did not erase the measured offline improvement on these cases.
This diagnostic uses BF16 arithmetic on dequantized weights and does not reproduce quantized runtime kernels or draft KV exactly.

The most useful next hypothesis is broader workload coverage: mix varied synthetic coding and multi-step tool trajectories, then validate on completed tasks that were not used for checkpoint selection.
That is an inference from the workload-specific results, not proof that a larger or more diverse training run will deliver a general gain.
All experiment workers were released after validation, and the permanent inference configuration was preserved.

## Selected rank-32 mixed-workload checkpoint

The final selected run mixes synthetic coding, long tool, and agent-state feature sets.
It uses rank-32 LoRA with alpha 64 across attention, MLP, and feature-projection matrices, a learning rate of `7e-5`, the prefix-weighted D-PACE objective, and 480 optimizer steps.
The training corpus remains synthetic-only and contains no private Marathon conversation content.

The Q4_K_M export changes 36 adapted tensors.
Every other tensor byte, tensor shape, quantization type, and GGUF metadata field matches the stock Q4_K_M drafter.
The selected file is 1,143,006,752 bytes with SHA-256 `e096aa09c26d5096b63b1a4d0400258819980b16250ef5c5d08fcf830e6bb6a6`.

On the held-out synthetic agent-state set with six speculative proposals, stock Q4 accepted 4.8463 tokens per block and the selected checkpoint accepted 6.1074, a 26.02% increase.

The matched serving comparison ran 40 measured requests per variant with the same target, runtime, sampler, and six-proposal configuration.

| Measurement | Stock Q4_K_M | Selected rank-32 Q4_K_M | Change |
| --- | ---: | ---: | ---: |
| Decode throughput | 101.0067 tok/s | 128.4254 tok/s | +27.15% |
| Total request time | 142.1430 s | 135.7699 s | -4.48% |
| Draft acceptance | 61.77% | 84.01% | +22.24 percentage points |

Nineteen of twenty matched cases produced identical output bytes across repeated stock and selected-checkpoint runs.
The remaining case differed by two generated tokens.

A separate generic coding comparison generated exactly 6,144 tokens per variant.
It measured a 0.99% decode-throughput increase and a 0.95% wall-time decrease.
The selected checkpoint therefore has strong evidence for a narrow structured-agent gain and only weak evidence for improvement on general coding output.

Both variants passed all ten separate coding, debugging, extraction, SQL, and tool-use quality checks with identical final answers.
The selected checkpoint also passed four of four deterministic synthetic agent tasks, but those trajectories differed and do not provide a controlled speed comparison.
