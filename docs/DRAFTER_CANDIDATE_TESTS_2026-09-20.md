# Draft candidate investigation, September 20, 2026

Completed: retain the current R32 draft at six proposals.
No tested replacement established a meaningful everyday decode improvement, and production settings remain unchanged.
Jev selected the same deployment decision after receiving the measured results and caveats.

## Conversation screen results

| Draft/configuration | Explanation tok/s | Planning tok/s | Fiction tok/s |
| --- | ---: | ---: | ---: |
| Current R32, six proposals, microbatch 256 | 64.79 | 55.45 | 59.89 |
| R32, six, microbatch 128, timing instrumentation | 64.46 | 55.63 | 60.86 |
| R32, six, corrected Dao-compatible overlay control | 65.42 | 55.38 | 60.63 |
| R32, seven, microbatch 128 | 64.96 | 54.33 | 60.50 |
| b32, six, microbatch 128 | 62.63 | 55.46 | 62.77 |
| b32, eight, microbatch 128 | 55.46 | 45.44 | 47.95 |
| DaoCloud adapted, six, microbatch 128 | 57.82 | 53.53 | 53.53 |
| DaoCloud adapted, seven, microbatch 128 | 60.07 | 52.59 | 53.48 |
| Apathy unmodified, two, microbatch 64 | 50.30 | 46.91 | 51.40 |
| Apathy draft window 4096, six, microbatch 128 | 55.80 | 48.56 | 52.81 |
| DSpark draft window 8192, six, microbatch 128 | 42.13 | 41.15 | 41.83 |

These are single-pass screens on the same GPU, not confidence intervals or an exhaustive ranking of all possible configurations.
Small differences are not established gains, and some sampled output paths differ.
The independently checked quality suites passed 12/12 for R32, b32 at six proposals, and the adapted DaoCloud at six proposals.
Apathy and DSpark did not pass the speed screen, so no expensive near-capacity qualification or expanded quality suite was run for their windowed variants.

## Final Jev decision

The final measured-evidence review chose `retain_r32` for deployment and found no established additional 10-20 percent conversation gain.
It retained a bounded DFlare-style per-layer conditioning prototype as a future research direction, not a ready replacement or a measured winner.
It did not recommend spending near-capacity test time qualifying the losing candidates before retaining unchanged production.
The exact request and response are `measured-review-request.json` and `measured-review-response.json` under `.marathon/diagnostics/jev-drafter-frontier-20260920`.
Jev's choice weights are advisory judgments, not calibrated probabilities of engineering success.


## Protocol

All new GPU tests use an exclusively leased, initially unloaded GPU 3, a single regular RTX 3090 at its unchanged 250W power limit.
The merged IQ4_XS target, Q8/Q8 target KV, Q4/Q4 draft KV, CPU vision, and 196000 configured context capacity remain fixed.
Each scratch worker uses a fresh isolated cache directory and is removed before the next comparison.
Private Marathon conversations and snapshots are excluded.
Weights and converted GGUFs are stored outside the repository under `/home/deforest/AI/experiments/drafter-frontier-20260920`.

The conversation screen uses three generated prompts, medium reasoning, temperature 0.7, fixed seed, a warmup, and a 768-token output cap.
These are throughput screens, not completed-task accuracy tests or measurements at 196000 occupied tokens.
Different output hashes limit direct speed comparisons.
The separate accuracy screen covers six tasks at two sampling temperatures: structured edits, latest-revision selection, arithmetic, Python interval merging, tool arguments, and document retrieval at roughly 18K input tokens.
The Python checker runs 304 independently checked cases inside a restricted process.
Passing this small screen does not establish general quality equivalence.

## Reproduced memory constraint

Seven proposals at the production microbatch of 256 completed a tiny request but failed on a larger prompt during CUDA graph instantiation.
Eager CUDA module loading moved the failure to draft compute-buffer allocation, rather than resolving it.
A microbatch of 128 allowed the seven-proposal copy task to complete twice with exact output checks, at 188.68 and 190.29 decode tokens/s.
The 512-token peak windows were 196.84 and 195.49 tokens/s.
This is feasibility evidence on short synthetic input, not a new long-context or normal-chat speed record.

The source computes recurrent rollback storage from the neural proposal count.
Increasing the draft width therefore increases target-side memory as well as draft work.
B32 widths 11 and 15 failed before the draft weights could be allocated, even with microbatches of 64 and 128 respectively.

## Runtime cost audit

An isolated diagnostic common library measured 901 small feature-processing calls.
Mean host time was 863 microseconds: approximately 85 for feature gathering, 452 for encoding and output retrieval, and 324 for injection submission.
Calls averaged 6.86 rows and include warmups across three tasks.
These are host timings with synchronization effects, not a complete GPU kernel attribution.
The measured path alone is too small to explain most of the roughly 40ms verification round.
The diagnostic library preserved all three baseline conversation output hashes and acceptance counts.

## Candidate compatibility

JonasLoos b32 converts with the production DFlash2 schema and retains five layers, 2048 sliding windows, and trained block metadata of 32.
Its Q4_K_M tensor-type choices match the deployed R32 draft.
Apathy v3 has six layers, with five 4096 sliding-window layers and one full-attention layer.
Its approximate draft KV alone is 238MiB at 196000 tokens, compared with roughly 11MiB for the five-layer 2048-window design, before allocation padding and other buffers.
DSpark Agentic uses five full-attention layers; its measured KV allocation request was about 1077MiB despite smaller model weights.
Apathy failed draft compute allocation at width six, and DSpark failed at widths six and two with a microbatch of 64.

DaoCloud's published configuration uses nested transformer metadata and an anchor-first seven-query layout.
The production DFlash2 selector constructs an unused row zero and begins proposals at row one, so metadata normalization alone is insufficient.
An experimental source overlay adds explicit anchor-first lattice construction and sampling while preserving the original default layout.
Its first baseline regression caught an inherited DSpark default applying to DFlash2; that diagnostic result is invalid for candidate ranking.
The corrected adaptation reproduced all three R32 baseline response hashes and accepted/generated draft counts before candidate measurements began.
It remains an experimental local port, not a validated upstream implementation.

DFlare has no compatible released 27B checkpoint found in this search and would require new architecture support and training.
Its paper's full 8B-target training used 32 GPUs for roughly 160 hours; this is not a minimum requirement for a small prototype, but the published result cannot be reproduced by adding a layer to our GGUF.
DeLS-Spec similarly requires a new local head and runtime integration; its paper compares against DFlash rather than our already transition-conditioned DFlash2.
Simply adding an untrained sixth layer, changing long-context metadata, or re-encoding existing Q4 values as Q8 does not establish additional predictive capability.
The selected R32 adapter weight file is absent from its local checkpoint directory, so a higher-precision reconstruction also requires recovering its original training artifact rather than upcasting the deployed GGUF.

## Reproducibility

Harness: `scripts/evals/speculation_cost_probe.py`.
Per-run command, config hash, candidate hash, responses, timing data, and failure logs are under `.marathon/diagnostics/drafter-frontier-*-20260920`.
Pinned checkpoint revisions and hashes are in `.marathon/diagnostics/jev-drafter-frontier-20260920/checkpoints.json`.
Source overlays and build scripts are confined to diagnostic directories.
Public sources and initial Jev review are listed in `docs/JEV_DRAFTER_FRONTIER_2026-09-20.md`.

## Completed accuracy comparison

R32 and b32 at six proposals both passed all 12 independent task checks.
Nine of twelve complete response messages matched exactly.
Across those nine pairs, b32 had a median decode change of -0.92 percent in this single-pass screen.
The sampled coding response grew from 587 to 699 tokens with b32, despite both implementations passing all 304 functional checks.
A slightly higher decode rate on a different, longer output should not be reported as a task-time improvement.
These results do not justify replacing R32.

## Interpretation of the research-only candidates

A custom six-layer version of R32 with the same dimensions would contain about 2.257 billion parameters versus 1.924 billion now.
The Apathy experiment does not establish that every six-layer design is infeasible: its full-attention layer and different feature taps add costs that an all-sliding-window design would avoid.
No trained checkpoint for a six-layer extension of our exact R32 model was found or fabricated.
DFlare's per-layer feature fusion is a distinct mechanism worth keeping on the research list, but there is currently no local speed or quality result for it.
These designs cannot honestly be ranked as measured winners or failures.
The metadata-only long-context conversion and stock MLX quantization were screened out because they do not provide new predictive capacity for our existing 196K CUDA setup.

## Draft-only attention-window experiment

Jev recommended one bounded memory adaptation before rejecting the full-attention drafts.
The hypothesis limits only the draft's attention span, leaving the target context capacity and target weights/cache unchanged.
DSpark was given an 8192-token sliding window in all five layers; Apathy's full-attention layer was restricted to its existing 4096-token sliding-window span.
All draft weight tensor bytes were verified identical after writing these explicitly labeled metadata variants.
This changes inference behavior relative to the published checkpoints, so it is not evidence about their unmodified performance and requires validation beyond short context.
The hypothesis is that target hidden features still carry global information while the drafter uses a bounded recent history for proposals.
It does not establish unchanged draft acceptance at long contexts, and no variant should be deployed solely on a short-prompt result.

DaoCloud's adapted six-proposal variant also passed all 12 accuracy checks, but was slower in the three conversation screens.
R32 at seven proposals produced 64.96, 54.33, and 60.50 tokens/s on explanation, planning, and fiction respectively.
This does not establish a meaningful improvement over the existing six-proposal baseline.
The unmodified Apathy model also failed at three proposals with microbatch 32; two proposals was the only tested unmodified configuration that ran.
DSpark still failed at two proposals and microbatch 32 before the draft-window adaptation.
With an 8192 draft window, DSpark ran six proposals at 41.15 to 42.13 tokens/s, so memory feasibility did not translate into a speed advantage.
No long-context deployment gate was run for this losing variant.

## Final state and artifact retention

GPU 3 was released and no synthetic worker remains running.
The GPU-control configuration and repository are unchanged; power limits remain 230/275/250/250W.
Original revision-pinned source checkpoints, final Q4 exports, variant manifests, source patches, and measurement logs are retained for reproducibility outside Git where appropriate.
Temporary BF16 GGUF conversion intermediates were removed after final quantized exports and their tests completed; they are reproducible from retained safetensors and conversion commands.
No private production conversations were accessed or submitted to Jev.
