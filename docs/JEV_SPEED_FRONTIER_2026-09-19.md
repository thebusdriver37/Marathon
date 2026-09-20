# Jev review of the existing Qwen speed evidence

## Correction after the user's speculation-history reminder

This correction supersedes the recommendation below to prioritize adaptive speculation and excludes the historical 300 W option.
The user sets an absolute 275 W ceiling and prefers leaving power unchanged.
No power setting was changed.
The [September 11 draft-policy investigation](DRAFT_ACCEPTANCE_EVALUATION_2026-09-11.md) already tested two-, four-, and five-token windows, with decode changes of -26.8%, -9.2%, and -2.1% relative to six.
Confidence cutoffs of 0.2 and 0.5 changed decode by -2.9% and -13.0%.
Seven-token drafting failed at full context, and the small Q8 draft-KV decode gain did not improve completed-task latency.
The September 7 media follow-up also compared three versus six across repeated workloads and selected six.
These records establish substantial prior speculation-policy testing, although they do not prove that every possible dynamic controller was tested.
I had not incorporated these specific records into the first Jev packet and overstated adaptive speculation as the strongest remaining software lead.
Availability of a request-level control is not evidence of a speed opportunity.
Keep the six-token policy and Q8/Q8 target KV; reconsider only if existing approved synthetic traces identify a specific reproducible mechanism beyond these exhausted settings.
An additional 14 Jev judgments reviewed this correction in `packet-correction.json` and `correction/`, bringing the cumulative review to eight calls and 280 judgments.
The sections below preserve the earlier review and should be read subject to this correction.

The most defensible next directions are a speed-oriented power policy and cost-aware speculative decoding, while retaining the current SWIFT+uncensored merge.
Further target or drafter training remains research, with no demonstrated additional gain on the current deployment.
The existing Q4 KV results do not support a general speed promotion.
No global Pareto optimum has been established.

## Scope and privacy

This review made six calls to pinned `jev-1.13.0`, receiving 266 typed judgments across an initial screen, evidence-driven follow-up, and final revision.
Reported API usage totaled 65,261 input tokens and 6,656 output tokens; no dollar cost was inferred.
There were no new GPU inference runs, training jobs, service restarts, power changes, or deployment changes.
Production conversation stores, production transcripts, and raw historical private replay payloads were excluded.
Jev received only authored aggregate evidence cards, configuration facts, synthetic-test measurements, and brief public research summaries.
No conversation text was transmitted.
Historical aggregate measurements from experiment reports were allowed, even where the underlying replay payload was excluded.
This is a targeted synthesis of relevant reports and selected raw synthetic measurements, not an exhaustive import of every artifact in the benchmark directory.

## Recommended order

| Direction | Evidence and next decision | Status |
| --- | --- | --- |
| Keep the current merged target | Preserves the user's preferred behavior and demonstrated small-suite token savings; do not replace it with an unvalidated efficiency fine-tune. | Current baseline, not a new gain |
| Choose a speed-oriented power policy | Historical same-card 3090 results support 275 W or 300 W over the current 250 W efficiency setting; exact current-merge uplift remains unknown. | Strongest measured incremental decode lever, with power cost |
| Cost-aware speculation | Current source and runtime documentation expose per-request `speculative.n_max` from zero through the configured six; the earlier claim that every request-level control was disabled is stale. | Best bounded software hypothesis; no measured policy gain yet |
| Target reasoning efficiency | Inspect synthetic failure patterns and correctness-conditioned efficiency mechanisms; additional training must improve on the merge, not on the original base. | Research, not a training recommendation yet |
| Target-matched drafter adaptation | Match the actual merged IQ4 target and account for serving numerics, while changing the failed objective materially. | Secondary research; no demonstrated mismatch requiring retraining |

For speculation, existing acceptance counters alone cannot establish an optimal window.
Choosing the policy requires separating draft cost, verification cost, and accepted tokens, ideally from existing synthetic traces.
The current review does not establish that those costs are separable in all saved traces.
Any later GPU validation should resolve that specific uncertainty rather than repeat a broad window sweep.
Larger-than-six proposals remain a separate memory and implementation question.

The historical regular-3090 power sweep measured 79.87, 85.04, and 87.93 decode tokens/s at 250, 275, and 300 W on one 15K workload.
Those are approximately 6.5% and 10.1% improvements over 250 W, with unchanged outputs and acceptance within that sweep.
At 190K occupied context the corresponding rates were 46.57, 49.55, and 51.38 tokens/s.
The Ti sweep measured 81.68 versus 87.85 tokens/s at 275 versus 300 W, with additional concurrency caveats.
These results come from the original target and stock-drafter era and are not promises for the current merge.
Power settings were not changed by this review.
See the [historical operating-envelope report](/home/deforest/Documents/DEV/gpu-control/qwen38-ceiling-2026-09-04.md).

## Findings that change the search

### SWIFT gains are real in the saved suite but concentrated

Reconstructing all 51 natural-completion pairs reproduced 173,804 baseline and 154,568 merged thinking tokens, or 11.0676% savings.
Seed 42 requires combining the original naturally completed controls with four higher-budget reruns, as the original reporting script does.
Reading only the seed-42 rerun file would silently omit 13 controls.
All 33 objective responses per model passed the original correctness checks, although that does not establish broad quality or behavior preservation.

One `restricted_permutations` problem across three seeds saves 21,723 tokens, exceeding the entire suite's net savings of 19,236 tokens.
Excluding that problem leaves 115,255 baseline versus 117,742 merged tokens, approximately 2.16% more for the merge.
This is a sensitivity analysis, not a reason to exclude that problem from the original result.
It shows why the three-seed average is not a general workload forecast.
The `binary_dp` and `villain_task` cases instead add 16,349 and 5,176 tokens respectively.
Do not turn these evaluation cases into training data or fit a routing policy to their outcomes.

The saved real-CLI synthetic tasks provide a different signal:

| Task | Original wall seconds | Merge wall seconds | Outcome |
| --- | ---: | ---: | --- |
| Intervals | 236.287 | 42.556 | Both passed |
| Miniboard | 192.157 | 97.172 | Both passed |
| Graph | 246.029 | 229.539 | Both failed hidden tests |

These are single attempts with different trajectories, predating later tool/router fixes.
They support investigating reduced work, not a universal speedup percentage.
Whole synthetic backend logs contain 28,557 versus 21,121 logged generated tokens and 371.025 versus 277.455 decode seconds.
They include warmups and failed-task work, so they are not a matched per-token benchmark.
Their accepted/proposed totals are 20,811/45,785 versus 15,285/34,452, which do not demonstrate a catastrophic drafter mismatch.
See the [merge experiment record](/home/deforest/AI/experiments/swift-uncensored/README.md).

### The deployed drafter is the original R32

The configured drafter's directly computed SHA-256 is `e096aa09c26d5096b63b1a4d0400258819980b16250ef5c5d08fcf830e6bb6a6`.
This matches the original R32 deployment, not the step-2,767 candidate hash `ee36e07d1124cceb629111cf7e55216aab35a8453545b008e36ec8d0429bc5d3` discussed in later research notes.
The later candidate's historical aggregate decode gain was only 0.38%, with essentially flat deep turns and some changed outputs.
This resolves an identity discrepancy without establishing that deploying the candidate would improve the merged target.
See the [pair-scale evaluation](DRAFTER_PAIR_SCALE_EVAL_2026-09-17.md).

### Q4 KV has no established broad speed win

An independently running synthetic probe finished while this read-only review was underway.
This review launched none of those requests.
Server timings were aligned to the saved requests by exact generated-token counts, avoiding estimates from streaming wall time.

| Synthetic case | Q8/Q8 decode tok/s | Q4/Q4 decode tok/s | Important qualification |
| --- | ---: | ---: | --- |
| 32K retrieval, seed 41 | 82.58 | 76.51 | Both correct |
| 32K retrieval, seed 73 | 97.02 | 86.37 | Both correct; different output length |
| 131K retrieval, seed 41 | 74.65 | 58.87 | Both correct; different output length |
| 131K retrieval, seed 73 | 74.19 | 63.93 | Both correct; different output length |
| 32K prose | 51.92 | 67.72 | Output grows from 990 to 2,293 tokens; wall time grows from 21.05 to 35.88 seconds |
| 131K prose | 65.23 | 43.69 | Wall time grows from 43.70 to 50.18 seconds despite fewer output tokens |

The parser task is faster in wall time, 19.92 to 15.29 seconds, while generating fewer tokens along a different trajectory.
Both variants passed the probe's retrieval, arithmetic, and parser checks; prose quality was not scored.
All workers allocated 196K, but the longest occupied prompts were approximately 131K.
Prompt lengths differ by four tokens between variants, and there is only one run per case/variant.
These results reject a blanket Q4 speed claim, not every possible Q4 use or future memory-enabled interaction.
The separately completed [KV probe report](MERGE_KV_PROBE_2026-09-19.md) records sampled memory falling from 23,980 to 20,916 MiB, approximately 3 GiB of headroom.
Using that headroom for a different verification configuration remains an unmeasured interaction, not an established combined speedup.
Cold first-request CLI times are additionally confounded because the control was preloaded for tokenization while the candidate paid startup cost.

Mixed Q8-key/Q4-value did start, contrary to the early interpretation of the proxy log.
Its saved server log shows very slow prompt processing, reaching 7,773 tokens after about 77.84 seconds before cancellation.
The inspected CUDA source returns no mixed-type fast-attention kernel unless `GGML_CUDA_FA_ALL_QUANTS` is enabled.
That is consistent with a fallback explanation, but the exact deployed compiler flag was not independently established here.
Do not treat that source observation as a proven diagnosis or repeat the same configuration blindly.

### Several attractive ideas are already implemented or exhausted

Direct GGUF inspection shows the deployed DFlash2 has five layers, each already using a 2,048-token sliding window, block size eight, selector rank 256, and selector top-k 16.
Adding that draft window is not a new optimization.
Scheduler reuse, prefix snapshots, and recent tool/router corrections are also already present and cannot be counted as future gains.
The [wasted-work report](MARATHON_WASTED_WORK_2026-09-19.md) records the distinction between reproducible infrastructure defects and model behavior.

The [completed deep-mining record](DRAFTER_DEEP_REJECTION_MINING_PLAN_2026-09-17.md) supersedes its own earlier recommendations.
More epochs, learning-rate changes, soft targets, exact-anchor selector calibration, and broader LoRA updates failed the relevant held-out serving gates.
A 22.2% same-fixture gain did not generalize.
Nearby kernel scheduling and tiling sweeps also failed.
These should remain excluded until a materially different mechanism or contradictory measurement appears.

## Focused research findings

[Quantize the Target, Quantize the Drafter](https://arxiv.org/abs/2607.04244) uses a high-precision-to-quantized-target training sequence and draft attention optimizations on Qwen3.5-4B/A10G.
The target-adaptation mechanism is relevant to investigate; its reported speedup does not transfer to this 27B IQ4_XS custom runtime.
Its sliding-window idea is already present locally.
The permitted adaptation here would preserve the user's target quantization, rather than adopting the paper's complete target-quantization pipeline.

[DeLS-Spec](https://arxiv.org/abs/2607.07409) adds an independently trained short-context head to a fixed DFlash model.
Local DFlash2 already has learned transition conditioning, so architectural overlap must be resolved before proposing another head or training run.

The [Swift model card](https://huggingface.co/ukisai/Swift-Qwen3.8-27b) describes reasoning-marker penalties and a ThinkingCap transfer component.
Its published BF16 base-versus-adapter results are not evidence for an additional gain on the current uncensored IQ4 merge.
The useful next target-training question is how to improve correctness-conditioned efficiency across tasks without inheriting the observed token-length regressions.

## What Jev contributed and what it did not

Jev consistently selected stopping the repeated kernel sweep and failed drafter objective.
It favored retaining SWIFT, analyzing the existing KV/speculation evidence, and researching materially different training mechanisms before running GPU work.
The final review returned no `validate` action requesting an immediate new GPU trial.
Power candidates had the highest final decode-evidence scores, around 1.9 on a zero-to-three rubric, but challenge judgments remained cautious about prioritizing them.
The recommendation to consider power policy is therefore the analyst's synthesis of measured tradeoffs, not a unanimous Jev endorsement.
The already-present sliding-window proposal received a final `stop` action after metadata inspection.

Each challenge pass reversed candidate/evidence order and withheld the preceding screen's answers.
Those passes are sensitivity checks, not independent expert votes or calibrated success probabilities.
Requests and responses are retained, including earlier interpretations corrected by later evidence.
The first packet's mixed-KV status and draft-window assumptions must not be treated as the final findings.

The available results do not support a single numerical frontier across power, target behavior, KV precision, and draft training because their workloads and quality measurements differ.
Higher power offers a measured speed/energy tradeoff within historical matched workloads.
The merged target offers a small-suite token-efficiency tradeoff with substantial case variance.
Claiming a universal top configuration by combining those numbers would invent evidence.

## Reproduction and handoff

All curated packets, source hashes, derived synthetic measurements, and six exact Jev requests/responses are under [the local artifact directory](/home/deforest/Documents/DEV/Marathon/.marathon/diagnostics/jev-frontier-20260919).
The final input is `packet-final.json`; the final advisory table is `final/review.json`.
`followup-analysis.json` records the reconstructed natural-completion pairs and synthetic backend totals.
`kv-comparison.json` records all sixteen probe cases with their per-request server timings.
The initial source manifest is `sources.json`; `additional-sources.json` records subsequent inspected evidence.
Artifacts contain no API key or conversation payload.

The reusable [review runner](/home/deforest/Documents/DEV/Marathon/scripts/evals/jev_evidence_review.py) consumes explicitly curated packets and never scans conversation directories.
It prepares requests by default, sends only with `--send`, pins the Jev version, checks returned question coverage, and binds cached responses to exact request hashes.
Validation included Python compilation, actual API response coverage, an offline replay with an unavailable credential path, exact 51-pair aggregate reconciliation, and KV timing/output-token alignment.
No human intervention was required for this review.
