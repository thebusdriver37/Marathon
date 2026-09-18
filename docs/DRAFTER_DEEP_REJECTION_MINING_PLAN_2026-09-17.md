# DFlash2 deep-turn rejection mining plan

## Decision

Run one bounded training round that combines exact serving rejection spans with varied deep-turn and broad anchor data.

Do not scale the earlier corpus distribution.

The earlier checkpoint already learned its repeated opening patterns, improving first-turn acceptance by 9.77 percentage points while changing deep-turn acceptance by negative 0.21 percentage points.

Deep turns dominate measured request time, so the next dataset must target recurring late-turn errors directly.

## Jev recommendation

Jev assigned 95 percent probability to exact low-acceptance span mining as the best selection granularity.

Jev assigned 76 percent probability that implementing the exact trace was justified before another training round.

Jev narrowly preferred a supervised-position mixture of 50 percent hard spans, 30 percent random deep turns, and 20 percent broad anchors.

Jev assigned 92 percent probability to the eligibility rule requiring assistant turn 8 or later, at least 3,000 prompt tokens, at least 64 completion tokens, and composite ranking with semantic and archetype controls.

Jev preferred a generation allocation of 50 percent iterative patch chains, 20 percent failure recovery, 20 percent cross-tool synthesis, and 10 percent late revision or verification.

Jev estimated the best bounded design most likely falls in the 40 to 60 percent probability range for achieving at least a one percentage point deep-turn acceptance gain.

Jev also assessed meaningful overfitting risk, which is why the random deep and broad anchor groups remain in the mixture.

## Data flow

1. The Spark model authors varied, grounded software projects with independent executable checks.

2. Two Marathon workers solve each project while a third worker writes adaptive follow-ups that depend on actual edits, failures, tool results, identifiers, and measurements.

3. The current scaled drafter replays eligible late-turn request boundaries with an opt-in serving trace.

4. Each trace records the output offset, proposed token count, and accepted token count for every speculative verification block.

5. Jev classifies only generated candidate turns for semantic value, recurring difficulty, context dependency, and reproducibility.

6. The selector captures features for hard correction-centered spans, randomly sampled deep responses, and a small broad anchor set.

7. One continued-training checkpoint is compared with the current scaled checkpoint on untouched generated evaluation conversations and the fixed broad suite.

## Trace contract

The trace is disabled unless `LLAMA_DRAFT_TRACE=1` is set.

The trace-on and trace-off validation responses had identical output bytes.

The trace event sums exactly matched the existing server draft and accepted counters.

Replay after a recurrent-state checkpoint restore is suppressed so one speculative decision cannot be counted twice.

The target correction token for a partially accepted block is at `output_offset + accepted`.

Hard spans include the preceding anchor and a small neighborhood around that correction token.

Rejected proposal tokens are not included as if they appeared in the target response.

The first end-to-end smoke selected 238 hard tokens from a 288-token generated response after correction-centered tightening.

The earlier sparse capture smoke stored exact training features with zero rank-one mismatches and zero target tokens outside the recorded top-k.

## Selection policy

Hard eligibility requires assistant turn 8 or later, at least 3,000 prompt tokens, at least 64 completion tokens, a valid exact trace, and at least four rejected proposals in one speculative block.

Hard ranking combines rejected proposal count, rejection density, zero-accept blocks, Jev hard-mining value, Jev reproducibility, and a response-length penalty.

Hard selection enforces generation-archetype quotas so one frequent pattern cannot consume the corpus.

When Jev classification is enabled, an unclassified turn cannot enter the hard set.

Random deep examples are sampled from every eligible late turn without using acceptance or Jev hardness.

The random deep set is therefore an actual distribution control rather than a softer hard-example set.

Validation responses are selected randomly from the validation split without using acceptance.

Evaluation conversations remain untouched until the final serving comparison.

## Bounded round

The initial target is approximately 12,000 hard supervised tokens, 7,200 random deep supervised tokens, and 4,800 broad anchor tokens.

The exact number of training samples will be calculated after tokenization because the trainer splits sparse spans into chunks of at most 256 tokens.

Training continues from the step-2,767 scaled adapter at the conservative learning rate used by the successful prior round.

The offline evaluator is used only for checkpoint selection and regression diagnosis.

The serving benchmark remains the authority.

## Promotion gate

The new checkpoint must improve untouched deep-turn acceptance by at least one percentage point against the scaled checkpoint.

The confidence interval, paired decode throughput, broad-suite acceptance, output consistency, and task completion checks must not show a material regression.

If the new checkpoint misses that gate, stop this training line before generating or capturing a larger corpus.

If it clears the gate, use the new trace taxonomy to decide which recurring rejection families justify the next incremental batch.

## Outcome

The bounded round captured 12,416 hard supervised tokens, 7,286 random deep supervised tokens, and 4,926 broad anchor tokens.

The held-out set contained 6,994 supervised tokens from untouched generated validation conversations.

The initial one-epoch branch produced 282 optimizer steps from the step-2,767 scaled adapter.

Its offline accepted-block diagnostic rose from 5.4730 to 5.4828, but its step-94 and step-282 serving screens were null on byte-identical outputs.

Because one epoch was insufficient to characterize the optimization curve, a bounded recipe sweep then trained three branches for five corpus passes each and evaluated every 282 steps.

| Recipe | Offline accepted length from epoch 0 through epoch 5 | Offline selection | Serving result |
|---|---|---|---|
| DPACE 0.5, learning rate 5e-6 | 5.4730, 5.4751, 5.4778, 5.4728, 5.4760, 5.4723 | Epoch 2 | Failed promotion |
| DPACE 0.5, learning rate 1e-5 | 5.4787, 5.4771, 5.4760, 5.4741, 5.4726, 5.4737 | Initialization | Eliminated offline |
| DPACE 1.0 plus total variation, learning rate 1e-5 | 5.4787, 5.4676, 5.4683, 5.4823, 5.4766, 5.4764 | Epoch 3 | Failed promotion |

The standard epoch-2 branch initially showed positive acceptance in a concurrent 20-boundary replay, but that result did not reproduce in isolated testing.

The sequential same-GPU validation and untouched-eval replays covered 36 boundaries across nine conversation groups.

On their 34 byte-identical boundaries, paired acceptance changed by negative 0.07 percentage points with CI95 negative 0.51 to positive 0.42, paired decode throughput changed by negative 0.36 percent with CI95 negative 1.04 to positive 0.39, and paired wall time increased 0.79 percent with CI95 positive 0.46 to positive 1.15.

The total-variation branch changed paired acceptance by negative 0.09 percentage points on 17 byte-identical boundaries, with no reliable throughput or wall-time gain.

Jev assigned 60 percent probability that the conflicting standard epoch-2 measurements came from serving-sample uncertainty and selected a larger evaluation as the best next action with 56 percent probability.

The untouched eval replay resolved that uncertainty against the epoch-2 candidate.

After seeing the expanded result, Jev assigned 97 percent of its next-action probability to exactly one low-weight soft-target branch, 3 percent to stopping immediately, and zero to a broader recipe sweep.

Jev estimated only an 18 percent chance that the soft-target branch would meet both promotion thresholds, but rated the single inexpensive test as justified with probability 0.81.

The final soft-target branch used alpha 0.1 with standard DPACE at learning rate 5e-6 for five corpus passes.

Its offline accepted-length curve was 5.4730, 5.4728, 5.4701, 5.4741, 5.4698, and 5.4762, making epoch 5 the offline selection.

On 16 untouched byte-identical serving boundaries, the soft-target checkpoint changed paired acceptance by positive 0.40 percentage points with CI95 negative 0.59 to positive 1.38, decode throughput by positive 0.22 percent with CI95 negative 0.81 to positive 1.26, and wall time by positive 0.20 percent with CI95 negative 0.66 to positive 0.80.

The soft-target checkpoint therefore missed both the positive 1 percentage point acceptance threshold and the positive 1 percent decode-throughput threshold.

All serving replays required and observed real server-side prefix reuse.

The serving analyzer reports byte-identical request boundaries separately so output divergence cannot create a false acceptance or throughput win.

The step-2,767 scaled adapter remains the selected drafter.

This incremental deep-turn corpus line stops here because ordinary extra epochs, a larger learning rate, total-variation regularization, and soft-target distillation all failed the serving promotion gate.

Jev rated classification of repeatedly reproduced rejection spans by error mechanism as its highest-value role in any future, materially different data round.

## Final serving-parity viability gate

One final mechanism test examined whether a serving-faithful evaluator could support a materially different training objective.

Jev assigned all of its action probability to a two-gate test: first require evaluator parity, then deliberately overfit a tiny set of exact rejection blocks only if parity passed.

Jev rated the test as justified with probability 0.76, rated its information value as high with probability 0.84, and assigned probability 0.87 that a negative result would support stopping this DFlash2 training line.

Source inspection corrected an earlier explanation of the mismatch.

DFlash2 evaluates the anchor and six mask positions in one draft-model pass, then follows a learned transition lattice through the candidates.

It does not rerun the draft network autoregressively after every proposal.

The existing Python evaluator already follows the selected predecessor through that transition lattice, so adding another nominally self-conditioned selector would duplicate existing behavior.

The remaining useful parity test was to evaluate the exact response offsets visited by real serving instead of sampling arbitrary clean anchors.

The new exact-anchor evaluator at `.marathon/drafter-training/corpus-pilot/evaluate_dflash_trace_parity.py` imports the serving GGUF, replays captured target features at every recorded draft-event offset, follows the DFlash2 greedy transition path, and compares its accepted length with the server trace.

The parity thresholds were declared before the result: absolute aggregate acceptance error at most 1.0 percentage point, at least 90 percent exact event agreement, and accepted-token mean absolute error at most 0.25 per event.

Across five generated conversations and 998 six-token draft events, the server trace measured 48.8143 percent acceptance while the exact-anchor Python evaluator measured 47.5785 percent.

The aggregate error was negative 1.2358 percentage points, exact event agreement was 85.2705 percent, and accepted-token mean absolute error was 0.3627 per event.

Gate 1 therefore failed all three parity requirements.

The evaluator error is larger than the one-percentage-point improvement the experiment was intended to detect.

The mismatch can arise from the different numerical execution paths used by dequantized BF16 PyTorch evaluation and quantized llama.cpp serving, and it prevents the differentiable evaluator from acting as a trustworthy ruler at the required effect size.

Per the predeclared contract, the exact-block overfit training gate was not run.

### Token-path localization

A final bounded diagnostic extended the opt-in serving trace with the six draft token IDs selected for each verification block.

The diagnostic replayed one generated fixture that reproduced its recorded output byte-for-byte and compared the server proposals with Python proposals at the same captured feature rows.

Thirty six-token events fit completely inside the captured feature region.

The first proposed token matched in 29 of 30 events, or 96.67 percent, which rules out a broad prompt, feature-row, or anchor-position alignment failure.

Only 18 of 30 complete proposal paths matched, or 60 percent.

Per-position proposal agreement was 96.67, 83.33, 80.00, 66.67, 63.33, and 66.67 percent from positions one through six.

The first divergence occurred at proposal position three of the first verification block.

The server and Python implementations use the same top-k candidate construction, anchor predecessor, transition equation, and greedy predecessor walk.

The remaining mismatch is therefore localized to their numerical forward paths: llama.cpp executes the Q4 GGUF with serving kernels while Python dequantizes the GGUF into BF16 tensors and executes PyTorch operators.

That numerical difference rarely changes the first token but compounds through later transition choices.

On this small fixture the server accepted 52.78 percent while Python predicted 57.22 percent, exact accepted-length agreement was 86.67 percent, and mean absolute error was 0.40 token per event.

This result does not identify a small selector or alignment bug that can be tuned away.

Exact parity would require evaluating through the serving numerical path or developing quantization-aware differentiable training, both of which are materially different systems rather than minor adjustments to the current simulator.

The raw comparison is stored at `.marathon/drafter-training/trace-token-parity-selected-r32-stable-case.json`.

### Server-loop calibration feasibility

The numerical mismatch did not make training through the serving behavior impractical.

A margin diagnostic examined the server-selected token at the first divergent selector decision in each event.

Eight of 12 divergent server tokens remained in the Python top-k: seven ranked second and one ranked third.

The median Python score gap from the selected token was 0.8125 and the maximum was 2.6094.

This showed that small selector updates could plausibly cross the quantized serving decision boundaries.

`train.py` now accepts exact serving anchors through `--forced-anchor-file` and can train the selector tensors at those recorded response offsets.

The bounded loop is capture exact serving anchors, train from the dequantized served GGUF, requantize the changed selector tensors onto that GGUF, and measure the result in the real server.

Training takes less than one minute, export and requantization take about 40 seconds, and a small serving gate takes a few minutes.

The first test intentionally overfit 30 exact serving events from one generated fixture in 36 optimizer steps at learning rate 1e-4.

On that same fixture, real server acceptance increased from 50.52 to 69.33 percent, a gain of 18.81 percentage points.

Decode throughput increased from 73.40 to 89.72 tokens per second, or 22.2 percent, while the generated output remained byte-identical.

This establishes that the quantized serving behavior is controllable with a very small calibration loop.

It also disproves the assumption that a serving-in-the-loop experiment would require a prohibitively large new system.

The one-fixture checkpoint did not generalize.

Across five held-out generated conversations in an ABBA server test, acceptance decreased from 83.85 to 82.21 percent and decode throughput decreased 1.32 percent.

A diverse calibration set was then constructed from 40 rejection windows across 13 generated conversations and 4,103 exact serving anchors.

The first selector-only recipe used two epochs and learning rate 1e-5.

After requantization it produced exactly the same 5,794 accepted tokens from 6,910 proposals as the baseline, with a decode difference of negative 0.16 percent.

The update was too small to change quantized serving decisions on the held-out set.

The stronger recipe used the same anchors and two epochs at learning rate 5e-5.

Across all five held-out conversations, acceptance decreased from 83.85 to 83.25 percent and decode throughput decreased 0.51 percent.

One tool-error-recovery case changed its target output reproducibly and accounted for the aggregate regression.

On the four cases whose outputs were byte-identical across both drafters and both ABBA repetitions, acceptance increased from 86.23 to 86.48 percent, decode throughput increased 0.30 percent, and wall time decreased 0.34 percent.

Three of those four cases had exactly unchanged acceptance counts, while one debugging case gained 0.58 percentage points.

The complete stronger-recipe comparison is stored at `.marathon/drafter-training/server-loop-selector-diverse-lr5e5-heldout-abba-20260918/summary.json`.

The final conclusion separates controllability from generalization.

Server-loop calibration is technically viable and fast enough to iterate.

The current selector-only objective has not demonstrated a reliable held-out improvement: its only output-stable gain is about 0.3 percent on four cases, while a fifth case regressed and changed generation.

No checkpoint from this calibration line should replace the selected step-2,767 drafter.

A larger server-loop training round was not justified by that evidence because the observed stable gain was below the experiment's practical effect threshold and was supported by only one improving held-out case.

### Conversation-level scaling study

A final small scaling study tested whether the earlier failure came from too few independent conversations or from restricting updates to the selector tensors.

The selector training sets were cumulative and contained 3, 7, and 13 independent generated conversations, corresponding to 194, 885, and 4,103 exact serving anchors.

Each selector model used 12 shuffled epochs, learning rate 1e-4, D-PACE alpha 0.5, and the selected step-2,767 drafter as its starting point.

A fourth model trained rank-32 LoRA adapters over all 36 drafter weight tensors on the 13-conversation set for 12 epochs at learning rate 5e-6.

All four models were requantized onto the same serving GGUF and evaluated with warm-cache ABBA ordering against the selected drafter.

The locked held-out set contained five different generated conversations, four turn boundaries per conversation, and 20 paired request boundaries in total.

Every continuation request demonstrated real server-side cache reuse.

The primary comparison used only boundaries whose output bytes were identical across all four ABBA stages.

| Training set | Trainable scope | Stable boundaries | Acceptance change | Decode change | Wall-time change | Acceptance CI95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 3 conversations, 194 anchors | selector | 19 | -0.4383 pp | -0.2414% | +0.1101% | [-1.6328, +0.8531] pp |
| 7 conversations, 885 anchors | selector | 19 | -0.4904 pp | -1.2680% | +0.6171% | [-1.5284, +0.5493] pp |
| 13 conversations, 4,103 anchors | selector | 17 | -1.7090 pp | -2.1262% | +1.9395% | [-3.2449, +0.2736] pp |
| 13 conversations, 4,103 anchors | rank-32 LoRA, 36 tensors | 20 | +0.0700 pp | -0.1422% | +0.1442% | [-0.2530, +0.4797] pp |

Positive wall-time changes mean the candidate was slower.

The selector scaling curve was negative rather than merely inconclusive.

Increasing independent conversations made held-out acceptance and speed progressively worse, even though the offline representative score rose sharply for every selector checkpoint.

The 13-conversation selector also changed three held-out output paths, while the full-LoRA model preserved all 20 outputs.

Broader LoRA capacity removed the large selector regression but produced no measurable gain because all three performance intervals crossed zero.

The one-fixture 18.81 percentage-point gain therefore remains evidence of local controllability, not evidence that exact-anchor calibration generalizes.

The result is consistent with the objective learning fixture-specific quantized decision boundaries rather than a reusable serving policy.

More anchors, more epochs, or broader parameter scope do not rescue this exact-anchor objective under the tested conditions.

No checkpoint from the scaling study should replace the selected step-2,767 drafter.

The combined machine-readable result is stored at `.marathon/drafter-training/corpus-pilot/runs/deep-mining-r40-v2/server-loop-scaling-20260918/results.json`.

## Privacy boundary

Only generated experimental Marathon conversations and their outputs enter this pipeline.

Private Marathon usage histories are excluded from generation, Jev classification, feature capture, training, and evaluation.
