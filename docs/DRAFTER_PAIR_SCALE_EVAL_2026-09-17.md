# DFlash2 pair-scale checkpoint evaluation

## Verdict

The scaled checkpoint is a credible research improvement, but it should not replace the deployed Marathon R32 checkpoint yet.

It improved acceptance and decode speed on sealed generated requests and on the fixed broad workload suite.

The improvement was concentrated on first turns, while deep multi-turn requests were flat.

The next bounded experiment should mine difficult deep-turn serving rejection spans and retain a broad semantic mixture.

Collecting ten times more data from the same generator distribution is not supported by these results.

## Training integrity

The frozen corpus contained 110 training conversations, 21 validation conversations, and 25 sealed evaluation conversations.

Feature capture produced 464,472 supervised positions and 763,009 replay positions.

Capture integrity passed, no replay token fell outside the recorded target top-k, and the sealed evaluation split was not captured for training.

Seven one-token terminal training chunks could not define an anchor plus next-token pair and were excluded by a structural admission check.

All 523 validation chunks remained eligible.

The run trained the original Marathon R32 adapter for 2,767 steps at a learning rate of 5e-6.

The step-2,767 adapter was exported onto the stock Q4_K_M serving GGUF.

The export changed exactly the expected 36 LoRA target tensors and verified that every other tensor byte remained identical.

The exported candidate SHA-256 is `ee36e07d1124cceb629111cf7e55216aab35a8453545b008e36ec8d0429bc5d3`.

## Validation curve

The offline evaluator is a training diagnostic and was used only to choose the checkpoint.

| Step | Accepted proposals per sampled anchor | Selector accuracy proxy |
| ---: | ---: | ---: |
| 0 | 4.7978 | 84.09% |
| 700 | 4.8523 | 84.97% |
| 1,400 | 4.8621 | 85.13% |
| 2,100 | 4.8675 | 85.23% |
| 2,767 | 4.8730 | 85.31% |

At step 2,767, 411 validation chunks improved, 72 regressed, and 40 tied relative to the initial R32 adapter.

## Sealed serving ABBA

The primary serving test sampled the first and deepest assistant boundary from each of 25 sealed generated conversations.

The four-stage ABBA produced 200 requests, and every one of the 100 deep continuation requests had measured server-side prefix reuse.

| Measure | Original R32 | Scaled R32 | Change |
| --- | ---: | ---: | ---: |
| Aggregate acceptance | 64.1370% | 65.4575% | +1.3205 pp |
| Accepted proposals per verification | 3.8191 | 3.8962 | +0.0771 |
| Aggregate decode throughput | 102.1599 tok/s | 102.5449 tok/s | +0.38% |
| Aggregate wall time | 465.7385 s | 466.0281 s | +0.06% |

The paired per-boundary acceptance change averaged +5.8086 percentage points with a conversation-group bootstrap CI95 of +3.8082 to +7.8226.

The paired per-boundary decode-throughput change averaged +2.9833% with CI95 +0.7583% to +5.3679%.

The paired wall-time change averaged -0.7661% with CI95 -1.5986% to +0.0766%, so the wall-time result did not clear zero.

## Turn-depth result

The first-turn subset improved acceptance from 70.2196% to 79.9858% and decode throughput from 110.343 to 115.409 tok/s.

The deep-turn subset changed acceptance from 62.8366% to 62.6233% and decode throughput from 100.417 to 99.913 tok/s.

Deep turns accounted for about 416 of 466 wall seconds per variant, which explains why the large first-turn acceptance gain produced almost no aggregate wall-time gain.

The corpus audit had already found that 123 of 131 train and validation conversations began as complete-or-implement-TODO tasks, 125 were ordinary, and 106 were primarily code changes.

The measured turn-depth split is therefore consistent with specialization to the repeated first-turn distribution.

## Broad fixed workloads

Against the original Marathon R32 checkpoint, the scaled checkpoint improved broad-suite acceptance from 50.5094% to 51.7218%, decode throughput by 1.8102%, and wall time by 1.5092%.

All eight measured broad-suite outputs were byte-identical across the original-R32 ABBA.

Against the stock drafter, the scaled checkpoint improved the same acceptance measure from 50.5094% to 51.7218%, decode throughput by 1.3512%, and wall time by 1.2011%.

Seven of eight outputs were byte-identical in the stock comparison.

The scaled checkpoint improved analysis, code boilerplate, novel prose, SQL, and tool-call acceptance relative to original R32, while recovery regressed and the already-strong structured and verbatim regimes were unchanged.

## Output consistency

Forty-five of fifty sealed request boundaries produced identical output hashes across all four stages.

For each of the five divergent boundaries, both original-R32 repetitions matched one another and both scaled-R32 repetitions matched one another.

The five differences were small wording changes in generated task summaries.

The identical-output subset still improved acceptance from 66.3380% to 68.4083%, so changed output does not explain the serving signal.

A structured Jev comparison rated each divergent pair as semantically equivalent with probabilities from 0.83 to 0.96 and assigned 0.20 probability to any material candidate regression.

This check is supportive, but a production promotion should still require a matched task-quality gate because the drafter reproducibly changed target output on some greedy requests.

## Jev interpretation

Jev selected short-turn distribution specialization as the dominant explanation.

Jev assigned 88% of its next-intervention probability to deep-turn hard-example mining, 12% to semantic rebalancing alone, and zero to collecting more of the same distribution or stopping immediately.

It estimated a 22% chance that scaling the same corpus distribution by ten times would produce at least a one-point deep-turn acceptance gain.

It estimated a 52% chance for generated-only deep-turn serving rejection mining and a 42% chance for semantic rebalancing alone.

It gave one more bounded training round a 59% probability of being justified.

These probabilities are advisory classifications of the measured evidence.

The server counters and matched quality gates remain the promotion authority.

## Recommended next experiment

Keep the current checkpoint as a research candidate and do not deploy it yet.

Generate or replay privacy-safe multi-turn conversations, but select training spans from deep turns where serving counters show early draft rejection.

Retain the broad base mixture so hard-example mining does not erase strong structured, tool-call, and ordinary code behavior.

Increase unusual analysis, SQL, recovery, and long-form tasks, and reduce repeated complete-or-implement-TODO openings.

Train one bounded round from the step-2,767 checkpoint and gate it on deep-turn acceptance, the fixed broad suite, output consistency, and matched task completion.

Stop this training line if that round does not improve deep-turn acceptance by at least one percentage point without a task-quality regression.
