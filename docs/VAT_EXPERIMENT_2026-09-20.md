# VAT-style DFlash2 experiment, September 20, 2026

Completed: retain the production R32 drafter.
The bounded experiment found a small serving change but no established useful VAT-specific gain over matched continued training.
No candidate is deployed.

## Repository search

The paper's advertised [NAVER VAT repository](https://github.com/naver-ai/VAT) returned HTTP 404.
Exact-title repository searches, code searches for the verification head and first-rejection logic with DFlash, and the first 100 NAVER repositories ordered by update did not locate a released implementation.
This bounded search does not establish that no private or differently named implementation exists.
The [paper](https://arxiv.org/html/2608.30135v1) remains the method specification.
[SpecForge](https://github.com/sgl-project/SpecForge) provides the existing training foundation; it is not being represented as a released implementation of VAT.

## Data and method

The experiment reuses exactly the previously authorized local suffix captures, with 16 training examples, eight validation examples, and eight test examples from a separate conversation.
A new teacher capture uses the pinned production runtime libraries and exact merged IQ4_XS target to record full-vocabulary FP32 logits for all 6,008 supervised tokens.
All 6,008 teacher argmax tokens match the existing captures.
These are teacher-forced predictions conditioned only on earlier tokens, with row zero predicting the first completion token.
No conversation text or probability arrays were sent outside the machine.

The matched control and VAT branches start from the same dequantized served R32 draft.
Both use rank-16 LoRA on the same 36 linear matrices, learning rate 5e-6, 96 updates, eight anchors per update, and full-soft plus hard teacher supervision.
Both retain the DFlash2 selector loss.
The target embedding and output matrices remain frozen.
The supervised-position denominator is fixed across the objectives; the control uses exponential positional decay with gamma seven.

The VAT branch shifts that decay to each block's first rejection and adds a binary cumulative-survival head with coefficient one.
Labels follow the DFlash2 predecessor-dependent greedy candidate path, rather than independent unary predictions.
The auxiliary head trains jointly with draft representations and is omitted from the exported model.
This is a bounded paper-based LoRA adaptation, not a reproduction of the published full training campaign.
It uses greedy verification labels and does not establish results at other sampling temperatures.

The initial VAT branch clipped all gradients together.
Its head gradients dominated the norm, so one controlled follow-up clipped head and adapter groups separately to check whether that suppressed adapter learning.
No broader learning-rate or dataset sweep was conducted.
Sequential-survival boundary cases, first-rejection weighting, and a gradient path from the auxiliary objective into hidden representations were checked.
Training gradients and losses remained finite.
Peak allocated training VRAM was about 9.27 GiB, with the target not resident during training.
That is not the production serving VRAM requirement.

## Offline validation

Checkpoints were selected only on the fixed validation anchors at updates 0, 32, 64, and 96.
The separate test conversation was not used for training or checkpoint selection.

| Branch | Baseline advanced-token proxy | Best proxy | Selected update |
| --- | ---: | ---: | ---: |
| Matched control | 4.55469 | 4.58203 | 96 |
| VAT, joint clipping | 4.55469 | 4.57813 | 32 |
| VAT, separate clipping | 4.55469 | 4.58984 | 64 |

These small changes are offline BF16 proxies, not serving speeds.
The selected separate-clipping checkpoint improves this proxy by approximately 0.77%, but its serving result is the deciding evidence.

## Export and serving controls

Only the 36 adapted linear matrices are merged and requantized into the existing draft tensor types.
All other tensor bytes, shapes, metadata, and quantization types remain unchanged.
The auxiliary classifier is not included in the GGUF.
Each export retains the original draft file size of 1,143,006,752 bytes.

A zero-adapter export was also measured because dequantizing to BF16 and requantizing changes tensor bytes even without training.
Its changed output on one held-out case matches the output change seen with the trained branches, localizing that discrepancy to the export round trip rather than uniquely to VAT.
This is not proof of general quality equivalence or a reason to ignore output changes.

Serving uses an exclusively leased GPU 3 at unchanged 250 W, the pinned production container, the merged IQ4_XS target, Q8/Q8 target KV, Q4/Q4 draft KV, six neural proposals, lookup enabled, and 196000 configured context capacity.
Each variant uses a separate local cache directory and loopback endpoint.
Each held-out prompt has a warmup and a measured greedy completion capped at 256 tokens, with real prefix reuse checked from the response timings.
The test is an isolated runtime replay of reconstructed suffixes, not execution of historical tools or exact full production histories.
The initial baseline and trained-control screens overlapped CPU export work; repeat runs after export completion are needed to assess small timing differences.

## Artifacts

Local artifacts are under `.marathon/diagnostics/vat-experiment-20260920/`, restricted to the user and excluded from Git.
They include teacher capture source and logits, the training adapter and head artifacts, export manifests, objective checks, protocol hashes, serving commands, aggregate metrics, and private responses.
Teacher logits take approximately 5.56 GiB and can be regenerated from the retained capture helper and existing local fixtures.
No production configuration or model file was modified.

## Final serving result

All eight held-out cases completed for the original, zero-adapter export, trained control, VAT joint-clipping, and VAT separate-clipping variants.
The original, trained control, and VAT joint-clipping variants were repeated, giving 64 measured requests with real prefix reuse.
Output hashes and accepted/proposed counts reproduced exactly across each of those three repeated variants.
Seven cases have identical output tokens between the original and exported variants; all eight match between the trained control and VAT.

| Later comparison, after CPU exports finished | Identical-output cases | Mean paired decode change | Accepted-token difference |
| --- | ---: | ---: | ---: |
| VAT versus original | 7 | +1.53% | +7 |
| Trained control versus original | 7 | +1.13% | +6 |
| VAT versus trained control | 8 | +0.39% | +1 |

The first VAT-versus-control pass measured +1.71%, while the later pass measured +0.39%, despite identical output and acceptance counts within each variant's repetitions.
That timing variation makes a sub-percent method-specific improvement an unsupported deployment claim.
These are unweighted means of per-case decode-rate changes, not aggregate workload throughput or confidence intervals.
The eight excerpts come from one held-out conversation and should not be treated as eight independent conversation-level observations.

The separate-clipping variant was only +0.32% over the first original baseline on its seven identical-output cases.
Separating the auxiliary-head gradient clipping therefore did not reveal a useful serving advantage in this screen.
The zero-adapter export gained two accepted tokens across those seven cases and changed the same eighth output, confirming that export numerics are a material control.

Jev reviewed aggregate evidence and recommended retaining R32 and closing this bounded pilot.
Its response is advisory, not a calibrated confidence claim.
There is no demonstrated general quality regression or equivalence: one output changed, and the generation screen is not an independent task-accuracy suite.
Expanded accuracy and near-capacity occupied-context qualification were not run because no convincing VAT-specific gain justified promotion.
The result does not disprove full VAT, larger or more diverse data, or a serving-faithful quantization-aware adaptation.
It does not justify blindly scaling this recipe.

GPU 3 was released to 15 MiB idle usage with its 250 W cap unchanged.
The GPU-control repository remains clean and the production worker was not restarted or reconfigured by the experiment.
