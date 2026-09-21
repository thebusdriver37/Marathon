# Verification-Aware Training feasibility, September 20, 2026

The research audit found a distinct training mechanism worth a controlled experiment, but no released compatible checkpoint or deployable speed gain.
A new local candidate-coverage diagnostic supports investigating selection quality before enlarging the drafter.
Production remains unchanged.

## Mechanism and availability

[Verification-Aware Training](https://arxiv.org/html/2608.30135v1) combines a training-only verification-survival classifier, first-rejection-anchored loss decay, and hard plus soft teacher supervision.
Its classifier must backpropagate into trainable draft representations; fitting only a disposable head on a frozen drafter cannot improve inference.
The published evaluation uses BF16 EAGLE-3 and DFlash with smaller targets on A100 GPUs, not our quantized DFlash2 runtime.
The reported maximum speed improvement elsewhere is 8.7%, not a prediction for Marathon.

The paper's advertised `naver-ai/VAT` repository returned HTTP 404 on September 20.
GitHub repository searches for the paper title and VAT speculative decoding returned no matching implementation in this bounded search.
This establishes that the advertised release was unavailable to this audit, not that no implementation exists anywhere.

Our local trainer already implements DPACE confidence-based weighting, selector supervision, and optional sparse-top-k soft targets.
Previous rejection mining, DPACE variants, a low-weight soft-target branch, selector calibration, and all-layer LoRA failed held-out serving promotion.
Source inspection found no auxiliary cumulative-survival classifier or first-rejection-anchored VAT schedule in those experiments.
Therefore VAT is materially different, but it inherits substantial counterevidence against assuming that another small fine-tune will generalize.

## New local diagnostic

An unchanged, dequantized copy of the exact R32 serving GGUF was evaluated on the existing local feature captures.
The test used 16 training excerpts and eight validation excerpts, with 32 sampled anchors each.
The separate test conversation was excluded.
No target or draft parameters were updated.

| Offline metric | Training, 512 anchors | Validation, 256 anchors |
| --- | ---: | ---: |
| Current greedy-path advanced-token proxy | 4.46484 | 4.62891 |
| Ideal top-16 candidate-choice proxy | 5.91016 | 6.22656 |
| Per-position target candidate coverage | 88.87% | 92.08% |

Both advanced-token proxies include the anchor plus the accepted proposal prefix.
The ideal chooser knows the correct target tokens and selects them whenever they occur in the candidate shortlist.
It is an oracle diagnostic, not a runnable optimization.
Per-position coverage is not sequential acceptance: one early miss discards the remaining block.

This shows that the existing candidate lists contain useful predictions that the current selector does not always choose.
It does not prove that a trainable selector can recover the gap, that VAT will improve it, or that the difference converts directly into a throughput gain.
These are dequantized BF16 predictions on bounded suffix captures, not exact quantized serving predictions or 100K-token histories.
The known offline-versus-serving mismatch still applies.

## Outside implementations checked

[LukasParke/qwen38-27b-5090](https://github.com/LukasParke/qwen38-27b-5090), inspected at `e6c9211f7ceb87a1212d02073b2d3f5107ad9f94`, reports an approximately 11% improvement from changing verification dispatch from MMVQ to MMQ.
It uses different hardware and a 131K/F16-cache configuration.
Our new 75K trace already places the expensive verification work in MMQ kernels.
The summed quantized MMVQ activity is only about 19.5 ms across the cached request, so repeating that dispatch change is not supported as an equivalent opportunity here.

[syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090), inspected at `feaffb676ba0ac6ba5060bd2edbd39cce952829e`, provides real implementation ideas including split-KV verification, sampler changes, draft-vocabulary selection, and lookup drafting.
Its documented DFlash2 configurations involve a different weight format and runtime, with context and memory tradeoffs including 56K/64K configurations.
Those are not drop-in replacements for the fixed merged IQ4_XS target and 196K capacity.
Lookup drafting is already deployed locally.
MTP-specific vocabulary improvements must not be advertised as measured DFlash2 improvements.
The source remains useful for targeted kernel ideas, rather than a justification for replacing the whole runtime now.

## Recommended controlled experiment

1. Capture soft teacher predictions from the exact merged target alongside the hard labels and target features.
   The recent feature captures do not contain these distributions.
   The older sparse-top-k approximation is not the same as the paper's full soft-label objective.
2. Establish matched initialization, trainable scope, data, and hard-plus-soft supervision for a control and VAT candidate.
   Compare the new weighting and auxiliary classifier, rather than confounding them with a changed dataset or distillation strength.
3. Derive acceptance labels from the DFlash2 predecessor-dependent candidate path.
   Independent unary argmax labels would describe a different drafter.
4. Train draft representations, with the target frozen, and remove the auxiliary head for export.
5. Export onto the same serving format and measure actual quantized serving before accepting an offline gain.
   Use held-out conversation boundaries, matched outputs where possible, task accuracy checks, and later long-context qualification for a candidate that first shows a gain.

Jev reviewed only aggregate evidence and selected teacher-data preparation and a matched objective design before training.
It rejected interpreting the oracle gap as a demonstrated speedup or proof that a bigger drafter is necessary.
Its advice is a judgment, not a calibrated probability of success.

A hard-label-only, frozen-representation approximation would not provide a fair go/no-go test of the published method and was not substituted for it.
No training claim or checkpoint promotion is made by this feasibility audit.

## Artifacts and final state

Local-only artifacts are under `.marathon/diagnostics/vat-feasibility-20260920/`.
`oracle.py` reproduces the diagnostic and `oracle.json` contains its complete aggregate ratios.
The Jev request and response contain only curated aggregate evidence.
The diagnostic reused already authorized local captures without inspecting additional conversations.
GPU 3 was leased for the diagnostic and released afterward.
No weights, production settings, context capacity, or power limits changed.
