# DFlare feasibility assessment, September 20, 2026

## Decision

Worth a bounded prototype, but not a full training campaign yet.
No DFlare model was trained or benchmarked here, and no local speed improvement is established.
Production workers, models, tool definitions, caches, and private conversation data were untouched.

## Evidence that changes the recommendation

The paper’s matched five-draft-layer comparison reports average speedup 5.12 versus 4.98, about 2.81 percent relative, and chat 3.22 versus 3.14, about 2.55 percent.
Those results use another target and compare against DFlash, not our tuned DFlash2.
The full eight-billion-target training took 32 GPUs for about 160 hours; that is not a pilot cost estimate.
The published recipe uses seven draft layers, nine feature taps, and larger training data.
These findings justify investigating the mechanism but do not predict a 10–20 percent gain locally.
Source: https://arxiv.org/html/2606.02091v2, Tables 1 and 5, Appendix A.3.

## Released implementation

Inspected Tencent/AngelSlim main at revision `ee8ddb2b43e20800bcfdda1e9ac34ea2aab5de5d`.
`angelslim/compressor/speculative/train/models/draft/qwen_dflare.py` implements per-draft-layer softmax feature mixing and separate target K/V projections.
The code includes a sliding-window attention path, although the example configuration uses full attention.
The example configuration has hidden width 2560, seven layers, and vocabulary 151936, so it is not a configuration for our target.
Source: https://github.com/Tencent/AngelSlim/blob/ee8ddb2b43e20800bcfdda1e9ac34ea2aab5de5d/angelslim/compressor/speculative/train/models/draft/qwen_dflare.py

## Exact local weight accounting

Read the production R32 GGUF tensor metadata without loading it on a GPU.
It contains 1,924,404,480 parameters.
Its shared feature projection has shape 25600 by 5120: 131,072,000 parameters and 73,728,000 stored bytes.
Each layer’s K and V projection separately has 5,242,880 parameters and 2,949,120 stored bytes.

For a five-layer adaptation with otherwise unchanged dimensions, adding independent target K/V projections costs 52,428,800 parameters.
Removing the shared projection yields a net reduction of 78,643,200 parameters before small fusion/norm differences.
At the current tensors’ Q4_K packing, the corresponding net weight saving is approximately 42.19 MiB.
This is tensor accounting, not measured peak VRAM or proof that the model fits.
A residual adaptation that retains the old projection instead adds approximately 28.13 MiB of target K/V weights at the same packing, before other allocations.

Retain five layers, the 2048-token draft sliding window, six proposals, and the unchanged target/context/cache configuration for the first comparison.
Adding layers or increasing proposal width would introduce memory problems already seen in previous experiments.

## Runtime work and memory gates

The local `src/models/dflash.cpp` requires one shared FC projection and uses its output for every layer’s target KV injection.
The small-batch path in `common/speculative.cpp` also encodes features into one shared vector before injection.
Both small-batch and fused-prefill paths need adaptation, along with tensor naming/loading and GGUF export.
DFlash2’s convolution and selector must be preserved or explicitly compared; replacing it with stock DFlare would confound the experiment.
Validate feature-layer indexing, normalization, positions, media handling, rewind, and save/restore before any deployment.

Do not retain raw hidden states for the entire context on the serving GPU.
At 196000 positions and width 5120, five BF16 feature taps alone occupy about 9.35 GiB; nine occupy about 16.82 GiB.
Our current host feature path uses float32, doubling those raw sizes if retained in that format.
Stream feature batches into bounded draft KV storage and discard raw features.
Feature buffers, activation allocations, and graph workspaces still require measurement at occupied long context.

## Training feasibility

The existing local trainer reads concatenated five-tap raw features with width 25600, rather than only the fused output.
That makes the existing synthetic feature format useful for an initial five-tap pilot, subject to verifying target revision, capture alignment, and train/evaluation separation.
Nine-tap training requires new captures.
One million captured tokens occupy approximately 47.68 GiB for five BF16 taps or 85.83 GiB for nine, before tokens, labels, and other artifacts.
A measured capture/training throughput is needed before quoting a trustworthy completion time.

An arbitrary trained dense projection cannot generally be reproduced by a scalar weighted sum of layer features.
Consequently, copying R32 weights and initializing uniform fusion does not preserve R32 behavior.
A native replacement requires meaningful adaptation; a zero-initialized residual branch can preserve the baseline initially but is a DFlare-inspired experiment, not a faithful implementation.
The original R32 adapter remains absent from the searched training artifacts, so reconstructing its trainable checkpoint is an explicit gate.
Loading dequantized GGUF weights is another starting point, but does not recover the original precision.
Existing DPACE and other objective trials mean that merely changing the loss name is not new evidence.

## Smallest useful experiment

1. Verify or reconstruct a trainable R32 reference and reproduce its acceptance behavior on existing held-out synthetic examples.
2. Implement a five-layer, five-tap conditioning experiment with a baseline-preserving control; label native and residual variants distinctly.
3. Measure feature/injection overhead and peak memory before scaling training; keep target verification unchanged.
4. Run one bounded training pilot with separate training and held-out learning curves, including a matched conventional-adaptation control.
5. Continue only if extra accepted tokens compensate for measured complete-round overhead; a poor tiny pilot does not disprove full DFlare.
6. Require repeated actual-runtime comparisons for chat, coding, tools, and occupied long context before deployment, with task correctness and sampler checks.

A reasonable project decision threshold is a repeatable 5 percent everyday decode gain with no quality or memory regression, preferably reaching the user’s 10–20 percent goal.
That threshold is a proposed engineering gate, not a statistical confidence claim or expected result.

## Jev review

Two Jev 1.13.0 requests reviewed only an authored public/aggregate evidence packet.
The first chose further analysis for native replacement, validation for a residual prototype, and stopping the full-recipe campaign.
The reversed-order challenge gave weak support to prioritizing either prototype, so the review was not a strong endorsement.
Its numerical scores are uncalibrated judgments and are not probabilities or measured speedup evidence.
Requests, responses, and the input packet are retained under `.marathon/diagnostics/dflare-feasibility-20260920/` outside tracked documentation.

## Conclusion

Conditional go for a narrow feasibility prototype; no-go for a large training run at present.
The best new opportunity is better per-layer conditioning while preserving the existing DFlash2 strengths.
Neither this inspection nor the published results establish a route to 200 tokens per second in ordinary conversation.
