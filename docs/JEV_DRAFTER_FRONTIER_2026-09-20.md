# Jev drafter frontier review, 2026-09-20

## Decision

Jev chose runtime cost and layout analysis as the best immediate action, and JonasLoos/Qwen3.8-27B-DFlash2-b32 as the best released alternative to investigate.
The latter is a longer-block trained checkpoint, not a larger parameter-count model.
Neither has demonstrated an additional gain over our deployed R32 drafter.

Three Jev 1.13.0 requests produced 82 structured judgments: 44 screening answers, 33 reversed-order challenge answers without the screen answers, and five explicit priority choices.
These are advisory judgments from the same model over our curated evidence, not independent experiments or calibrated success probabilities.
Only public research descriptions and aggregate synthetic measurements were submitted.
No production conversation text was read or transmitted for this review.

## Concrete candidate and gates

Keep the merged IQ4_XS target, target Q8/Q8 KV, at least 196000 context, single 24GiB card, and power no higher than 275W.
Investigate the JonasLoos b32 checkpoint with its trained five-layer architecture, 2048 sliding window, and block-size-32 metadata intact.
Q4_K_M is the proposed local draft export format for a fair initial comparison, but conversion and runtime support have not yet been validated.
Choose runtime proposal width only after confirming the engine can execute the trained layout correctly and fit all attention and verification scratch allocations at 196000 context.
Do not infer that setting a flag to 32 provides this capability.
The previous seven-proposal CUDA dynamic shared-memory failure is an unresolved gate.

First inspect and instrument common_speculative_process, feature gathering/injection, synchronization, and the failed attention path.
The measured draft-generation fraction was only about 12.5 percent of decode time; the remaining time is not all known target verification cost.
Compare accepted tokens per complete round divided by complete round latency, not acceptance alone.

If compatibility and memory gates pass, compare the candidate directly against deployed R32 on held-out generated chat, coding, and media tasks with matched target, sampler, power, context, and cache conditions.
Include long-context and near-capacity coverage before deployment.
Measure successful-task wall time, whole-response decode throughput, round latency, accepted tokens, and output correctness.
Investigate deterministic output differences and verify sampling correctness rather than assuming any implementation is lossless.
Keep private production histories outside training and evaluation.
No candidate was downloaded, trained, benchmarked, or deployed during this research review.

## Primary-source findings

[JonasLoos b32](https://huggingface.co/JonasLoos/Qwen3.8-27B-DFlash2-b32) is trained for longer blocks with unchanged five-layer dimensions.
Its creator reports stock-to-candidate accepted tokens per pass improving from 6.0 to 7.0 on math, 3.9 to 4.2 on chat, and 11.9 to 15.5 on code using a 32-token draft tree on an M5 Mac.
Those are acceptance measurements, not local throughput gains, and compare against stock rather than our R32 model.
Training used thinking-mode outputs and features from an MLX four-bit target, rather than our exact merged target.
Only the creator's MLX engine was tested.

[DFlare](https://arxiv.org/html/2606.02091v2) addresses depth saturation with distinct learned target-feature mixtures for individual draft layers.
Its reported gains over DFlash are approximately 5 to 11 percent on other target architectures and hardware.
Official code and [an 8B-target checkpoint](https://huggingface.co/AngelSlim/Qwen3-8b-dflare) exist, but no ready Qwen3.8-27B checkpoint was found in this search.
Jev selected it tentatively when forced to choose a future capacity-training direction, while the unconstrained screening deferred it.
It is a research direction, not a ready local configuration.

[DaoCloud Exp](https://huggingface.co/DaoCloud/Qwen3.8-27B-DFlash2-Exp) changes anchor sampling and requires a layout and feature-index audit.
[Apathy v3](https://huggingface.co/onewhosighs/Apathy-Qwen3.8-27B-DFlash-drafter-v3) supplies a six-layer alternative, but its GB10, runtime, cache, and narrow capped benchmark do not establish a local advantage.
[DSpark Agentic](https://huggingface.co/tiyuvta/Qwen3.8-27B-DSpark-Agentic) is smaller but uses full attention, leaving long-context draft KV memory and runtime compatibility unresolved.
[The B70-tuned DSpark](https://huggingface.co/rwmacy/qwen3.8-27b-dflash-drafter-fp8-b70) documents an output-position alignment fix in another runtime; this motivates checking alignment, not assuming our implementation has that bug.
[The longctx conversion](https://huggingface.co/akumaburn/Qwen3.8-27B-DFlash2-W4A16-longctx) changes only context metadata and explicitly offers no improvement below 262K.
[NewHorizon oQ4](https://huggingface.co/NewHorizonGroup/Qwen3.8-27B-DFlash2-oQ4) is a stock MLX quantization, not newly trained draft capacity.

Search covered indexed web sources and Hugging Face name searches for DFlash, DSpark, and DFlare, including small creators.
It was not an exhaustive assessment of every returned repository.

## Local counterevidence included

The current draft already has 1,924,404,480 parameters and its R32 adaptation previously improved matched synthetic serving throughput by 27.15 percent over stock.
Six layers with otherwise identical dimensions would contain approximately 2.257 billion parameters and require training, extra bandwidth, and memory headroom.
The original DFlash depth ablation found five layers faster overall than eight despite higher acceptance from eight layers.
Our later exact-anchor selector and broader LoRA experiments did not produce a reliable held-out gain.
Smaller proposal widths and tested confidence thresholds were slower than the current six-proposal configuration.
Lookup acceleration and the media regression fix are already deployed and cannot be counted as new benefits.

## Evidence artifacts

The curated packet, exact requests, raw Jev responses, and per-candidate review are under `.marathon/diagnostics/jev-drafter-frontier-20260920/`.
The initial screen favored validating b32, but the reversed-order challenge gave runtime analysis stronger support.
The explicit priority request selected runtime analysis first, b32 as the released-model candidate, and no proven additional gain.
The tentative DFlare architecture choice should not override its weak local evidence.

## Follow-up investigation

The authorized local candidate tests are recorded in [DRAFTER_CANDIDATE_TESTS_2026-09-20.md](DRAFTER_CANDIDATE_TESTS_2026-09-20.md).
The research-only recommendations above are superseded by those measurements where a candidate was actually tested.
