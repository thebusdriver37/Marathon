# Drafter research, 2026-09-14

This search found additional candidates worth investigating, but no demonstrated 10% improvement over Marathon's deployed drafter.
The strongest immediate experiment is activation-calibrated draft quantization.
The strongest training direction is adapting the existing DFlash2 architecture to the exact deployed target and representative agent sessions.
These are research priorities, not measured Marathon speedups.

## Scope and baseline

The source-of-truth GPU configuration still uses the JonathanColetti uncensored IQ4_XS target, official DFlash2 Q4_K_M draft, six proposals, Q8 target KV, Q4 draft KV, and 196,000-token allocation on one RTX 3090 per worker.
The runtime image is now `sha256:765a84864664d953cc274adb0fdb161b793269857e4602d228460a790919cea8`, so historical speed measurements should not substitute for a fresh control.
No workers, settings, model weights, or production code were changed in this research.
Only metadata, model cards, configurations, and source references were downloaded; no new inference or training ran.

The Hugging Face API returned 75 repositories for `Qwen3.8-27B-DFlash` and 16 for `Qwen3.8-27B-DSpark`.
These are name-search results, including mirrors, quantizations, target-model bundles, and unrelated deployment packages, not 91 independently trained drafters.
Additional web and GitHub searches covered exact-target matches, consumer-GPU results, training support, and runtime compatibility.
This is a broad search, not an exhaustive enumeration of every differently named or private checkpoint.

The [September 11 investigation](DRAFT_ACCEPTANCE_EVALUATION_2026-09-11.md) already tested window sizes, confidence cutoffs, Q8 draft weights and KV, RadixArk DSpark, updated official metadata, and taniii1234's uncensored draft.
It found no candidate worth replacing production with.

## What actually changed

The official GGUF repository still resolves to `51962825493a48b846b40126d35c799ac4093ad0`, and Anbeeld's DSpark GGUF still resolves to `77843de41a67edb1c6294d641ea38fa78f9c8124`.
Both are exactly the repository revisions recorded in the previous investigation.
The taniii1234 checkpoint is also unchanged at `bd03640a3e49a6eff782e82efafc1ab78e01fedf`.

Hugging Face's published LFS SHA-256 values for Inco's BF16 DFlash2, tiyuvta's memra package, and wyattearp's package are identical: `67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c`.
This comparison uses registry metadata, not newly downloaded multi-gigabyte files.
The memra card also explicitly describes unchanged weights and attributes its gains to serving changes.
Its September 12 commit changes documentation.
[Official weights](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2), [memra package](https://huggingface.co/tiyuvta/Qwen3.8-27B-DFlash2-memra), [wyattearp package](https://huggingface.co/wyattearp/Qwen3.8-27B-DFlash2).

The Agentic checkpoint's actual weight upload was August 20; its September 12 update is also documentation-only.
The interesting candidates below are mostly newly identified in this review, not releases since our last test.

## Candidate shortlist

### 1. Activation-calibrated Q3_K_M: first practical comparison

[andrew-paul's Q3_K_M](https://huggingface.co/andrew-paul/Qwen3.8-27B-DFlash2-Q3_K_M-GGUF) is 916,703,008 bytes, about 216 MiB smaller than the current official Q4 file.
The publisher reports 65.55% acceptance versus 65.56% for stock Q4 across 20 fixed prompts on an RX 7800 XT/Vulkan setup, at greedy sampling and 16K context.
No transferable speed result is established.
The repository includes a calibration patch because the standard collector misses small draft batches and top-level draft tensors.
This suggests an intermediate experiment before training: collect activations from our exact target/drafter pair and compare calibrated Q4 and Q3 against stock.
That proposal remains untested.
Inspect rotary metadata before multimodal use; another publisher reports missing vision-related metadata in this release.

### 2. Mixed Q2/Q3 draft: memory experiment

[HermiHg's Q2_K_S-MIX](https://huggingface.co/HermiHg/Qwen3.8-27B-DFlash2-Q2_K_S-MIX-GGUF) is 561,241,824 bytes, approximately 555 MiB below stock Q4.
Its author reports throughput between roughly 96% and 101% of stock across proposal widths three to five, using one conversational prompt, five repeats, and an unspecified 24 GB NVIDIA GPU.
This is a memory-saving candidate, not evidence of faster execution at our six-token setting.
Its per-tensor precision allocation is relevant to the user's layer-level optimization idea: compress large feed-forward tensors more aggressively while protecting sensitive components.
Whether freed memory enables a useful second optimization requires a separate test.
Seven-token drafting was already unhelpful in the previous equal-capacity comparison, so merely making it fit is insufficient.

### 3. Agentic DSpark: workload-specific trained weights

[tiyuvta's Agentic drafter](https://huggingface.co/tiyuvta/Qwen3.8-27B-DSpark-Agentic) is a distinct 1.36B-parameter checkpoint trained from scratch on agentic sessions and chat prompts regenerated by the target.
Its published acceptance lengths are 2.88 and 2.92 for two held-out session buckets.
Its 1.48x greedy speed claim compares against plain decoding on Blackwell, not stock DFlash2.
It ships BF16 safetensors, not a ready GGUF.
Its five layers use full attention, different feature taps, and YaRN configuration, so smaller weights do not guarantee lower total memory at 196K.
Conversion, feature alignment, quantization, and full-context loading need validation.
It is interesting enough for a compatibility study, but not an established upgrade.

### 4. DaoCloud experimental DFlash2: different query layout

[DaoCloud's checkpoint](https://huggingface.co/DaoCloud/Qwen3.8-27B-DFlash2-Exp) produces seven proposals from seven query positions instead of eight, using `sample_from_anchor=true`.
It publishes acceptance measurements over eleven benchmark groups on H200/vLLM, but no matched throughput comparison against our baseline.
It has 81 tensors and the same approximate parameter count as stock, but a different configuration schema and query convention.
[vLLM PR 54154](https://github.com/vllm-project/vllm/pull/54154) remains open as checked today.
The local llama.cpp source has anchor-sampling machinery, but the inspected DFlash converter expects the stock nested configuration, and the proposal-count special case is DSpark-specific.
Existing machinery does not establish support for this checkpoint.
Do not assume that changing metadata alone implements the required geometry.

### 5. DimInfer: quantized-target training precedent

[DimInfer's DSpark](https://huggingface.co/DimInfer/Qwen3.8-27B-Dspark-v1) reports training from Q4_K_M target-generated responses and hidden states, using 40,000 prompts and four RTX 4090D GPUs.
Its published speedups compare with speculation disabled on a 48 GB 4090D, at greedy sampling.
The repository provides Q8 and BF16 GGUFs; a smaller quant would be needed for a realistic full-capacity screen here.
Its card still lists release of the complete capture/training pipeline as future work.
This is useful evidence that quantized-target training has been attempted, but it does not establish superiority over our drafter or reproduce our target.

## Other leads and exclusions

- [jfan's Heretic DFlash2](https://huggingface.co/jfan/Qwen3.8-27B-heretic-dflash2) targets a different modification of Qwen; its own three-prompt comparison is essentially tied on one prompt and slower on two.
- [Apathy v3](https://huggingface.co/onewhosighs/Apathy-Qwen3.8-27B-DFlash-drafter-v3) uses six layers and a different runtime; its fixed-prompt GB10 result includes target-KV changes and is not an isolated drafter win.
- [syvai W4A16](https://huggingface.co/syvai/Qwen3.8-27B-DFlash2-W4A16) is a calibrated stock drafter for vLLM/Marlin, not a GGUF replacement or newly trained model.
- [syv-ai's runtime](https://github.com/syv-ai/qwen38-27b-rtx3090) adds context-lookup drafting and reports gains on a 3090, especially when output repeats input text; this is a separate engine-level lead whose headline figures do not transfer to Marathon.
- The [memra package](https://huggingface.co/tiyuvta/Qwen3.8-27B-DFlash2-memra) reports vocabulary trimming and adaptive verification benefits, but our [previous shortlist experiment](LLAMAMPERE_EVALUATION_2026-09-11.md) already found only small gains with a memory regression and one greedy divergence.
- Very recent packages for Tenstorrent or alternate fine-tuned target bundles do not demonstrate a compatible improvement for our unchanged target and NVIDIA runtime.

## Training our own

[SpecForge PR 772](https://github.com/sgl-project/SpecForge/pull/772) merged DFlash2 training support on August 30.
Its [current documentation](https://github.com/sgl-project/SpecForge/blob/3d64e7a61f5fcc7f7d78ba6164c881f831943947/docs/sections/concepts/DFlash2.md) covers pretrained initialization, selector loss, offline feature input, and Hugging Face export.
This is more directly relevant than a generic DFlash1 recipe.

The [Qwen3.8-specific recipe](https://github.com/sgl-project/SpecForge/blob/3d64e7a61f5fcc7f7d78ba6164c881f831943947/docs/recipes/qwen3.8-27b-dflash2-disaggregated.md) matches stock DFlash2's architecture and target taps.
Its measured training layouts use eight or sixteen H200/B300 GPUs with patched SGLang feature capture.
Those examples neither prove consumer GPUs are unsuitable nor establish a working four-3090 training setup.
The documented target formats do not implement capture from Marathon's IQ4_XS GGUF.

The exact-target requirement is the central engineering gap.
Saved text gives token sequences but not the intermediate layer activations or target-head behavior needed to reproduce this training setup.
Our data capture must preserve tokenization, position conventions, feature taps, and the deployed quantized target's computations.
Using ordinary BF16 Qwen features would weaken the claim that we are adapting to the exact modified and quantized target.

[DaoCloud's public corpus](https://huggingface.co/datasets/DaoCloud/Qwen3.8-27B-Drafter-SFT) contains 450,401 tokenized training rows and a prompt-only configuration.
It includes reasoning-effort labels and loss masks but no hidden states.
Its prompts could supplement a local training set after regenerating responses with our target; it is not ready-made exact-target supervision.

[Quantize the Target, Quantize the Drafter](https://arxiv.org/abs/2607.04244) studies high-precision pretraining followed by low-precision-target adaptation on Qwen3.5-4B.
It supports investigating adaptation as a method, but its combined-system speedup is not an estimate for Marathon or a DFlash2 result.

## Recommended next experiment

1. Compare stock Q4, published calibrated Q3, and mixed Q2/Q3 on the unchanged target at the full 196K allocation and current six-token policy.
2. Retain cold/warm separation and before/after controls; check short and long contexts, real tool turns, and image continuation.
3. Measure completed-task latency and correctness, plus draft time, verification time, accepted tokens per pass, and peak memory.
4. If calibration looks useful, reproduce activation collection against our exact target and produce controlled calibrated-Q4 and calibrated-Q3 variants.
5. If trained weights remain attractive, qualify Agentic DSpark conversion and DaoCloud's query contract separately before inference comparisons.
6. Start a custom training pilot only after an exact-target feature-capture and export round trip succeeds.

The training pilot should initialize from stock DFlash2, keep the target frozen, and split train/evaluation data by project or conversation rather than adjacent turns.
Select checkpoints using held-out served latency and correctness, not training loss or acceptance alone.
No numerical speedup or training duration is justified by the evidence collected here.

## Retained evidence

Raw search responses and 18 inspected repositories' metadata/cards/configurations are retained under `.marathon/diagnostics/drafter-research-20260914/`.
`candidate-manifest.json` records repository revisions, artifact sizes, and published LFS hashes without downloading weights.
The three inspected commit histories distinguish weight uploads from documentation updates.
These files support this dated investigation and are not an automatic update service.
