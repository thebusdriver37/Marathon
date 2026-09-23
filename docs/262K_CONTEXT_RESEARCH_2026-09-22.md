# 262K context on the current Swift merge and one 24 GB GPU

Follow-up: the [local Q5 probe](262K_Q5_PROBE_2026-09-22.md) allocated 262144 successfully, but both Q5 formats lost approximately 30-32% decode throughput at matched 128K input.
Neither was deployed; the research and proposed probe below describe the earlier hypothesis.

Research found credible ways to allocate 262144 context on a 3090-class card, but no matched demonstration preserving our exact Swift merge, drafter, Marathon capabilities, long-context accuracy, and speed.
The strongest next candidate is Q5 target KV with correctly compiled CUDA attention, retaining the existing weights and DFlash2 drafter.
This is a research conclusion and memory estimate, not a validated configuration or deployment.
No inference service or production setting was changed.

## What the public recipes actually demonstrate

| Source | Mechanism and evidence | Why it is not a direct answer for our setup |
| --- | --- | --- |
| [rndhouse context benchmark](https://github.com/rndhouse/qwen36-27b-rtx3090-context-bench) | Qwen3.6 27B, Q4 weights, Q4 KV, MTP off; 262144 allocation at 20543 MiB | Different model/backend route; approximately 41 tok/s is a short generation probe, not demonstrated speed at a nearly full 262K prompt; no broad quality test |
| [HyperQwen long-context results](https://github.com/syv-ai/HyperQwen/blob/main/docs/long-context.md) | Qwen3.8 on 3090 using vLLM and KVarN compressed KV; retrieval passes at 240K and small reported perplexity change | Single-user MTP results at 112K fall from 68.1 to 32.0 tok/s; good evidence that fitting and preserving some quality metrics do not imply preserving speed |
| [MiaAI EXL3 recipe](https://github.com/MiaAI-Lab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw/blob/main/README.md) | Recommends Hadamard 4-bit KV for Ampere and 262K with MTP | Different target weight format and engine; documented performance is measured on Spark, with RTX performance explicitly unbenchmarked |
| [fast-long-context](https://github.com/satellitedown/fast-long-context) | Tested 260K input with NVFP4 target/KV, DFlash2 and SGLang patches | RTX 5090 32 GB, different target, and thinking disabled in the speed test; not a 3090 recipe |

The [official model configuration](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json) specifies 262144 maximum positions, 16 full-attention layers among 64 trunk layers, four KV heads, and head dimension 256.
Native architectural capacity is not a guarantee of equal task accuracy across context lengths, particularly for a merged checkpoint.
No RoPE extension should be assumed necessary merely to reach that native limit.

## Why Q5 is worth checking before Q4

An [independent developer benchmark](https://anbeeld.com/articles/kv-cache-quantization-benchmarks-for-long-context) compares Q5, Q4, Q8 and TurboQuant on Qwen3.6 27B, including IQ4_XS weights at 128K.
Its Q5 cache has lower distributional distortion than Q4 on the reported tests, with broadly similar prefill throughput among ordinary scalar formats.
These are related-model perplexity/KL results, not a 262K Swift coding or tool-use accuracy demonstration.
The article's speed column is prefill, not decode, and its derived precision percentages must not be presented as task accuracy.
It supports a candidate to test, not a promise of preserved capabilities.

For our 16 trunk attention layers, cache storage is estimated from:

`16 layers * 4 KV heads * 256 elements * 2 (K and V) * context * bytes_per_element`.

Q8_0 uses 34 bytes per 32 elements, Q5_1 uses 24, Q5_0 uses 22, and Q4_0 uses 18, including per-block metadata.

| Context and target cache | Estimated target attention KV |
| --- | ---: |
| Current 196000, Q8_0/Q8_0 | 6.355 GiB |
| 262144, Q8_0/Q8_0 | 8.500 GiB |
| 262144, Q5_1/Q5_1 | 6.000 GiB |
| 262144, Q5_0/Q5_0 | 5.500 GiB |
| 262144, Q4_0/Q4_0 | 4.500 GiB |

These figures exclude weights, draft model/cache, recurrent state, checkpoints, CUDA allocations, scratch space and allocator padding.
The verified local model metadata is recorded in `.marathon/reviews/mechanism-review-20260920.md`.
The memory calculation makes Q5 at 262K plausible because its target-cache allocation is smaller than our current target cache, but total peak memory still requires a real measurement.
Q5_1 offers the less aggressive option with tighter headroom; Q5_0 offers another half GiB of space.
Neither has yet been benchmarked locally at 262K.

## The backend trap

The local deployed-source snapshot in `.marathon/optimization-queue/workspaces/q8-residual/source/fattn.cu` rejects Q5_0, Q5_1, and mixed K/V formats when `GGML_CUDA_FA_ALL_QUANTS` is disabled.
The retained `iq4-build-q8` build container reports `GGML_CUDA_FA_ALL_QUANTS:BOOL=OFF` in `/app/build/CMakeCache.txt`.
A [llama.cpp issue with runtime measurements](https://github.com/ggml-org/llama.cpp/issues/28455) reports CPU attention fallback for these formats and recovery after compiling the missing CUDA kernels.
Its measurements are another user's hardware, not proof of our eventual speed.
Merely adding Q5 launch flags to our current binary is therefore not an appropriate test of Q5 GPU performance.

Our [earlier merged-model probe](MERGE_KV_PROBE_2026-09-19.md) already encountered very slow mixed Q8/Q4 prefill, consistent with this missing-kernel mechanism, although that exact run was not profiler-confirmed.
Q4/Q4 saved about 3 GiB and passed the small retrieval/tool screen, but long-context decode often slowed.
The current custom attention optimization also specifically targets Q8, so retaining its benefit under Q5 cannot be assumed.

## Bounded next probe

1. Build an isolated candidate from the existing runtime revision with Q5 CUDA attention enabled, retaining the current custom patches and unchanged merged model/drafter.
2. Verify attention actually runs on the GPU and compare against Q8 at the same occupied context before spending time on a 250K prompt.
3. Check peak memory at a 262144 allocation, including long prefill, verification, and warm follow-up turns.
4. Compare independent retrieval, multi-hop reasoning, structured tool calls and code-regression checks at shared context lengths, then exercise approximately 240K-250K occupied input with output headroom.
5. Judge end-to-end time, prefill, decode and draft acceptance separately; do not infer capability preservation from a few needle checks or perplexity alone.

A full 262K request need not decode as fast as a 196K request, even with ideal kernels, because more historical attention data must be processed.
The relevant targets are no substantial extra penalty at matched context and acceptable measured performance at the expanded length.
For now, Q5 with proper GPU support is a plausible path worth probing, while no public result establishes a no-compromise drop-in upgrade for our exact setup.
