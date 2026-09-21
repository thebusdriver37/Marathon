# Updated llama.cpp: bounded Marathon comparison

Both backends passed all four tested Marathon turns, but the updated candidate was not a clear end-to-end improvement.
Keep current production for now: coding took longer on the candidate, while reasoning was approximately unchanged.
No runtime deployment or central configuration change was made.

## What ran

The current production image was `sha256:8af4fa77e493f6765b7c66d4f6cbfb673e0add0919b470c11526e08d54365e04`.
The updated candidate uses upstream `ce8caa6e60a03093351d6016a818720e0d46f0fb` plus retained local features, committed in the isolated candidate checkout as `41350946125a79ff95015608c69ca9af11f334e8`.
The complete candidate diff is preserved in `scripts/experiments/upstream-backports-20260920/04-full-upstream-candidate.patch`; it is experimental and is not in the automatic production patch stack.

Both ran on GPU 3, an RTX 3090 at the unchanged 250 W limit, using the deployed Swift/uncensored IQ4_XS merge and Marathon R32 Q4_K_M DFlash2 draft.
Both retained the 196000-token configured capacity, target Q8/Q8 cache, draft Q4/Q4 cache, lookup plus DFlash speculation, and production server arguments.
The test used genuine Marathon CLI sessions through an isolated loopback adapter, with medium reasoning, temperature zero, seed 8123, and a 4096-token response cap.
Each backend received the same user prompts in fresh workspaces; each session's second turn resumed its first turn.
Only synthetic test conversations were accessed.

## Results

Decode rates aggregate server-reported generated-token intervals across the requests within each turn.
Wall time includes Marathon, inference, and tool execution.

| Task | Current wall time | Updated wall time | Current decode | Updated decode | Checks |
| --- | ---: | ---: | ---: | ---: | --- |
| Implement record parser | 44.79 s | 53.16 s | 83.66 tok/s | 104.66 tok/s | Both pass |
| Extend parser on follow-up | 21.11 s | 33.45 s | 97.96 tok/s | 99.48 tok/s | Both pass |
| Inventory arithmetic with background context | 23.87 s | 23.69 s | 118.55 tok/s | 118.10 tok/s | Both pass |
| Correct inventory on follow-up | 3.38 s | 3.16 s | 125.10 tok/s | 129.38 tok/s | Both pass |

Independent parser assertions covered valid input, invalid lines, strict-mode errors, empty input, and age filtering on the follow-up.
The reasoning answers correctly returned 280 sellable units without reordering, then 197 with reordering after the correction.

Updated coding generated 3235 and 2316 tokens, compared with 1939 and 1058 on production.
It also made 14 and 8 inference requests, versus 8 and 4 on production.
That additional work accompanied 18.7% and 58.4% longer coding completion times despite similar or higher measured decode throughput.
This is an observed end-to-end slowdown on these tasks, not proof of slower CUDA kernels.
Both versions reused cached context on follow-ups; the reasoning follow-up reused approximately 25.9K tokens.

## Patch disposition and compatibility

Retained features include IQ4 crossover tuning, local MMVQ/top-k changes, compact fused Q8 attention staging, cache preference, scheduler reuse adapted to upstream's graph-result cache, speculative/recurrent snapshot support, checkpoint-buffer reuse, legacy draft metadata compatibility, and per-request speculative length controls.
Upstream supplies the newer GDN normalization, checkpoint eviction rules, convergent attention barrier, image-position handling, and fused DFlash injection.
The old DFlash injection split and media-specific attention-reuse implementation were not carried over.
The candidate retains our dense attention implementation and explicitly disables upstream sparse-attention dispatch for that implementation.
This is a combined update, so the test cannot attribute behavioral differences to a particular upstream commit or patch decision.

Compilation required adapting the sparse-dispatch interface and using CUDA driver stubs for linking in the CPU-only build container.
The first startup attempt exposed a changed attention-launch argument order; passing an explicit dense-mode argument fixed the reproduced assertion.
The final candidate built successfully and completed the entire Marathon comparison.

## Scope and evidence

This was one run per task per backend, as requested, rather than a statistical performance or broad quality qualification.
The production run overlapped six CPU compilation jobs; the updated run did not, so small timing differences should not be treated as established gains.
The backend inputs also differed in workspace paths, generated content, and subsequent tool results.
Active prompt lengths were approximately 10K-26K, not a full 196K long-context soak.
Media inputs, disk snapshot restoration, and lengthy production conversations were not tested.

Raw commands, harness, requests, answers, checks, logs, binary hashes, and `comparison-summary.json` are retained under `.marathon/diagnostics/upstream-audit-20260920/`.
The candidate can be rebuilt with CUDA 12.8, GCC 14, SM86, Release mode, and the retained patch applied to the exact upstream revision.
Large binaries and test artifacts remain ignored rather than tracked.
GPU 3 returned to 15 MiB after testing, and existing Marathon workers were left running.
The central configuration SHA-256 remained `43e82ae3c3ab9cf0712d4fd2c20a75953fc167fdec5e2be1377102bbb3a40b70`.
