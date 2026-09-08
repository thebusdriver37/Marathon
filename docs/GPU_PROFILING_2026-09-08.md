# Qwen and RTX 3090 profiling, 2026-09-08

The current configuration remains a good measured operating point.
This session identified kernel work worth investigating later, but did not establish a new optimization to deploy or prove a global Pareto optimum.
No runtime tuning changes were retained.

The subsequent [kernel optimization trials](GPU_KERNEL_TRIALS_2026-09-08.md) tested scheduling, smaller tiles, and prefetching without finding a candidate worth deploying.

## Configuration and method

Measurements used physical GPU 2, an RTX 3090 at its existing 250 W limit, through the registered llama-swap worker and native chat completion endpoint.
The configuration used Qwen3.8 27B IQ4_XS, Q8_0 target KV cache, Q4_0 draft KV cache, DFlash2 with window 6, batch 1024, microbatch 256, and requested context 196000.
The production image was pinned to `sha256:443d87c87fe673faf379b29dd11bfa22e03c76e4d2fe14e9ce9503c810daf34c`.
No power limits or clock settings were changed.

A real 70-message conversation replay supplied the short fixture.
The long fixture inserted synthetic archived build records into that conversation.
Each request generated 256 tokens with fixed seed 424242, temperature 1, prompt caching, and EOS ignored for a bounded measurement.
Four cached repeats per fixture supplied the uninstrumented baseline after cold prefill.
These are inference measurements through the broker, not a new full Marathon UI end-to-end suite.

| Fixture | Input tokens | Cold prompt processing | Cached median request time | Cached median decode |
| --- | ---: | ---: | ---: | ---: |
| Short conversation | 15,102 | 16.70 s | 4.888 s | 56.62 tokens/s |
| Long conversation | 120,130 | 190.12 s | 4.734 s | 59.98 tokens/s |

Cached repeats reused all but four input tokens.
The two fixtures produce different text and speculative acceptance rates, so this is not a controlled context-length scaling comparison.
The short fixture accepted 158 of 577 drafted tokens, while the long fixture accepted 181 of 431.
That workload dependence helps explain why the longer fixture decoded faster here.

## Kernel observations

Nsight Systems 2023.4.4 recorded CUDA graph nodes and GPU metrics on the test GPU only, with CPU sampling disabled.
Cold prefill occurred outside active capture.
The short and long decode captures each matched their respective baseline input hash, output hash, reused-token count, generated-token count, and draft acceptance counts.
Profiler instrumentation increased request duration, so captured latency is not used as the performance baseline.
One resumed long request processed an uncached remainder and was excluded from warm comparisons.

| Kernel family | Short decode GPU kernel time | Long decode GPU kernel time |
| --- | ---: | ---: |
| IQ4_XS matrix operations | 45.10% | 37.06% |
| IQ4 fixups | 6.21% | 5.12% |
| Attention | 4.32% | 24.00% |
| Q6_K matrix operations | 7.38% | 6.13% |

Percentages use summed kernel durations, not request wall time.
The short capture contained 364,854 kernel activities and the long capture contained 275,137.
IQ4_XS J8 matrix operations remain the leading short-context kernel target, while attention becomes a substantial second target at 120K tokens.
Sampled DRAM read throughput averaged about 39% and 37% of the reported peak over the respective kernel spans.
Those averages do not imply an available twofold speedup, because instruction execution, dependencies, occupancy, and host scheduling also constrain progress.
Neither decode capture recorded `cudaMallocHost` or `cudaFreeHost` runtime calls, consistent with the earlier allocation reuse improvement remaining effective.
Large stream synchronization durations largely represent waiting for GPU work and must not be counted as independent CPU overhead.

A separate changed-suffix capture rewound to 14,964 cached tokens and processed 157 tokens before generating one output token.
Its 899 ms instrumented request included real kernel work and substantial transfers.
It does not directly explain the previously observed 27-token cached follow-up latency, because the rewind path and instrumentation differ.

## Decision and restoration

Retain the current production settings.
Future kernel experiments should target IQ4_XS J8 and its fixups first, with long-context attention as a separate workload target.
Any candidate needs uninstrumented repeated benchmarks and output validation across representative workloads before deployment.
This session did not repeat the earlier power-limit sweep or compare alternative kernels, so it cannot establish their current Pareto frontier.

Changing only the test worker definition unexpectedly caused installed llama-swap v250 to restart all managed workers, interrupting active requests on other workers.
The documentation had understated the scope of reloads and has been corrected.
After the user authorized continuation, the original broker configuration was restored byte-for-byte.
Its SHA-256 is `9da896acf2d0686bd7e523b895a44a4ba05686d5249245e36f74ca81e4dd534d`.
The broker returned healthy, worker 1 was healthy after restarting, and GPUs 2 and 3 were unloaded at final inspection.
No profiler worker remained running.

## Evidence

Local raw measurements, response hashes, Nsight reports, SQLite exports, and aggregate summaries are in `/tmp/marathon-profile-20260908`.
That directory is temporary diagnostic evidence and may be removed by normal temporary-file cleanup.
Raw conversation responses are deliberately excluded from this repository report.
The preceding application end-to-end checks are documented separately in the existing evaluation records; this report covers the subsequent GPU profiling session.
