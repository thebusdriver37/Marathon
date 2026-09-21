# Marathon runtime profile, September 20, 2026

The measured bottleneck is target-model computation, predominantly quantized matrix multiplication, with a growing attention cost at longer context.
This diagnostic establishes no deployable speed improvement and does not establish an absolute hardware or architecture ceiling.
Keep the production configuration unchanged.

## Production-matched results

| Measurement | 4,096 occupied tokens | 75,000 occupied tokens |
| --- | ---: | ---: |
| Untraced decode, before / after | 61.51 / 61.67 tok/s | 54.51 / 54.58 tok/s |
| Traced decode | 57.00 tok/s | 51.30 tok/s |
| Additional decode time during tracing | 8.06% | 6.32% |
| Target share of summed GPU kernel time | 88.23% | 89.87% |
| Quantized matrix multiplication, target plus draft | 2,739.63 ms | 2,665.16 ms |
| Flash attention, target plus draft | 99.05 ms | 720.15 ms |
| Total GPU kernel time | 3,730.25 ms | 4,211.58 ms |
| CUDA copies plus memset | 74.48 ms | 77.66 ms |
| State get/set host ranges, target plus draft | 53.66 ms | 53.69 ms |
| Mean GPU span of a seven-row target call | 41.09 ms | 48.06 ms |

Each measured request generates 256 tokens.
GPU totals cover the cached request, including processing its four-token uncached suffix, rather than a perfectly isolated decode-only range.
The initial full prompt fill is outside the captured trace.
The long prompt fill took 101.65 seconds, approximately 738 tokens/s.

Outputs and token hashes matched exactly before, during, and after tracing at each context length.
Draft proposal and acceptance counts also matched: 528/165 at 4K and 508/168 at 75K.
The two context lengths produce different outputs, so their throughput difference is descriptive and is not a controlled estimate of context cost alone.
The 75K replay did not reproduce the previously reported sustained slowdown below 20 tokens/s.

## Interpretation

Quantized matrix multiplication accounts for approximately 73% of summed kernel time at 4K and 63% at 75K.
Flash attention rises from approximately 2.7% to 17.1%.
These kernel names and timings do not independently establish whether each kernel is limited by memory bandwidth, arithmetic throughput, or occupancy.

Target calls dominate the traced GPU work; draft kernels occupy the remaining approximately 10-12%.
Better draft acceptance could amortize the expensive target passes across more useful output tokens.
Making only the draft computation cheaper has a much smaller opportunity.
A larger draft is useful only if its additional accepted tokens outweigh its added runtime and memory costs.

CUDA copy and memset time is small relative to the request duration.
That figure does not include all data movement: tensor-copy, gather, and concatenation kernels contribute another approximately 259-265 ms.
Measured state get/set operations are approximately 54 ms per cached request and do not explain a multi-second decode stall here.
The target recurrent delta-net kernels contribute approximately 135-140 ms.

Device activity occupies approximately 83-84% of the interval between the first and last recorded GPU activity.
The remaining interval is not a measured recoverable speedup: tracing itself adds overhead, and required CPU preparation and dependencies can create gaps.
Large CUDA synchronization API durations mostly overlap real GPU execution and must not be counted as additional removable CPU work.
Projected GPU ranges can overlap and must not be summed with CPU ranges as independent elapsed time.

## Method and boundaries

The final run uses the pinned production container `sha256:fc98366ec06248a2b9d4df9d37e4fe86d6419fc79598e0d40fb0c63554c038a1`, its CUDA libraries, inherited environment, and configured server flags.
GPU 3 was exclusively leased and initially unloaded, at its unchanged 250 W power cap.
The merged IQ4_XS target, R32 Q4_K_M drafter, Q8/Q8 target KV, Q4/Q4 draft KV, six neural proposals, lookup settings, and 196000 context capacity were unchanged.
The diagnostic worker used a separate cache directory and loopback port.
No production checkpoint was loaded or modified.

Nsight Systems 2023.4.4 recorded CUDA graph nodes, CUDA API calls, and NVTX ranges.
A diagnostic-only interposition library labeled target/draft decode, feature encoding, synchronization, and state get/set functions while forwarding their arguments unchanged.
It introduced no additional synchronization calls or model calculations.
Each context had an untraced warmup, an untraced measurement, a traced measurement, and another untraced measurement.
Here, untraced means collection disabled in a profiler-launched process, not a claim of zero residual profiler instrumentation overhead.

The local fixture is reconstructed from one of the three conversations previously authorized by the user.
The reconstructed available history rendered to 94,334 tokens; the test uses its 4,096- and 75,000-token suffixes with the same final request.
The reconstruction omits unavailable encrypted reasoning, folds a developer message into the leading instructions, and replaces one malformed recorded tool argument.
Suffix truncation and reconstruction make this a runtime replay, not the exact original request or a full Marathon client/tool execution.
Recorded tool calls were treated as data and never executed.
No conversation text, token IDs, identifiers, paths, or hashes were sent to Jev.
This screen does not qualify behavior at 196K occupied context, across images, across conversation switches, or on every production workload.

## Preliminary runs excluded from production conclusions

The initial standalone launches used copied runtime libraries but inherited the host CUDA user libraries and omitted the image's `LLAMA_REUSE_SCHEDULER=1` environment setting.
Those runs were used to establish trace collection and phase labeling, not to report production-matched attribution.
The second preliminary run was stopped during long-context prefill once the launch discrepancy was identified.
The final container run restores the exact image environment and CUDA library versions.
Scheduler reuse lowered cached-request setup time in this comparison; it was already deployed and is not a new gain.

## Next step

Jev received only a curated aggregate evidence packet and selected a Verification-Aware Training novelty audit ahead of another cache sweep or immediate training.
Its judgment is advisory and does not establish an improvement or a calibrated success probability.
Compare the auxiliary verification-survival head and rejection-adaptive loss with the already tested rejection-span, DPACE, selector, and server-calibration work.
Resolve the known offline-versus-quantized-serving prediction mismatch before promoting any training result.
If the mechanism is materially distinct, a later candidate must earn its place through actual serving acceptance, speed, and accuracy checks.

A target-kernel source and build-provenance audit is the other supported direction, particularly the dominant quantized matrix operations and long-context attention.
Do not repeat the previous withheld short-attention CUDA build without isolating its unresolved compiler/kernel regression.
The profile supplies a cost ranking, not evidence that a replacement kernel is faster.

## Artifacts and final state

Authoritative traces, attribution, request timing records, and the final protocol are under `.marathon/diagnostics/runtime-profile-20260920/production/`.
The protocol records the exact container launch, configuration hash, profiler version, and hashes of the diagnostic source files.
The parent directory retains the diagnostic harness, NVTX wrapper source, analysis code, preliminary traces, private fixture, and aggregate Jev exchange.
These are ignored local diagnostic artifacts, not model files or material added to Git history.
The directory is restricted to the local user.
Retain it to reproduce or investigate this profile; the shareable report contains no conversation content.

The diagnostic worker was stopped and GPU 3 returned to 15 MiB idle usage.
The GPU-control repository and configuration hash remain unchanged.
Production GPU 1 was not stopped or reconfigured.
No quality-affecting optimization was deployed.
