# Longer lookup verification: measured result

Two isolated synthetic copy-and-edit completions exceeded 200 tokens/second averaged across decoding, with unchanged target weights, target cache precision, configured context capacity, and power limit.
This is a specialized copy workload result, not a general chat improvement or a production deployment.

## Measurements

| Configuration | Whole decode, tokens/s | Qualifying 512-token window, tokens/s | Full request, seconds |
|---|---:|---:|---:|
| Lookup-only, 15 proposals, cold prompt | 208.23 | 252.33 | 17.68 |
| Lookup-only, 15 proposals, warm prompt | 210.36 | 254.22 | 14.66 |
| Production configuration, fresh control, cold prompt | 187.11 | 196.54 | 19.67 |
| Production configuration, fresh control, warm prompt | 186.07 | 194.08 | 16.58 |
| Lookup-only, six proposals, cold prompt | 103.89 | 125.05 | 32.48 |
| Lookup-only, six proposals, warm prompt | 116.04 | 128.39 | 26.44 |

The two-run mean decode gain against the fresh control is approximately 12.2%.
The six-proposal lookup-only follow-up also passed both exact checks with the same output hash, but was substantially slower.
Removing the neural drafter by itself therefore does not explain the win; longer lookup is essential in this measured configuration.
The six-proposal run-to-run spread also cautions against treating these two samples as a precise general performance estimate.
Full request time fell approximately 10.1% cold and 11.6% warm.
Each response produced 3048 tokens, ended naturally, passed an exact AST dictionary comparison, and had the same output hash as the preceding baseline tests.
The output hash is `209d04375b1a28fd6da18b3cf59ec2e255a46f64fad9451ff081f42738d36bdc`.
All 3048 streaming token IDs agreed with final cumulative counts, and the reported window times were independently recomputed from saved arrival timestamps.
The two lookup windows lasted 2.029 and 2.014 seconds.

The task is a 100-record Python dictionary copied with exactly one quota changed.
The prompt occupies roughly 3040 tokens; configured capacity remains 196000, with the runtime padding allocation to 196096.
This is not a measurement at 196000 occupied tokens.
Sampling was greedy with medium reasoning.

Hardware was one RTX 3090 Ti at its existing 275 W limit.
The merged IQ4_XS target and Q8_0 K/V cache were unchanged.
The existing runtime image was unchanged.
The diagnostic configuration removed the neural drafter and used `ngram-map-k` with match length 24 and proposal length 15.
The request verification cap was also 15.
The production control retained ngram length six and the trained DFlash2 drafter at six proposals.
Observed lookup-only GPU memory was 21672 MiB during generation, not a peak allocation measurement.

## Why the tweet mattered

The benchmark author decoupled lookup proposal length from neural draft length; this differs from our failed trial that enlarged both together.
Their implementation conditionally verifies longer prompt-derived sequences while keeping the neural block fixed.
That is the transferable idea, rather than copying their weight preparation or cache settings.
[Author's optimization explanation](https://github.com/syv-ai/HyperQwen/blob/main/docs/optimizations.md).

Their current documentation assigns the 381 tokens/second reproduction configuration 56K context capacity, measured with about 25K occupied.
It separately reports a 240K configuration at 67 tokens/second mixed and 164 while quoting.
Their engine uses a different quantization and serving stack, so these are not direct comparisons with our merged IQ4_XS model.
[Author's configuration table](https://github.com/syv-ai/HyperQwen/blob/main/README.md).

Their quality report describes four-bit weights with 16-bit activations in single-user mode and additional output-head quantization in the fast variant.
The reported evaluations do not suggest catastrophic degradation, but they do not establish equivalence to our model or unchanged quality on all workloads.
[Author's quality report](https://github.com/syv-ai/HyperQwen/blob/main/docs/quality.md).

## Interpretation and next implementation boundary

Jev selected the isolated lookup-only experiment over immediately patching the mixed runtime after reviewing the local failure site and the external mechanism.
Its judgments were advisory, not independent performance evidence.
The experiment demonstrates that longer prompt lookup can exceed 200 tokens/second under the weight, cache, context-capacity, single-card, and power constraints.
It does not establish that removing the drafter helps ordinary prose, nor that long lookup and the drafter fit together.

Local source inspection found that `common_speculative_n_max` already takes the maximum across lookup and neural proposal lengths.
However, the HTTP `speculative.n_max` schema defaults and hard cap use `params.speculative.draft.n_max`, and the slot also caps verification at the task's draft maximum.
A mixed implementation must separate the allowed verification length from the neural forward block length, preserve existing rollback and context-boundary checks, and keep short fallback rounds when no strong lookup match exists.
The successful lookup-only experiment does not bypass target verification.
Future mixed-runtime work must also validate stochastic sampling and long occupied context before deployment.

## Isolation and artifacts

The selected worker was initially unloaded and exclusively leased.
Diagnostic containers mounted a fresh empty slot directory.
No production conversations or saved production slots were read.
Jev received only curated aggregate evidence.
Other workers were untouched; the central configuration and power settings were unchanged.

- Harness: `scripts/evals/speculation_cost_probe.py`.
- Lookup result: `.marathon/diagnostics/speculation-lookup-copy-20260920`.
- Fresh production control: `.marathon/diagnostics/speculation-copy-control-20260920`.
- Six-proposal lookup-only control: `.marathon/diagnostics/speculation-lookup-six-20260920`.
- Jev evidence and decision: `.marathon/diagnostics/speculation-lookup-review-20260920`.

The window selector now retains both the predeclared minimum of 512 tokens and minimum duration of two seconds while advancing its left endpoint.
Earlier logic could discard valid longer windows above 256 tokens/second by shrinking below two seconds.
The lookup results reported here came from the original selector and satisfy both thresholds without reprocessing or relaxing them.
