# Speculative acceptance evaluation, 2026-09-08

The acceptance gains seen in one earlier conversation did not generalize to this broader test.
Neither experimental IQ4 scheduling variant earned deployment.
Higher speculative acceptance is beneficial when it reduces overall response time without compromising task completion; slower individual kernels alone are not sufficient grounds to reject a variant.
This evaluation directly tested that overall tradeoff.

## Test design

Ten new public prompts covered Python implementation, Python debugging, SQL aggregation, arithmetic, structured extraction, meeting summarization, dependency scheduling, tool calling, constrained creative writing, and long-context retrieval.
The retrieval prompt occupied 25,183 tokens.
Each prompt used seeds 424242, 1729, and 8675309 at temperature 1.
Seed order rotated by task and was identical across variants.
Answers stopped naturally, with task-specific limits of 384-640 tokens; none hit its limit.
The production answers averaged about 87 generated tokens.

The four runs were production, double-block/L1, forced tiling, and a production return control.
Each run produced 30 scored answers, for 120 total, plus one unscored cold warmup per prompt.
Warmups used one generated token and established a reusable prefix before timing the three seeded answers.
Cold prefill and worker startup are excluded from the primary comparison.
Request wall time includes backend preparation and complete non-streaming response delivery through the native chat endpoint.
It does not include Marathon UI, router, tool execution, or the external answer checker.
Tool calls were validated but not executed.

All runs used the same RTX 3090 on GPU 3 at 250 W, full production context capacity, model files, KV types, and six-token DFlash window.
The model image was `sha256:443d87c87fe673faf379b29dd11bfa22e03c76e4d2fe14e9ce9503c810daf34c`.
Candidates preloaded only the experimental IQ4 translation unit from the preceding [kernel trials](GPU_KERNEL_TRIALS_2026-09-08.md).
The preload SHA-256 was `5c11fb47e5da688421794a12e75604384865882de7709e2bc1924a240e79b044`.
Mode 3 doubled the J8 block grid while retaining L1 preference; mode 2 forced tiling and omitted the Stream-K fixup.

## Main results

| Configuration | Paired response-time change | Accepted / drafted tokens | Acceptance | Task checks passed |
| --- | ---: | ---: | ---: | ---: |
| Production | Reference | 2103 / 2990 | 70.3% | 24 / 30 |
| Double blocks, retain L1 | 5.9% slower | 2095 / 3273 | 64.0% | 23 / 30 |
| Force tiling | 11.7% slower | 2109 / 3085 | 68.4% | 24 / 30 |

Time changes are geometric means of the 30 paired candidate/baseline latency ratios.
Each baseline is the geometric mean of its initial and return-control times, giving every task/seed pair equal weight.
Total response time across all 30 answers was 29.42 seconds for initial production, 32.30 seconds for double blocks, 33.20 seconds for tiling, and 29.29 seconds for the production return control.
Those totals weight longer answers more heavily than the paired geometric means do.

The double-block variant was faster in one of ten category averages and three of thirty individual pairs.
Tiling was slower in all ten category averages and all thirty pairs.
With a hierarchical bootstrap over prompts and then seeds, the exploratory 95% latency-change intervals were +0.4% to +15.0% for double blocks and +8.6% to +15.6% for tiling.
These intervals describe this small suite, not all possible Marathon workloads.

Restricting comparisons to pairs that passed task checks in both production controls and the candidate still found no established improvement.
Double blocks was 2.4% slower over 23 common passing pairs, with an interval from 1.1% faster to 6.1% slower.
Tiling was 11.5% slower over 24 common passing pairs, with an interval from 7.8% to 16.3% slower.
This secondary comparison avoids crediting obviously incomplete or incorrect answers as useful speedups, but conditioning on successful answers is not an unbiased population estimate.

The production repeat matched all thirty answer messages after excluding server-generated tool-call IDs, and reproduced generated-token and aggregate draft-acceptance counts.
Its paired geometric-mean response time differed by only 0.12%, and total response time differed by 0.44%.
There was no material production timing drift in this check.

## Answer checks and review

Python answers were executed as pure functions in bounded child processes against functional tests, including input preservation and a logarithmic probe budget for binary search.
SQL answers ran against an in-memory SQLite fixture with read-only authorization and expected result checks.
Other checks covered exact arithmetic, extracted records, scheduling dependencies and worker overlap, tool arguments, summary fields, and writing constraints.
All ten reference answers passed the checker self-test before inference began.

All configurations failed the three arithmetic cases and three constrained-writing cases.
For example, the correct red-unit remainder was 78, while the model returned 102.
Writing answers violated the requested word count or four-sentence structure.
Those are weaknesses exposed by this suite, not regressions introduced by either candidate.
Double blocks additionally produced a merge-intervals implementation that omitted the branch handling a disjoint interval and failed executable tests.
One additional observed failure is not enough to establish a general quality regression, but provides no reason to prefer that candidate.

An initial summary check was too strict about capitalization and an ambiguous cancellation field.
One candidate correctly stated in prose that the marketing email was cancelled and the release was not, but used the field value `No` to describe release status.
The checker was corrected to accept that interpretation only when the prose preserves both facts, and every answer was regraded uniformly.
This changed only the double-block summary at seed 424242 from fail to pass, increasing its total from 22 to 23.
Raw scores and responses were preserved separately from reviewed scores.

Manual summary review found the main release facts preserved across configurations, but several answers strengthened Noel's offer to help into confirmed assistance.
Task-check passes should therefore not be read as comprehensive semantic-quality certification.

## Original conversation with more seeds and a longer output window

The original 15,102-token conversation received an additional diagnostic check with all three seeds and a fixed 512-token output window.
This added nine timed requests across production and both candidates, reversing candidate order to production, tiling, then double blocks.
EOS was ignored in this diagnostic only, so it measures a fixed decode window rather than completed-answer quality.
It is a follow-up on the selected conversation, not an independent held-out prompt.

| Seed | Production decode | Tiling decode | Double-block decode |
| --- | ---: | ---: | ---: |
| 424242 | 63.87 tokens/s | 64.75 tokens/s | 57.17 tokens/s |
| 1729 | 77.13 tokens/s | 63.22 tokens/s | 63.64 tokens/s |
| 8675309 | 76.73 tokens/s | 63.42 tokens/s | 71.19 tokens/s |

Tiling retained a small benefit at the original seed, taking 1.3% less request time, but took approximately 20-21% more time at the other two seeds.
Across the three seeds, request time was 12.8% higher for tiling and 12.9% higher for double blocks on a geometric-mean basis.
The earlier 20-28% decode improvement over a single 256-token window was not a reliable general improvement.

## Decision, limitations, and reproduction

Keep production unchanged.
The broader workload results, quality checks, and longer replay all support this decision.
They do not prove that future kernels, other speculative policies, or particular workloads cannot improve.
Most primary-suite answers are short, the long-context primary task is retrieval, and this was not a full agent trajectory or media benchmark.
No substantial claim about model-wide answer quality follows from ten prompts and three seeds.

The reusable workload runner and checker are [scripts/evals/speculative_acceptance.py](../scripts/evals/speculative_acceptance.py).
It targets an already isolated worker and does not itself reserve GPUs or manage inference services.
The surrounding benchmark runner acquired Marathon's existing worker-3 lease and monitored broker conflicts.
No broker reload was triggered and active workers were not interrupted.
Final broker health returned HTTP 200, the test lease was released, and no scratch container remained running.
GPU 3 returned to 15 MiB used with idle clocks of 210 MHz core and 405 MHz memory at its unchanged 250 W power limit.
The broker configuration remained byte-for-byte unchanged, with SHA-256 `9da896acf2d0686bd7e523b895a44a4ba05686d5249245e36f74ca81e4dd534d`.

Temporary evidence is in `/tmp/marathon-acceptance-20260908`, subject to normal temporary-file cleanup.
It contains request fixtures, complete responses, raw and reviewed scores, timing and GPU telemetry, manifests, bootstrap analysis, original evaluator source, and the longer replay results.
The primary summary is `summary.json`; the diagnostic summary is `replay-summary.json`.
`review-changes.json` records the checker correction and evaluator hash.
The original conversation fixture remains outside the repository.
