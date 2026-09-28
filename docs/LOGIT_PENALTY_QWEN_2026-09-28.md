# Logit-penalty experiment on the promoted Qwen merge

## Research and interpretation

The linked [Meta paper](https://arxiv.org/html/2606.00206v1) subtracts a fixed
logit penalty from 50 English hesitation/redirection markers at every decoding
step. Its models are R1-distilled Qwen/Llama and QwQ-32B, not our Qwen 3.8 merge;
its quantizers are GPTQ, AWQ, and FlatQuant, not GGUF IQ4_XS. It studies math,
coding, and science tasks at temperature 0.6 and top-p 0.95. The proposed
mechanism is that quantization disproportionately perturbs uncertain branching
positions. This is evidence for a particular failure mechanism, not proof that
all hesitation is waste.

Important qualifications: headline tables select the best accuracy from eight
penalty strengths; their error bars describe variation across strengths, not
independent-seed confidence intervals. Some configurations lose accuracy.
Larger models generally have smaller savings. Generalization beyond the tested
domains is explicitly open. A whole-output penalty can affect the final answer
as well as reasoning.

The older [NoWait paper](https://arxiv.org/html/2506.08343v2), linked in a comment,
uses stronger suppression and expanded keyword variants. It is related, not
the same intervention. Its detailed results also include accuracy losses,
including substantial multimodal losses, despite optimistic abstract wording.
It is not evidence that removing all reflection is safe.

## Adversarial information from the thread

Reviewed the post and the expanded, new-sorted [comment page](https://www.reddit.com/r/LocalLLaMA/comments/1wromzr/adding_logit_penalty_for_wait_maybe_and_perhaps/?sort=new).
Unexpanded nested replies were not assumed to be covered.

- A commenter warns that repeating `--logit-bias` can retain only the final
  flag in newer llama.cpp. We use a single per-request JSON map and verify it.
- Token IDs and word boundaries matter. ` recheck` splits into multiple tokens
  here. Penalizing its fragments would affect unrelated words. It is omitted,
  matching the post's 49-ID list rather than pretending to implement 50 tokens.
- Coding feedback warns against broad, strong suppression. Common words such
  as `or`, `error`, and `wrong` can be necessary grammar, identifiers, or useful
  correction signals. Literal strings, tool arguments, and code get checks.
- One observational Flash Next analysis suggests mild penalties on wait/hmm/
  actually and reports that many hesitations were useful. Its judge agreement
  was limited; it was not an intervention experiment. We treat it as a hypothesis.
- Reported Swift experiences are mixed. We test the actual promoted merge,
  not infer additivity from another model's reported speedup.

## Frozen comparison

Runner: `scripts/evals/logit_penalty_probe.py`.
Raw artifacts: `.marathon/diagnostics/logit-penalty-20260928/controlled/`.

- Baseline: no bias.
- Broad: -2 on 49 single-token, leading-space markers from the post.
- Narrow: -1 on 11 single-token space/no-space and capitalized variants of
  wait, hmm, and actually. Bare lowercase `hmm` is multi-token and omitted.
- Actual promoted Swift/uncensored Qwen 3.8 27B IQ4_XS, existing DFlash2 drafter,
  existing runtime image and KV types. Extra-high reasoning, temperature 0.6,
  top-p 0.95, top-k 20, min-p 0, repetition penalty 1.0 in every arm.
- 42 cases, two seeds (17/29), three arms: 252 case-runs / 264 requests.
  Thirty seeded integer-answer MATH-500 problems (18 level 4/5, 12 level 1-3),
  four implementation tasks, a falsy-value code repair, three evidence/logic
  checks, an exact-string check, two tool-call checks, one three-turn conversation.
- Each case and all its repeats/arms stay on one GPU; arms are counterbalanced.
  Three GPUs process different cases concurrently. No cross-GPU latency comparison.
- Code is independently executed with fixed tests in a network/filesystem sandbox.
  The tests were checked against correct reference implementations and broken code.
  Numeric answers and structured outputs are graded without an LLM judge.
- Both seeds and all arms share an 8,192-token output limit. Length truncations
  count as incomplete, not quietly excluded. Infrastructure errors stop the run.
- Cache reuse is disabled equally. Warmup and a positive-bias API check precede
  measurements. Reasoning token counts are separately retokenized estimates;
  total output counts and timings come from the server.

## Infrastructure qualification

The first, smaller screening run used the full production context allocation
(196,000 tokens). Two exclusively leased workers ran out of CUDA memory,
including during a baseline request. The interrupted pilot is retained at the
artifact root for diagnosis and must NOT be included in efficacy statistics.

The controlled run uses private copies of the existing worker commands with
65,536-token capacity for ALL arms. It omits production snapshot mounts and
slot saving. Models, quantization, speculation, GPU placement and sampling are
otherwise unchanged. All three normal pool locks remain held during testing.
Production broker configuration, OMP/Marathon settings and model files are not
modified. Private workers are stopped and leases released on exit.

## Scope

This is a direct model/API screening experiment, not an end-to-end Marathon or
OMP speed benchmark. The exact-answer subset, short contexts, two seeds, and
small coding suite limit generalization. Shared seeds help pair comparisons but
do not guarantee bitwise reproducibility under speculative/GPU execution.
No production promotion is implied by a positive mean. A promising candidate
still needs actual multi-turn agent testing and long-context validation.

Summarize a completed run with:

```sh
.marathon/venv/bin/python scripts/evals/logit_penalty_probe.py \
  --output .marathon/diagnostics/logit-penalty-20260928/controlled --summarize
```

The summary reports per-domain outcomes, paired gains/losses, per-seed results,
and 95% paired bootstrap intervals clustered by case (5,000 draws).

## Completed results

All 252 case-runs completed without infrastructure errors in the controlled
run. A case-run reaching its output limit is scored incomplete/unsuccessful,
not treated as evidence that the model could never solve the task.

| Arm | Successful case-runs | Output tokens | Summed response time | Incomplete |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 81/84 | 56,213 | 699.68 s | 1 |
| Broad -2 | 82/84 | 47,539 | 600.12 s | 1 |
| Narrow -1 | 81/84 | 56,964 | 718.14 s | 2 |

Times are sums across requests, not elapsed time of the three-GPU experiment.
Broad used 15.43% fewer tokens and 14.23% less response time. Narrow used
1.34% more tokens and 2.64% more time. Broad was faster in both seed totals
(17.3% and 10.6%). These repeats do not establish a universal speedup.

Case-cluster bootstrap 95% intervals for broad's changes: output tokens
[-35.42%, +1.35%], response time [-34.63%, +1.32%]. Both include no benefit.
The median paired token change is zero for both penalties. Excluding the
polynomial case leaves broad at -6.75% tokens / -6.34% time; excluding both
that case and retry implementation leaves -0.84% tokens / -0.66% time.
These are post-hoc sensitivity checks, not alternative headline scores.

Aggregate decoding rates were 84.17 / 83.52 / 82.72 tokens/s for baseline /
broad / narrow. Draft acceptance was 43.77% / 43.38% / 42.87%. This is evidence
of changing how much work gets generated, not faster inference kernels.
Aggregated rates also depend on the generated workload mix.

### Quality and concrete receipts

- Math: 58/60 baseline, 59/60 broad, 59/60 narrow.
- Executable coding checks: 9/10 baseline, 9/10 broad, 8/10 narrow.
- JSON checks: 8/8 each; tool-call checks: 4/4 each; three-turn conversation:
  2/2 each. Tool tests verify call shape/arguments, not a full agent tool loop.
- Broad has two paired successes where baseline failed and one paired failure
  where baseline succeeded. Narrow has one of each. Equal or better aggregate
  scores therefore do NOT mean no task-level regressions.

Polynomial, `math-412`, seed 17:

> Let P(x) be quadratic, x^2 - 2x + 2 <= P(x) <= 2x^2 - 4x + 3 for all real x,
> and P(11)=181. Find P(16).

Baseline returned `{"answer":408.25}` after 6,589 tokens / 74.41 s. Broad
returned the correct `{"answer":406}` after 1,140 tokens / 12.76 s. Narrow
also answered correctly but used 6,883 tokens. Independent verification:
both bounds meet at x=1 and force P(x)=a(x-1)^2+1; a=1.8, hence P(16)=406.
On seed 29 baseline already solved it in 974 tokens, versus broad's 1,035.

Retry implementation, `code-retry`: all arms passed executable tests on both
seeds. Baseline to broad was 3,042 to 1,854 tokens (38.41 to 22.50 s) on
seed 17 and 2,803 to 1,063 tokens (32.99 to 13.57 s) on seed 29.

Graph implementation, `code-cycle`: seed 17 baseline delivered passing code
in 6,914 tokens / 96.02 s; both penalties exhausted 8,192 tokens in reasoning
without delivering code (117.99 / 120.00 s). On seed 29 broad passed in
6,568 tokens / 91.81 s while baseline and narrow exhausted the budget.
The failure is failure to complete within the fixed budget, not a scored
incorrect implementation or evidence of an inherently unsolvable problem.

Exact requests, responses, code-test outcomes, and timings are in the named
case/seed/arm directories under `controlled/`. `summary.json` contains full
domain, seed, and paired statistics. No private conversations were accessed.

## Recommendation and cleanup

Retain broad -2 as an experimental candidate, not a production default.
There is a measured effect worth knowing about, especially on retry code,
but it is concentrated and not demonstrated to preserve quality task by task.
The narrow variant offers no compelling benefit here. This is not another
claimed Marathon harness speedup: no actual Marathon/OMP sessions were timed.
If pursued, the next gate is a small, held-out real-agent comparison, not a
larger sweep selected on these same answers. Long-context safety remains untested.

The three owned test containers were stopped and removed, worker leases were
released, and GPUs 1-3 returned to 15 MiB each. The production broker config
hash remained unchanged. No model weights, production prompts, OMP settings,
or Marathon defaults were changed. Retained artifacts are the reusable runner,
this report, public source snapshots, and synthetic test receipts.
