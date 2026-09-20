# Mixed lookup and DFlash2: speed frontier and quality checks

The isolated mixed runtime keeps the original six-token DFlash2 neural block while allowing longer prompt-lookup verification.
On the original synthetic copy-and-edit fixture, 127 lookup proposals reached 469.17 tokens/second on the cold-prompt completion and 552.81 on the cached-prompt completion.
Both outputs passed exact AST comparison and matched the production output hash.
This is a specialized copying result on a short occupied context, not a general chat rate.
The tested 127-token candidate was subsequently deployed to Marathon on 2026-09-20 at the user's request.

## Constraints

All experiments use one RTX 3090 Ti on GPU 1 at the existing 275 W limit.
The Swift uncensored merged IQ4_XS target, Q8_0 target KV, Q4_0 neural draft KV, CPU vision projector, and requested 196000 context capacity remain unchanged.
The original trained DFlash2 R32 draft remains unchanged in the mixed configurations.
Only synthetic fixtures were used, with no production conversations or snapshot contents read.
All experimental workers are exclusively leased, use fresh empty slot directories, and are unloaded after each experiment.
During the experiments, the central llama-swap configuration and other active workloads were unchanged.

## Runtime change

The server's request-level verification maximum was coupled to `params.speculative.draft.n_max`, which also describes the neural block limit.
The underlying `common_speculative_n_max` already computes the maximum supported length across configured proposal methods, including ngram lookup.
The candidate initializes the request verification default and hard limit from that combined maximum while leaving the global neural draft configuration at six.
Existing acceptance, rejection, rollback, sampling, and cache-boundary checks remain in place.
This is an inference scheduling change, not further quantization or acceptance without target verification.

The patch is `.marathon/diagnostics/lookup-runtime-build-20260920/lookup-verification-limit.patch`.
The build directory contains the exact build script, source hashes, paired control and candidate server libraries, and Jev requests and answers.
Both libraries were rebuilt from the same local CPU server source and compiler; only the candidate adds the verification-limit change.
The common, model, and CUDA libraries stay those from pinned production image `sha256:765a84864664d953cc274adb0fdb161b793269857e4602d228460a790919cea8`.
Libraries are mounted read-only into isolated diagnostic containers; no installed binary or runtime image is modified.

Candidate server library SHA256: `38ac97b32f75bf255b6b2d905ec777f0b4db54cf45d00842e137482a0e12ad93`.
Rebuilt control server library SHA256: `1f616d3303e029d05a6305590dd93ecffc58bb059888cf301c40bb5b21e065f2`.

The same tested server library is packaged in experimental image `sha256:f3fb424f02f442240a1b6f939685c136a78b0ef021fd713de156a99405e1b1e8`, locally tagged `marathon-lookup-verification:experimental-20260920`.
Its model, common, and CUDA libraries were hash-compared against production and are identical.
The packaged server library matches the tested overlay exactly.
The Docker build context, image ID, and library comparison are retained in the build artifact directory.
This image is now assigned to the shared Marathon worker anchor, with lookup length 127 and neural draft length six.

## Copy speed frontier

Each cell is one naturally completed 3048-token output, with approximately 3040 prompt tokens.
All completed outputs below have hash `209d04375b1a28fd6da18b3cf59ec2e255a46f64fad9451ff081f42738d36bdc` and pass exact AST comparison.
Streaming token IDs agree with cumulative output counts.
Peak windows must contain at least 512 tokens and last at least two seconds; longer windows are used when throughput exceeds 256 tokens/second.

| Maximum lookup proposals | Cold whole decode | Cached whole decode | Best cold qualifying window | Best cached qualifying window |
|---|---:|---:|---:|---:|
| Production six | 187.11 | 186.07 | 196.54 | 194.08 |
| Mixed 15 | 228.59 | 231.89 | 243.85 | 242.83 |
| Mixed 31 | 346.85 | 359.24 | 389.89 | 392.49 |
| Mixed 63 | 468.11 | 499.81 | 572.65 | 566.37 |
| Mixed 127 | 469.17 | 552.81 | 634.99 | 668.04 |
| Mixed 255 | 361.03 | 542.99 | 680.50 | 772.30 |

All rates are tokens/second, excluding prefill from whole-decode rates.
For mixed 127, full request times including prefill were 9.89 and 5.71 seconds, versus 19.67 and 16.58 for the fresh production control.
The largest isolated peak is not the best full-response setting: 255 loses to 127 on both complete responses in this pair.
The cold and cached distinction includes kernel/graph warmup effects as well as prefix reuse; a single pair cannot isolate their contributions.
No confidence interval or universal optimum is claimed.

The existing `get_single_column_n_max` limits proposals to the remaining space before a 256-token padded attention boundary, including the anchor.
Therefore increasing the configured maximum beyond 255 cannot increase the actual block without a separate correctness-sensitive runtime change.
This audit stops the geometric width progression at that mechanism's boundary.
Jev favored validating 127 for useful completion speed rather than selecting 255 solely for its brief peak.

## Independent quality screen

The screen uses six synthetic tasks at greedy sampling and temperature 0.7, with separate fixed seeds.
Checks cover multiple edits that require rejecting stale copied values, revision precedence, integer-cent arithmetic, generated interval-merging code, exact tool arguments, and an override in an 18260-token document.
Generated code runs without network or home-directory access in a bubblewrap sandbox with time and memory limits.
Each code answer is checked against four explicit examples and 300 independently generated cases, including input immutability.
These are 12 task executions per configuration, not hundreds of independent model tasks.

| Configuration | Initial screen |
|---|---:|
| Production | 12/12 |
| Lookup-only 15 | 11/12 |
| Mixed 15, clarified coding prompt | 12/12 |
| Mixed 127, clarified coding prompt | 12/12 |

The original sampled lookup-only code merged integer intervals separated by a unit gap and failed the reference checks.
The phrase "touching closed integer intervals" permitted an interpretation the evaluator did not intend.
The amended prompt explicitly treats intervals as closed real intervals with integer endpoints and gives both merge and nonmerge examples.
Production and lookup-only each passed both clarified reruns; both mixed configurations also passed the clarified case at both sampling settings.
The original failed answer and original prompt are preserved, rather than overwritten or counted as a pass.
This resolves that fixture ambiguity but does not prove universal distributional or quality equivalence.

Lookup-only was far slower on tasks requiring new text, even when correct: approximately 36-42 tokens/second on arithmetic and code.
Mixed 127 retained the neural fallback: arithmetic measured 146-152, tool calls 113-133, and clarified code 134-143 tokens/second in this small screen.
Comparable production results were approximately 149-155, 113-136, and 133-140 respectively.
These small differences are not a demonstrated general-task speed improvement.
The short multiedit task was slower with mixed 15 but approximately matched production with mixed 127 in the measured greedy request.

## Near-capacity retrieval

Mixed 127 passed the three-needle retrieval check with 192449 actual prompt tokens and 260 generated tokens.
The requested markers appeared near the beginning, middle, and end of a synthetic archive, and all code strings and revision numbers matched exactly.
Cold prefill took 357.74 seconds, averaging 537.96 tokens/second; answer decoding measured 56.58 tokens/second.
This establishes a working near-capacity example, not short-context copy throughput at 192K occupancy or a comparative long-context speedup.
The artifact directory is `.marathon/diagnostics/quality-long-mixed127-20260920`.

## Long-context copy and verification

The additional copy fixture places the original Python registry before 7000 unrelated archive records, then asks for the same single edit and full file output.
This forces lookup back to the beginning of a long input and exercises long verification blocks during output, rather than only a short retrieval answer.

| Run | Actual prompt tokens | Generated tokens | Whole decode | Qualifying window | Full request |
|---|---:|---:|---:|---:|---:|
| Cold | 185097 | 3053 | 213.42 tokens/s | 270.20 tokens/s | 353.78 s |
| Cached | 185097 total, four newly evaluated | 3053 | 242.32 tokens/s | 271.76 tokens/s | 13.05 s |

Both completions passed exact AST comparison, ended naturally, and matched the original short-context output hash.
All 3053 stream token IDs matched cumulative counts on each run.
Cold prefill took 339.17 seconds; the cached request reused 185093 prompt tokens and evaluated four new tokens in 148.28 ms.
Thus this candidate exceeded 200 tokens/second on both complete copy decodes with 185K occupied context while retaining 196K capacity.
There is no same-fixture production long-context speed control, so these are absolute measurements, not a claimed relative long-context speedup.
Artifacts are `.marathon/diagnostics/speculation-long-copy-mixed127-20260920`.

## Varied-data copy at temperature 1.0

This held-out fixture replaces the original repeating regions and sequential quantities with deterministic random strings, random quantities, and mixed booleans.
It still requires exactly one quota edit and preservation of every other field.
Both arms use freshly rebuilt CPU server libraries from the same source and compiler, isolating the verification-limit change and lookup width from other rebuild effects.
The control retains six lookup proposals and six neural proposals; the candidate uses 127 lookup proposals and the same six neural proposals.

| Configuration | Cold whole decode | Cached whole decode | Cold full request | Cached full request |
|---|---:|---:|---:|---:|
| Rebuilt control | 183.58 tokens/s | 184.66 tokens/s | 23.45 s | 19.70 s |
| Mixed 127 | 491.91 tokens/s | 555.48 tokens/s | 11.15 s | 6.69 s |

All four responses ended naturally, passed exact AST checks, emitted 3601 tokens, and shared output hash `f249f72a46206f441d2b0433c5b463cfa251b6cc6fb0711014ce8487b1839d2f`.
The candidate's decode ratio was 2.68x cold and 3.01x cached on this fixture.
Its full request time fell approximately 52% cold and 66% cached.
This establishes a copying gain beyond the original greedy fixture while remaining a specialized copy/edit workload.
The artifacts are `.marathon/diagnostics/speculation-varied-{control,mixed127}-20260920`.

## Application validation

The first fresh Marathon application invocation stopped before inference because the minimal test catalog omitted the machine's pool configuration.
GPU-conflict protection refused startup and no active workload was interrupted.
The harness was corrected to retain the existing lazy pool bootstrap definitions while directing the explicit test model to the isolated candidate endpoint.
The original failed harness attempt remains under `.marathon/diagnostics/app-mixed127-20260920`.

The retry completed successfully through the installed Marathon CLI and its normal Responses/tool path against the isolated external endpoint.
It created only `solution.py` and `test_solution.py` in the fresh test workspace, then ran `python3 -m unittest -v` successfully with ten tests.
The resulting implementation also passed the separate 304-case evaluator, including input immutability.
The CLI exited zero, and the tool/file-change events were inspected to confirm the workspace scope.
Artifacts are under `.marathon/diagnostics/app-mixed127-retry-20260920`.
Even these passing checks do not establish universal equivalence for arbitrary prompts, long sessions, stochastic output distributions, vision, concurrent routing, or saved-slot compatibility.
This validates one application task; a production rollout must still respect active users and retains these coverage limits.

## Decision and deployment status

Jev's final aggregate-evidence review selected staging the 127-proposal candidate for controlled copy/edit use, with the neural fallback retained.
Its response is advisory and is not a probability of quality equivalence or future speedup.
The 255-proposal setting is not selected merely for its larger brief peak, and lookup-only is not recommended as a general replacement.

A concrete central-configuration diff is staged at `.marathon/diagnostics/lookup-runtime-build-20260920/production-candidate-NOT-APPLIED.patch`.
It changes only the Marathon anchor image and ngram proposal length, keeping the neural maximum at six and all other production settings intact.
The diff affects the shared Marathon anchor and is not applied; the configuration watcher can interrupt active workers on reload.
The regular RTX 3090 workers have not been benchmarked with this candidate, so the reported rates belong to the RTX 3090 Ti.
Production conversations, saved production slots, central configuration, and power limits remain untouched.

## Artifacts

- `scripts/evals/speculation_cost_probe.py`: worker lease, isolation, copy timing, and diagnostic runtime overlay.
- `scripts/evals/speculation_quality_cases.py`: synthetic fixtures and deterministic quality checks.
- `.marathon/diagnostics/speculation-mixed{15,31,63,127,255}-20260920`: full synthetic streams and copy summaries.
- `.marathon/diagnostics/quality-{baseline,lookup15,mixed15,mixed127}-20260920`: quality requests, responses, and summaries.
- `.marathon/diagnostics/quality-code-clarified-{baseline,lookup}-20260920`: clarified coding reruns.
