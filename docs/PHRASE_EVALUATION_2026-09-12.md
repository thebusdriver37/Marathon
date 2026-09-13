# Marathon phrase evaluation review, 2026-09-12

## Conclusion

The saved audit is real and its recorded measurements are reproducible from the retained evidence.
It does **not** establish that `Minimize thinking.` or related instructions have no effect.
The defensible conclusion is: **these experiments found no reproducible overall winner for correctness or reasoning efficiency under this Marathon configuration.**
Reasoning volume varies substantially with the task, but the experiments do not establish that task choice alone determines the effect of the phrase.
Even the same tasks show opposite aggregate directions between batches.
There is insufficient evidence here to recommend changing the production phrase, or to claim that keeping it is optimal.

This review reran saved-workspace checks and analyzed existing sessions; it did not generate new model trials or modify the production prompt or evaluator.

## What was verified

- All 360 per-trial `result.json` files agree with their batch `results.json` records.
- All 360 saved task prompts match their batch's frozen `evaluator.py` cases.
- All saved variant prompt hashes match the manifests, and all 360 session metadata records contain the intended base instructions.
- All 3,218 recorded normalized router requests have the expected variant instruction hash.
- All 2,858 saved reasoning items decode successfully, with no empty decoded items; independently recomputed word, character, and block counts exactly match every trial's recorded values.
- Per-response output usage sums match all recorded output-token totals.
- Running each frozen evaluator's oracle against its saved workspace reproduces all 360 recorded oracle outcomes.
- Protected-file checks reproduce all recorded outcomes; additionally, all 40 `deep_trace` workspaces preserve every original file other than the intended `pipeline/stage_21.py`.
- All 360 sessions record medium reasoning and the same model/profile identifiers: `qwen3.8-27b-iq4-xs`, `one-gpu-196k-uncensored`, with configured temperature 1.0.

These checks establish consistency of retained evidence, not independent attestation of the historical server binaries or model weights.
The manifests record a Git HEAD and the evaluator is copied into each batch, but this is not a complete immutable snapshot of every runtime dependency or uncommitted change.

## Experimental design

Both batches compare eight variants, changing only the target sentence in the saved base instructions, with the deletion variant also removing its preceding blank line.
Baseline contains `Minimize thinking.` and `no_minimize` removes it.

| Variant | Replacement |
| --- | --- |
| `ph_reasoning` | Minimize reasoning. |
| `ph_brief` | Think briefly. |
| `ph_direct` | Be direct. Use the shortest reasoning that produces a correct answer. |
| `ph_avoid` | Avoid unnecessary thinking. |
| `ph_efficient` | Reason only as much as the task requires. |
| `ph_terse` | Spend thinking tokens only where the task is genuinely hard. |

V1 has five tasks, three repeats per variant/task, and 120 trials: patch, audit-only, retry, TTL cache, and ledger.
V2 has six tasks, five repeats per variant/task, and 240 trials: patch, retry, TTL cache, ledger, pipeline tracing, and document-guided slug implementation.
V1 uses a 300-second timeout and V2 uses 420 seconds; none of the recorded failures is a timeout.
Both use real Marathon CLI sessions, isolated homes/workspaces, private `/tmp` mounts, three concurrent workers, and a fixed shuffled job order.
The job-order seed is not a shared model-generation seed, and repeat numbers do not identify matched random draws.

The router records 2,498 budget selections with a native 2,048-token thinking cap and 720 unrestricted selections, all with an 8,192-token maximum output setting.
Thus, this tests wording within an existing runtime reasoning policy, not unrestricted reasoning on every model response.
The cap can limit how much a wording change is able to express itself, although this review does not establish how often it was reached.

## Recorded results

Reasoning words are whitespace-separated words in decoded reasoning text, not tokenizer-measured reasoning tokens or a measure of reasoning quality.
Output tokens also include generated tool arguments and final answers.
Latency is end-to-end trial time and is sensitive to worker assignment, cache state, startup, and concurrent load.

| Variant | V1 pass | V1 mean reasoning words | V1 mean seconds | V2 pass | V2 mean reasoning words | V2 mean seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 15/15 | 1,312.0 | 68.8 | 29/30 | 1,104.9 | 67.3 |
| No minimize | 14/15 | 764.3 | 48.3 | 28/30 | 1,431.3 | 77.3 |
| Minimize reasoning | 15/15 | 950.0 | 57.3 | 29/30 | 1,072.1 | 65.7 |
| Think briefly | 15/15 | 1,272.2 | 68.8 | 29/30 | 1,172.1 | 67.4 |
| Be direct | 15/15 | 1,145.0 | 61.8 | 29/30 | 1,213.8 | 72.6 |
| Avoid unnecessary thinking | 15/15 | 1,028.5 | 62.2 | 28/30 | 1,098.2 | 63.9 |
| Reason as required | 15/15 | 959.2 | 56.5 | 25/30 | 1,601.7 | 84.1 |
| Spend tokens on hard tasks | 15/15 | 1,123.4 | 60.4 | 28/30 | 1,381.0 | 74.5 |

The recorded total is **344/360**, comprising 119/120 in V1 and 225/240 in V2.
These are the original grading results, not corrected estimates of general task success.

Removing the sentence reduces mean reasoning words by 41.7% in V1 and increases them by 29.5% in V2.
This reversal is not just an artifact of changing the task mix: restricting both batches to patch, retry, TTL cache, and ledger gives 1,619.8 versus 916.4 words in V1, and 1,192.2 versus 1,629.5 in V2, baseline versus removal respectively.

| Task | V1 baseline words | V1 removal words | V2 baseline words | V2 removal words |
| --- | ---: | ---: | ---: | ---: |
| Patch | 182 | 246 | 54 | 144 |
| Retry | 1,502 | 1,165 | 1,043 | 1,417 |
| TTL cache | 1,788 | 980 | 1,656 | 2,823 |
| Ledger | 3,007 | 1,275 | 2,016 | 2,134 |

Patch shows the same direction in both batches, with less reasoning under baseline, but its absolute cost is small and each cell has only three or five runs.
Retry, TTL cache, and ledger reverse direction.
This is evidence against both a universal direction and a confident claim of zero effect.

As an exploratory uncertainty check, 50,000 bootstrap resamples within each variant/task cell give 95% percentile intervals for removal minus baseline mean reasoning words of approximately [-827, -280] in V1 and [+26, +619] in V2.
These resample repeats while keeping the observed task mix fixed, using NumPy's default RNG with seed 731 and processing batches, variants, and cases in sorted order.
They do not account for worker/time clustering, selection among many comparisons, or uncertainty over new tasks, and small cells make them fragile.
They illustrate why simply treating the observed changes as zero is unjustified; they do not establish a stable causal effect across batches.
No equivalence margin was specified, and no equivalence test demonstrates that any effect is too small to matter.

## Grading findings

All 15 V2 failures occur on `doc_buried`.
The other five tasks pass 200/200 across all variants, which creates a correctness ceiling rather than demonstrating equal reliability.

Fourteen slug failures occur on the final accent-handling assertion.
The specification says to NFKD-normalize before casefolding "so accents are dropped," but its numbered algorithm replaces all non-alphanumeric runs with dashes and never explicitly removes combining marks.
NFKD decomposes accented characters into a base character and combining marks; it does not itself delete the marks.
Following the numbered operations can therefore insert internal dashes, whereas the oracle expects `Ünïcödé 3.14` to become `unicode-3-14`.
Several failed final answers explicitly identify this ambiguity and explain their interpretation.
This case mixes specification interpretation with implementation correctness, so its pass-rate ranking is weak evidence that a phrase helps or hurts coding.

The remaining slug failure, `doc_buried-ph_efficient-r3`, is an unambiguous separator-collapse bug: `re.sub(r"[^\w]+|_", "-", text)` replaces consecutive underscores separately, so `--x__y--` becomes `x--y` instead of `x-y`.
Do not relabel all 15 failures as grading mistakes.
Do not selectively forgive only particular variants; retain the original scores and use exclusion of the entire ambiguous case as a sensitivity analysis.

V1's only failure, `ledger-no_minimize-r1`, is an incomplete task with malformed tool-call markup in the final answer and an oracle failure, despite a zero CLI exit code.
This is a real end-to-end failure, but one observation does not establish that removing the sentence causes it.

The evaluator also does not enforce every process requirement in the task text, including reproduction before editing and all requested final reporting.
For `deep_trace`, its protected list omits the 29 other stages even though the task prohibits changing them; this review's extra comparison found no actual violations in the saved runs.
The pipeline's output directly reveals the duplicated stage number, and the authoritative document is explicitly named `SPEC-AUTHORITATIVE.md` while decoys are marked archived.
These cases therefore do not establish performance on genuinely difficult long-context search.

## Follow-up needed for a stronger conclusion

Define separately what counts as a worthwhile correctness improvement, reasoning reduction, and latency reduction before collecting more trials.
For the primary comparison, use baseline versus sentence removal, clearer and more diverse tasks, more independent repeats, and worker/time-balanced randomized scheduling.
Keep the task set fixed across confirmation batches and record runtime versions, effective sampling settings, and reasoning-cap behavior.
Clarify the slug contract with an explicit combining-mark removal step and an example before rerunning it as a new version; preserve these original results.
Use predeclared practical equivalence bounds if the intended claim is that the sentence has no meaningful effect.
Alternative phrasings can remain exploratory until a finding reproduces in a separate confirmation batch.

## Evidence and portability

Original on-disk evidence remains under these Git-ignored directories:

- `.marathon/diagnostics/prompt-audit-20260912-phrase-sweep/`
- `.marathon/diagnostics/prompt-audit-20260912-phrase-sweep-v2/`

Each batch includes its frozen evaluator, manifest, exact prompt variants, aggregate results, and individual workspaces, final answers, task prompts, verification logs, rollouts, and runtime traces.
The Git HEAD in both manifests is `b9d505dc6d12498be038e2773175489df2d366ae`.

[The accompanying CSV](PHRASE_EVALUATION_2026-09-12.csv) contains all 360 trial rows with metrics, variant prompt hashes, and SHA-256 hashes of the original per-trial result files.
It makes the descriptive tables reviewable without copying the large transcripts into Git.
It is a portable summary, not a replacement for the raw evidence needed to independently audit prompts, generated code, and scoring.
The document and CSV were created during this review and are left uncommitted.
