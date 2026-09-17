# DFlash2 evaluation reconciliation, 2026-09-16

## Decision

Keep large synthetic corpus generation and expensive feature extraction paused.
Do not rebuild the evaluator around the handoff's teacher-forcing diagnosis: the inspected code and fresh serving experiments contradict that diagnosis.
The existing rank-32 checkpoint demonstrably improves serving on held-out structured-agent prompts, while its aggregate acceptance is unchanged on the separate eight-workload benchmark.
The next investment should be a representative serving evaluation set drawn from actual Marathon usage, with inexpensive request traces collected before feature extraction.
Use that evaluation set to justify and target a bounded training pilot before scaling the corpus.

This revises the initial recommendation to repair a presumed broken evaluator before collecting representative data.
There is no demonstrated universal training-to-serving failure to repair.
There is a demonstrated generalization limitation and a misleading comparison of metric units and workloads.

## Fresh end-to-end evidence

Both experiments use GPU 3, the production target and draft KV formats, a 196,000-token context, six speculative proposals, greedy sampling, and stock/candidate/candidate/stock ordering.
The experiment lease checks for conflicting registered workers and yields the GPU if one appears.
The native server, `libllama.so`, and `libggml-cuda.so` were verified byte-for-byte against deployment image `sha256:765a84864664d953cc274adb0fdb161b793269857e4602d228460a790919cea8`.
The copied candidate and the deployed candidate have the same SHA-256, `e096aa09c26d5096b63b1a4d0400258819980b16250ef5c5d08fcf830e6bb6a6`.
The stock SHA-256 is `18a380efc9b7ed8d88677fc895f5c11ae170653434ee378f7348f715c14d0594`.

### Held-out structured-agent prompts

The complete rerun contains 80 measured requests across 20 existing test prompts, plus four excluded warmups.

| Measurement | Stock | Rank-32 candidate |
| --- | ---: | ---: |
| Measured requests | 40 | 40 |
| Accepted / proposed tokens | 2,420 / 3,918 | 2,564 / 3,052 |
| Acceptance rate | 61.7662% | 84.0105% |
| Generated tokens / decode second | 100.1258 | 128.4388 |
| Total request time | 142.0455 s | 134.2388 s |

Acceptance improves by 22.24 percentage points, decode throughput by 28.28%, and aggregate request time by 5.50%.
The acceptance counts exactly reproduce the earlier experiment recorded in `DRAFTER_FINETUNING_2026-09-14.md`.
Nineteen of twenty prompts have identical output hashes across all four stages.
The remaining prompt has slightly different generated output, so the aggregate is not a perfectly fixed-output timing comparison.
These are serving continuations, not new completed-agent quality tests.

### Broad workload benchmark

The separate scorecard uses the eight existing `bench_drafter.py` workloads: structured JSON, code boilerplate, tool calls, novel prose, analysis, SQL, error recovery, and verbatim repetition.
The benchmark preserves full responses and measures server counter deltas around every request.
Final aggregate results are recorded in `.marathon/drafter-training/reconciliation-broad-abba/summary.json`.
The completed ABBA run contains 32 measured requests and four excluded warmups.

| Measurement | Stock | Rank-32 candidate |
| --- | ---: | ---: |
| Accepted / proposed tokens | 3,074 / 6,086 | 3,074 / 6,086 |
| Acceptance rate | 50.5094% | 50.5094% |
| Accepted proposals / verification | 2.9961 | 2.9961 |
| Generated tokens / decode second | 91.0148 | 90.9202 |
| Total request time | 50.8306 s | 50.8956 s |

This is evidence of no aggregate acceptance improvement on these fixtures, not behavioral identity on all workloads.
The roughly 0.1% throughput difference does not establish a meaningful timing effect.

| Regime | Stock acceptance | Candidate acceptance |
| --- | ---: | ---: |
| Analysis | 36.58% | 34.62% |
| Code boilerplate | 79.22% | 84.66% |
| Novel prose | 20.95% | 20.42% |
| Recovery | 55.85% | 59.18% |
| SQL | 53.66% | 52.68% |
| Structured JSON | 97.14% | 99.64% |
| Tool call | 80.00% | 80.00% |
| Verbatim repetition | 96.97% | 96.97% |

Seven of eight workloads produce identical output hashes across all stages after excluding generated tool-call IDs.
Analysis produces different text, which limits the interpretation of its acceptance comparison.
These prompts have output caps and are not a completed-task correctness evaluation or an estimate weighted by real deployment frequency.

## Why the handoff's diagnosis was wrong

1. **The block sizes already match.**
   Offline `block_size=7` includes the anchor and six proposal positions.
   Serving constructs `n_draft + 1` block tokens, so `--spec-draft-n-max 6` also produces seven total positions.
2. **The accepted-length units differ.**
   The offline helper computes `1 + sum(cumulative prefix correctness)` per valid block.
   Its stock validation score of 4.7667 therefore means 3.7667 accepted proposals per sampled anchor.
   The server's roughly 2.99 figure excludes that anchor and comes from different workloads and different anchor positions.
   Subtracting one fixes the unit mismatch but does not make the two measurements directly comparable.
3. **The reported offline serving-path score already follows predicted predecessors.**
   `CandidateSelector.greedy_path` uses the selected token as the next position's predecessor.
   `_dflash_serving_diagnostics` calls that method.
   Other teacher-forced selector diagnostics exist, but they are separate from this accepted-length metric.
4. **Future target features are masked.**
   The offline attention mask exposes captured target features strictly before the anchor and the current masked draft block.
   It excludes future target features and other sampled draft blocks.
   Serving likewise performs one masked-block forward pass and then walks the selector lattice.
5. **The handoff compared different workload distributions.**
   Training validation is based on structured synthetic agent states.
   The broad serving benchmark contains short, heterogeneous prompts.
   The fresh matched-domain serving test establishes that the trained benefit was not universally lost at deployment.

The existing dequantized-Q4 offline test scores are 3.8463 accepted proposals per sampled anchor for stock and 5.1074 for the candidate after removing the anchor.
They predict the direction of the matched-domain serving improvement.
They do not establish exact calibration: uniform offline anchors differ from the anchors visited by speculative generation, and BF16 arithmetic differs from quantized serving kernels and KV.
No token-by-token, tensor-by-tensor parity claim is made.

## Changes and verification

- `train.py` now writes an explicit evaluation contract, including execution mode, selector policy, block/proposal counts, data split, and whether it fell back to a training example.
  It additionally reports accepted proposals with the anchor removed, both overall and per case.
  Existing upstream metrics are retained for compatibility and explicitly marked as offline measurements.
- `benchmark.py` now supports the fixed broad scorecard through the same native ABBA runner used for the matched agent prompts.
  It retains full responses, checks speculative counters, records position survival counts, ignores generated tool-call IDs when comparing outputs, and shuts down its server on termination.
- `summarize.py --require-abba` rejects missing stages, duplicate cases, wrong variant order, and unmatched case sets.
  Instrumented runs must reconcile response timing counts, aggregate server counters, and per-position acceptance totals.
  The incomplete live run was correctly rejected before all stages finished.
- Two existing selector/objective tests passed using the standard-library unittest runner.
  Four additional direct checks passed for future-feature masking, draft-block isolation, predicted-predecessor selection, and anchor-inclusive accepted-length arithmetic.
  Their source hashes and results are retained in `reconciliation-contract.json`.
- A zero-step GPU evaluation completed successfully and emitted the new contract.
  Its legacy accepted length is 5.9375 and its explicitly normalized proposal count is 4.9375.
  It correctly labels the smoke fixture as a training-case fallback and produces no checkpoint.

The authoring job identified in the handoff was stopped, preserving all 50 generated JSONL records.
No corpus was deleted, no additional feature corpus was generated, no new checkpoint was trained or promoted, and the production model configuration was not changed.
The large negative-control files were left in place.
After verification, GPUs 1, 2, and 3 were idle at 24, 15, and 15 MiB respectively, and the experiment processes had exited.

## Reproduction and retained evidence

All paths below are relative to `.marathon/drafter-training/`, the existing ignored experiment workspace.

- `reconciliation-agent-abba/{results.jsonl,summary.json}` contains the complete matched-domain rerun.
- `reconciliation-broad-abba/{results.jsonl,summary.json}` contains the broad scorecard and position counters.
- Both directories retain per-stage commands, full response JSON, and server logs.
- `reconciliation-contract.json` records the executable semantic checks.
- `reconciliation-eval-smoke/` verifies the updated offline evaluation output without optimizer steps.

Run the serving experiments through `lease_run.py 3` with `venv/bin/python benchmark.py CANDIDATE --name NEW_DIRECTORY`.
For the matched domain, add `--tool-probes-only --tool-probe-file agent-state-cases.json --tool-probe-split test --tool-probe-all`.
For the broad scorecard, add `--broad-probes-only`.
Then run `python3 summarize.py NEW_DIRECTORY --require-abba`.
Use a fresh output directory for each experiment.

## Next promotion gate

Keep offline loss and acceptance as screening diagnostics.
Require repeatable acceptance improvement in the actual serving runtime on a fixed, representative workload before adopting another checkpoint.
Report per-regime regressions, position survival, decode throughput, and request latency alongside the aggregate.
Check output consistency or completed-task correctness as appropriate for the workload.
The structured-agent improvement is an established positive control for this gate; the broad null result is its generalization control.
Before spending on a larger training corpus, establish how often each regime occurs in Marathon and reserve representative requests as a held-out serving set.
