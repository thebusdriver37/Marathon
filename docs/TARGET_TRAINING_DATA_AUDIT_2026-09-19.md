# Target training data reuse audit

## Decision

Reuse the synthetic corpus as source material, but do not train the target on the current transcripts unchanged.
No target fine-tuning, model generation, GPU allocation, or deployment was performed in this audit.

## Scope and evidence

The audit scanned `rollout.json` records under `.marathon/drafter-training/corpus-pilot`.
The final report and review files are in `.marathon/drafter-training/target-reuse-audit-20260919-reviewed/`.
The earlier `target-reuse-audit-20260919/` output is a preliminary inventory superseded by the reviewed output; it incorrectly classified missing tool schemas as unknown tools.
The reusable audit script is `.marathon/drafter-training/corpus-pilot/audit_target_reuse.py`.
Seven small admission and deduplication checks passed, covering malformed JSON, patch envelopes, argument types, call linkage, missing schemas, and normalized call IDs.

| Inventory | Count |
| --- | ---: |
| Raw rollout records | 467 |
| Unique conversations after normalized exact deduplication | 293 |
| Unique conversations with recorded passing task checks | 46 |
| Unique conversations with recorded failing task checks | 23 |
| Unique conversations with task success unevaluated | 224 |
| Training conversations passing initial admission filters | 6 |
| Assistant turns in those six conversations | 230 |
| Reserved conversations passing initial admission filters | 2 |

The six training conversations are six distinct recorded families, not 230 independent tasks.
They cover tool-error recovery, configuration, frontend, documentation, repository investigation, and SQL.
The two reserved conversations are test-engineering tasks from different recorded families, so they do not constitute a representative evaluation set.
All families observed in validation, eval, or test remain excluded from training candidates.
Semantic near-duplicate checking beyond family labels is still required.

The older `cases.json`, `tool-cases.json`, `tool-long-cases.json`, `agent-state-cases.json`, and `long-response-cases.json` files contain prompt-oriented drafter fixtures rather than complete verified target demonstrations.
They may supply prompts for future evaluation or generation, but they are not counted as successful target answers here.
Feature tensor volume is not a measure of independent target-training examples.
This is an audit of the located drafter corpus, not a claim that every synthetic dataset elsewhere on the machine was searched.

## Why recorded success is insufficient

The large fastpilot corpus explicitly uses `quality_contract: protocol_only` and `oracle_pass: null`.
That was appropriate for capturing what the target predicts, but does not establish correct task behavior.
Older passing records sometimes lack tool schemas, contain malformed calls, or are shortened diagnostic copies whose trace no longer matches the visible conversation.

The deep-generation harness checks the original task oracle at the end, while dynamically generated follow-up requests generally have no independent per-turn oracle.
A passing original oracle therefore does not certify every later request, final claim, or intermediate action.
Even the six admitted candidates contain tool failures requiring review and selective supervision before training.
Recorded task checks were not independently rerun during this audit.

## Repair queue

The train-only review queue contains 152 malformed patch-envelope flags and four invalid argument-JSON flags, plus 571 tool-result failure flags.
These are events, may overlap, and are not 727 distinct errors or repaired examples.
A nonzero test exit may be an intentional and useful diagnostic action.
The queue records the source conversation and message index rather than inventing corrections or successful tool outputs.

The synthetic harness describes `apply_patch` only as applying a structured patch, with a string `patch` argument.
Its parser nevertheless requires `*** Begin Patch` and `*** End Patch` envelopes.
Observed failures include raw source code sent as the patch string, followed by actual patch rejection.
This makes incomplete tool-format instruction a plausible contributing factor, not a proven production-model defect.
Before generating repaired training examples, compare behavior with the actual production tool instructions and parser on a bounded set of these contexts.
Do not claim a target-training opportunity merely from errors caused by an underspecified synthetic interface.

## Review outputs and next decision

`inventory.jsonl` contains provenance, exclusion reasons, and known split reservations.
`review-candidates.jsonl` preserves six candidate conversations and their original tool schemas without altering the history.
`reserved-review.jsonl` preserves eligible existing holdout material separately.
`repair-review.jsonl` indexes uncorrected failures for contextual inspection and executable repair validation.
None of these files is certified as trainer-ready, and the summary explicitly records `ready_for_training: false`.

The next justified step is a small comparison of existing versus explicit tool-format instructions on malformed-call contexts, followed by sandboxed execution of any proposed corrections.
For prevention examples, regenerate results and dependent continuation after correcting a call.
For recovery examples, supervise the verified successful recovery while masking the failed action.
Add independent checks for follow-up requirements, preserve diverse correct behavior, and collect fresh evaluation tasks before choosing a target-training budget.
The earlier estimate of 500 to 2,000 verified examples was a pilot-sizing suggestion, not an established minimum or evidence that this corpus already supplies them.

## Completed bounded patch-instruction probe

Results and full responses are saved under `.marathon/drafter-training/target-patch-instructions-20260919/`.
The reusable runner is `.marathon/drafter-training/corpus-pilot/probe_patch_instructions.py`.
No target training or production changes were made, and the isolated GPU worker was stopped afterward.

The probe selected 20 distinct train-only families across ten domains from the frozen pair-scale corpus.
Each selected context previously produced an invalid patch envelope immediately after one read-only file-inspection command.
Initial task files therefore reconstruct the prefix workspace without replaying mutations or using final answers.
Task requirements were checked against the recorded user prompts.
The two arms used identical four-message prefixes and tool schemas, changing only the patch tool description to explain the format in the explicit arm.
The local IQ4_XS target and V1 speculation settings matched the previous production-quality comparison, with temperature zero, seed 424242, alternating arm order, a 2,048-token cap, and one unforced response per arm.
All 40 responses completed without token-limit truncation.

| One-response outcome | Original description | Explicit description |
| --- | ---: | ---: |
| Responses emitting a patch | 19/20 | 20/20 |
| Patches accepted by the repository's real patch executable | 0/20 | 4/20 |
| Additional accepted patches after adding only missing outer markers | 0 | 13 |

The last row is a separate deterministic repair diagnostic, not unassisted model success.
It adds `*** Begin Patch` and `*** End Patch` only to outputs already starting with a recognized file-operation header and containing neither envelope marker.
With that diagnostic repair, 17/20 explicit-arm responses yield applicable patches, but this does not establish 17 correct task solutions.
This failure-enriched sample cannot estimate general Marathon failure rates, production instruction quality, speed gains, or expected fine-tuning gains.
The synthetic JSON tool interface was retained; this was not a full Marathon-agent comparison.

### Confirmed harness defect

The synthetic harness removes lines by stripped-text matching and inserts replacements after the first surviving line, rather than applying context-positioned hunks.
It also lacks normal add-file support.
A minimal five-line reproduction confirmed that both parsers report success while producing different line ordering.
All four directly accepted explicit-arm patches produced different file contents under the harness versus the real executable.
Basic checks on three of those fixtures passed after real patch application and failed after harness application: shell configuration loading, C ring-buffer output, and Python record normalization.
The fourth accepted case was not behaviorally checked.
These are narrow visible-fixture checks, not comprehensive independent task oracles.
The shell check sources the script and inspects variables in that shell; the original request's implication that executing a child script changes its parent's environment is itself faulty.

### Decision after probe

Do not train on these transcripts unchanged or interpret their failures as clean evidence of a model reasoning deficit.
There is a concrete patch-format problem, but the data generator also introduces execution errors that contaminate subsequent recovery conversations.
The next justified implementation is to replace the synthetic patch implementation with the real parser and add parity regression tests before generating or certifying repaired examples.
Then regenerate results and dependent continuations under the corrected tool contract and verify actual task requirements.
The probe did not implement that harness change, deploy automatic marker insertion, or certify any examples as training-ready.
