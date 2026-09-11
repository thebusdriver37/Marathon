# Marathon prompt evaluation, 2026-09-11

## Decision

Keep the production prompt, medium reasoning, and `Minimize thinking.` unchanged for now.
The completed coding comparisons do not establish a correctness advantage from shortening the prompt, removing that sentence, or replacing it with more deliberative guidance.
The strongest actionable finding is a context-compaction failure, with a specific validation gap and a targeted prompt candidate worth evaluating separately.
No production prompt, router behavior, sampler, model, GPU assignment, or broker configuration was changed in this audit.

## What was actually tested

These were real Marathon CLI sessions using the registered three-worker pool, rather than direct model calls with an approximation of Marathon's tools.
The deployed model was JonathanColetti's Qwen3.8-27B-Uncensored IQ4_XS with the existing DFlash configuration.
All trials explicitly used medium reasoning.
Each coding trial received a fresh workspace and isolated Marathon home, and its saved session metadata was checked against the exact proposed base instructions.
Independent checks ran after inference, with checks for preserved files, required patch usage, and honest reporting where applicable.
Worker identity, transcripts, prompts, outputs, verification logs, and usage were retained.

The base prompt plus Marathon runtime additions measured 4,631 tokens using the live Qwen tokenizer.
The approximately 10,000-token initial context also includes tools, user/project instructions, and skill descriptions; it is not all one system-prompt document.
The prompt governing the assistant conducting this audit is a separate layer from the prompt Marathon sends its local Qwen backend.

| Variant | Change |
| --- | --- |
| Baseline | Current base prompt plus Marathon runtime instructions |
| Lean | Compress presentation, preamble, planning, and final-answer guidance; retain `Minimize thinking.` |
| Tool aligned | Replace an outdated patch example with current-tool-schema authority guidance |
| Deliberate | Replace `Minimize thinking.` with longer reasoning and tool-use guidance |
| No minimize | Remove only `Minimize thinking.` |
| Patch precision | Tool-aligned variant plus exact-match replacement guidance |
| Handoff | Baseline plus instructions to return the compaction summary directly and preserve continuation facts |

The lean prompt measured 2,167 Qwen tokens, a reduction of approximately 53% in this base-instruction component.
The experiment tests whether that reduction helps behavior, not whether fewer prompt tokens are intrinsically preferable.

## Coding results

| Batch and variant | Passed | Mean seconds | Mean output tokens |
| --- | ---: | ---: | ---: |
| Initial screen: baseline | 15/15 | 21.92 | 1,095 |
| Initial screen: lean | 15/15 | 26.96 | 1,459 |
| Initial screen: tool aligned | 15/15 | 29.54 | 1,586 |
| Initial screen: deliberate | 15/15 | 29.16 | 1,576 |
| Harder tasks: baseline | 9/9 | 70.34 | 5,376 |
| Harder tasks: lean | 9/9 | 78.51 | 4,930 |
| Harder tasks: no minimize | 9/9 | 91.59 | 5,616 |
| Harder tasks: patch precision | 9/9 | 72.86 | 5,051 |
| Isolated patch comparison: baseline | 6/6 | 14.22 | 478 |
| Isolated patch comparison: tool aligned | 6/6 | 14.97 | 534 |
| Isolated patch comparison: patch precision | 6/6 | 14.88 | 540 |

The initial screen covered structured editing, retry boundaries and exception handling, audit-only behavior, conflicting legacy/current specifications, and recovery from an unavailable test runner.
The harder cases covered TTL/LRU cache semantics, transactional ledger rollback and caller mutation, and fixing one issue while accurately reporting a separate pre-existing test failure.
The isolated patch comparison found zero failed patch matches for baseline, two for tool-aligned guidance, and zero for patch precision.
All final edits in that comparison were correct.
An apparently cleaner tool instruction therefore did not demonstrate a practical improvement over the current prompt.

Latency and output usage are descriptive, not a statistically established ranking.
The sampler remained stochastic, sample sizes were small, workers and caches varied, and the earlier harder-task runs encountered shared host `/tmp` interference.
The clean follow-up used a private `/tmp` per trial to remove that interference.
Output-token accounting includes model generation and should not be interpreted as visible answer length or a reliable separate measurement of reasoning tokens.

The final clean comparison passed all 18 trials across retry, TTL cache, and ledger tasks.

| Clean follow-up | Passed | Mean seconds | Mean output tokens |
| --- | ---: | ---: | ---: |
| Baseline | 6/6 | 81.98 | 5,625 |
| Lean | 6/6 | 89.33 | 6,140 |
| No minimize | 6/6 | 85.22 | 5,730 |

Across the four completed coding batches, all 132 trials passed.
The clean follow-up provides no observed correctness benefit from removing `Minimize thinking.` and no token-generation saving from the lean prompt on these tasks.
Its small sample does not establish that the sentence improves every task or that medium reasoning is optimal across all models; medium was deliberately held fixed based on the user's prior experiments.

## Compaction: a concrete failure

The first compaction comparison used two independent sessions per variant, with two successive compaction cycles in each session.
Each session introduced twelve exact handoff facts through tool output and then removed the source file from the workspace before testing tool-free recall.
Baseline passed 3/4 cycles, lean passed 4/4, and patch precision passed 4/4.
These small samples do not establish that either alternative is more reliable.

In baseline session `medium-1`, the second cycle returned a plan to write a handoff file followed by textual `<tool_call>` markup instead of a useful summary.
The installed summary omitted all twelve facts, and subsequent recall returned null for every fact.
This is an observed loss of continuation state, not merely excessive wording or unattractive formatting.
The evidence is in `.marathon/diagnostics/prompt-audit-20260911-compact-baseline/medium-1/summary-1.json` and `recall-1.txt`.

The router's `_compaction_summary_error` only checks whether stripped text starts with `<tool_call>`.
A prose prefix can bypass that check, matching the observed response.
A blanket rejection of every mention of tool markup would also reject legitimate summaries about tool protocols, so a future validator change should distinguish attempted calls from quoted task data.
The appropriate follow-up is a regression test based on the actual failed session, a narrowly scoped summary validator/retry improvement, and repeated end-to-end compaction tests.
A prompt instruction can help the model choose the right behavior, but should not be the sole protection against installing a useless summary.

The follow-up compared baseline with the handoff-specific addition across three new sessions and two cycles per session for each prompt.
Both passed 6/6 cycles, with all twelve facts present in summaries and recalled without tools.
Saved session metadata confirmed the intended prompts, allowing for removal of a final trailing newline.
Across the initial and follow-up comparisons, baseline passed 9/10 cycles, lean 4/4, patch precision 4/4, and handoff 6/6.
These are repeated cycles within sessions, not independent observations of a production failure rate.
The follow-up did not reproduce the baseline failure and therefore does not demonstrate that the added instruction fixes it.
Keep the addition experimental; prioritize the concrete validator regression before promoting prompt changes.

## Evidence quality and exclusions

Pilot runs are excluded from comparative totals.
An initial private-`/tmp` wrapper omitted the device bind and failed before inference; those startup failures are infrastructure failures, not model failures.
The corrected wrapper preserved `/dev` and passed its pilot before the clean comparisons.
An initial shared skill-installation race was addressed with isolated real system-skill directories in the evaluator.
Interrupted attempts were retained under `interrupted/` and excluded rather than scored as completed trials.
The initial precedence-task preservation check was corrected to allow append-only additional tests, which the task explicitly permitted; use `screen/results-rescored.json` for that batch.

These synthetic tasks do not establish performance on large repositories, long-running autonomous jobs, cancellation/resumption, or multi-agent coordination.
Passing all cases is a ceiling for this suite, not proof that the prompts are equivalent.
A production prompt change should require a repeatable benefit on relevant failures and a broader regression set.

Raw evidence lives under `.marathon/diagnostics/prompt-audit-20260911-*` and is intentionally retained for review.
The reproducible coding evaluator is `scripts/evals/prompt_audit.py`; the existing compaction evaluator now accepts `--model-instructions-file` for isolated comparisons.
Both require explicit `--run-gpu` to run inference.
A representative clean comparison is:

```bash
.marathon/venv/bin/python scripts/evals/prompt_audit.py \
  --run-gpu --private-tmp \
  --output-dir /path/to/new-evidence-directory \
  --variants baseline lean no_minimize \
  --cases retry ttl_cache ledger --repeats 2
```

## Model documentation

The [official Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B) describes reasoning-effort controls and agent-oriented thinking continuity.
Its guidance is useful context, but does not invalidate the user's observed advantage from medium reasoning and `Minimize thinking.` on this deployed stack.
The [quantized model publisher's card](https://huggingface.co/JonathanColetti/Qwen3.8-27B-Uncensored-GGUF) describes an ablated derivative rather than the untouched official model.
Saved model cards, backend properties, prompt-token counts, and worker provenance accompany the evidence.
No inference settings were changed to match generic recommendations during this prompt comparison.

## Repository changes and validation

Only the evaluation scripts and this report were added or changed.
The compaction evaluator supports a complete prompt override and isolated skill installation; the coding evaluator preserves reproducible trial evidence and supports interrupted-batch continuation.
Python syntax checks, evaluator CLI help, the two existing end-to-end smoke guard tests, and `git diff --check` passed.
Production prompt and runtime files remained unchanged.
All three test workers were unloaded through the broker after confirming their Marathon leases were free.
