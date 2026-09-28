# Logit-penalty experiment 2 (2026-09-28)

Follow-up to `docs/LOGIT_PENALTY_QWEN_2026-09-28.md`. Two stages: a bounded
request-level screen on the promoted Swift/uncensored Qwen 3.8 27B IQ4_XS merge,
then held-out validation through the real `marathon exec` CLI. Protocol: `.marathon/diagnostics/logit-penalty-2b-20260928/protocol.md`.

## Bottom line

- Recommendation: keep the current no-bias defaults. broad -2 is promising but
  unproven; it is the only candidate worth an optional user trial.
- Pooled over 12 matched held-out pairs, broad -2 cut summed response time by
  32.5% (paired bootstrap 95% [-56.2, -2.1], 8/12 pairs improved) while output
  tokens moved -14.9% with an interval that includes no benefit [-45.8, +28.0].
- Dropping the single dominant pair reverses the token result (+7.2%), so the
  token saving is not established. The narrow `actually`-only map showed no
  reliable benefit and is not recommended.
- Quality was identical in every arm: 6/10 runs passed the frozen graders in all
  three arms, 0 paired wins and 0 paired losses, and 10/10 passed after the two
  documented instrument corrections. No arm produced a wrong-but-shorter answer.
- This is not a claimed Marathon harness speedup, and no production default
  changed.

## Candidate configurations

Both maps were resolved with the actual model tokenizer through `/tokenize`
(`add_special: false`) and confirmed by a `/detokenize` round trip. Only pieces
whose decoded string equals the word, with or without one leading space, are
penalized. 15 requested words are multi-token for this merge and were dropped
rather than penalized as fragments; see `screen-a/token-map.json`.

Scope of application: one per-request JSON `logit_bias` map on every request of
the session, covering both reasoning and final output. No prompt classification
and no task-dependent routing.

- `broad-2`: 85 token IDs at -2. Hedge, connector and meta-verb groups:
  `269, 466, 694, 815, 1362, 1412, 1921, 1990, 2086, 2126, 2361, 2441, 2493,`
  `2892, 3222, 3315, 3384, 3404, 3482, 3655, 3850, 4213, 4370, 4387, 4598,`
  `4611, 4808, 5752, 6084, 6970, 7014, 7643, 7834, 8106, 10179, 10380, 10451,`
  `10883, 11158, 11746, 13264, 13428, 13784, 14673, 15029, 16036, 20734, 21143,`
  `21979, 27167, 29781, 32645, 33713, 33955, 34946, 35542, 35999, 36563, 37201,`
  `37488, 37781, 40576, 41484, 41933, 43355, 43468, 44846, 47132, 48219, 50821,`
  `51343, 57913, 59869, 61501, 62586, 62643, 63068, 70546, 70714, 73071, 77264,`
  `85152, 88842, 94459, 95500` (words: or, but, error, could, But, still,
  different, Or, another, might, either, hold, whether, though, actually, yet,
  wait, However, instead, rather, however, wrong, otherwise, maybe, unless,
  although, perhaps, doubt, Maybe, possibly, wait, Instead, anyway, Wait,
  incorrect, regardless, mistake, confused, retry, Hmm, uncertain, nevertheless,
  nonetheless, Alternatively, unsure, reconsider, rethink, backtrack, revisit).
- `actually-2`: 4 IDs at -2: `3404` " actually", `32645` " Actually", `50821`
  "Actually", `70546` "actually".
- `baseline`: empty map (zero penalty), the reference arm.

## Stage A: direct API screen (not the app harness)

Runner: `scripts/evals/logit_penalty_screen2.py`. Receipts: `.marathon/diagnostics/logit-penalty-2b-20260928/screen-a/`.
Private copies of the promoted worker commands on GPUs 1-3, 65,536-token
capacity, same model, drafter, KV types, batch, threads and speculation as the
2026-09-28 controlled run. Temperature 0.6, top-p 0.95, top-k 20, min-p 0,
repeat-penalty 1.0, xhigh reasoning, `cache_prompt: false`, 8,192-token limit,
2 seeds (17, 29), 13 dev cases, 12 configurations, 312 case-runs, no
infrastructure errors. Bias application was smoke-verified per worker on both
`/v1/chat/completions` and `/v1/responses` before measurement.

| Arm | Output tokens | Case-cluster 95% | Response time | Case-cluster 95% | Passes |
| --- | ---: | --- | ---: | --- | ---: |
| baseline | 23,049 | - | 295.6 s | - | 18/26 |
| broad-2 | 16,904 [-26.7%] | [-32.7, -9.1] | 207.2 s [-29.9%] | [-36.7, -11.9] | 18/26 |
| actually-2 | 18,997 [-17.6%] | [-29.6, +0.0] | 235.6 s [-20.3%] | [-33.2, -0.5] | 18/26 |
| hedge+connector-2 | 19,298 [-16.3%] | [-31.2, +10.9] | 236.5 s [-20.0%] | [-34.6, +6.8] | 18/26 |
| hedge-1 | 19,560 [-15.1%] | [-39.8, +22.1] | 244.0 s [-17.4%] | [-44.2, +25.1] | 18/26 |
| hedge+meta-2 | 20,542 [-10.9%] | [-48.4, +46.2] | 254.2 s [-14.0%] | [-52.6, +47.3] | 18/26 |
| connector-2 | 23,665 [+2.7%] | [-15.8, +22.1] | 296.9 s [+0.4%] | [-17.8, +21.1] | 18/26 |
| meta-2 | 24,084 [+4.5%] | [-11.5, +32.9] | 302.2 s [+2.2%] | [-14.4, +32.6] | 18/26 |
| hedge-2 | 25,069 [+8.8%] | [-9.7, +32.2] | 315.2 s [+6.6%] | [-11.1, +28.6] | 18/26 |
| hedge-3 | 26,937 [+16.9%] | [-22.0, +85.7] | 329.6 s [+11.5%] | [-27.0, +81.0] | 18/26 |
| hmm-2 | 28,858 [+25.2%] | [+0.0, +80.3] | 364.3 s [+23.2%] | [-3.1, +79.7] | 18/26 |
| wait-2 | 23,125 [+0.3%] | [-1.0, +2.3] | 295.5 s [-0.0%] | [-1.4, +2.4] | 18/26 |

- Only `broad-2` has intervals excluding zero on both metrics; `actually-2`
  reaches it on time only, with the token interval ending exactly at 0.
- Groups are hypotheses, not mechanisms: the 14-word hedge set at -2 scored worse
  than its `actually` subset alone (+8.8% versus -17.6%), and -3 was worse than
  -2 and -1, so strengths are not monotone and the groups are not additive.
- Every arm had identical pass counts and zero discordant case-runs, so the
  screen measures work generated, not answer quality.
- Dominant-case sensitivity, dropping the two largest token cases: broad-2
  -16.3%, hedge+connector-2 -14.8%, connector-2 -11.7%, actually-2 -6.6%, all
  others within +-2.9%.
- Discarded before validation: the two group-pair arms, all single-token arms,
  the -1 and -3 strengths, and the zero-penalty duplicate role of `wait-2`.

## Stage B and C: real Marathon CLI validation

Runner: `scripts/evals/logit_penalty_marathon_eval.py`, receipts in
`.marathon/diagnostics/logit-penalty-2b-20260928/marathon-b/` (10 pairs) and
`.../marathon-c/` (bounded repeat block, 2 pairs).

- Interface: `marathon --instance <name> exec --json --sandbox workspace-write -c
  approval_policy="never" -c model_reasoning_effort="xhigh" -C <workspace>`,
  with turns 2+ through `exec resume <session-id>` so each arm keeps its own
  history. Normal tool schemas, isolated per-arm catalog, `XDG_*` and Codex homes.
- Unmodified production inference: pool workers on GPUs 2 and 3 at 196,000
  context, catalog temperature 1.0, `MARATHON_MAX_OUTPUT_TOKENS=32768`, slot
  snapshots off, two concurrent instances, matched pairs on the same GPU.
- The only runtime difference is `scripts/evals/logit_bias_shim.py`, one loopback
  forwarder that adds `logit_bias`; the baseline arm uses the same shim with `{}`.
  End-to-end verification: `smoke2` forced arm `{"13428": 100}` made the CLI answer
  `"Wait Wait Wait ..."`, proving `/v1/responses` honors the map through the app.
- Five held-out families: `ledger` implement, then revised requirements, then an
  introduced edge case; `pager` diagnose and repair with hidden executable
  checks; `research` synthesize conflicting fixed local sources; `prose` write
  and revise under retained constraints; `state` correct earlier facts and keep
  unrelated details across four turns.

Pooled over all 12 matched pairs (`pooled-analysis.json`):

| Arm | Output tokens | Pair bootstrap 95% | Pairs improved | Response time | Pair bootstrap 95% | Pairs improved |
| --- | ---: | --- | ---: | --- | --- |
| baseline | 112,670 | - | - | 1279.4 s | - | - |
| broad-2 | 95,882 [-14.9%] | [-45.8, +28.0] | 5/12 | 863.1 s [-32.5%] | [-56.2, -2.1] | 8/12 |
| actually-2 | 97,926 [-13.1%] | [-42.4, +24.4] | 7/12 | 1372.3 s [+7.3%] | [-36.8, +78.8] | 6/12 |

- Excluding the single dominant pair (`marathon-b/ledger-r1`), broad-2 tokens
  become +7.2% and actually-2 +9.3%. The aggregate token gain is that one pair.
- Quality was identical: 8/12 runs passed the frozen graders in all three arms,
  with 0 paired wins and 0 paired losses; after the instrument corrections below
  all 12 runs per arm pass.
- The bounded repeat block reproduced the time effect for broad-2 (tokens -23.2%
  [-42.1, -8.8], time -20.7% [-27.4, -14.5], 2/2 pairs) but not the magnitude of
  `marathon-b/ledger-r1`, where the baseline itself used 31,662 tokens versus
  11,967 on the repeat. Block C for actually-2 was worse (tokens +23.1%
  [+2.4, +50.4], time +91.1% [-4.7, +194.8]).
- Per-task direction for broad-2 in block B: ledger -71.3% and +1.4% tokens,
  pager +79.6% and +59.3%, research -41.4% and +35.5%, prose +61.3% and -23.4%,
  state +9.3% and +115.0%. Sign flips between blocks mean no domain is settled.
- Infrastructure separated from model outcomes: 2 turns returned a nonzero exit
  (`ledger-r1-actually-2` turn 0 after a stale lease, `pager-r2-actually-2`
  turn 1), 2 invalid event lines total, 0 lease waits failed, 0 turn timeouts.
  Both runs still passed their tests and are kept in the denominators.

## Instrument corrections (documented, applied identically to all arms)

- `prose` graded the final chat message in the frozen pass, so all six runs
  failed on word count and bullet count. Re-scoring the requested artifact
  `release.md` gives 78, 96, 80, 103, 90 and 105 words, exactly 3 bullets, every
  required token and no em dash: 6/6 ok. See `marathon-b/regrade.json`.
- `research` shipped `notes/delta.json` in the initial workspace, so the frozen
  turn-1 expectation of 300 was wrong; the correct value is 240 from turn 1. With
  corrected expectations all six runs pass every turn, including
  `disagreeing_sources: 3` and non-empty uncertainty lists.
- Both corrections are measurement fixes on retained receipts; no arm was
  re-tuned after seeing outputs, and the finalist maps stayed frozen.

## Concrete receipts

- Improvement pair: `marathon-b/ledger-r1-baseline-w...-3/` 31,662 tokens and
  347.4 s across 13 commands versus`marathon-b/ledger-r1-broad-2-w...-2/` 9,082
  tokens and 79.2 s across 8 commands; both `result.json` files show
  `all_tests_pass: true` and `tests_preserved: true`.
- Token regression for the same arm: `marathon-b/pager-r1-*`, 8,810 to 15,824
  tokens with hidden checks passing in both.
- Turn-level detail, commands, prompts, stderr and Codex event streams are in
  every trial directory (`command-N.json`, `events-N.jsonl`, `answer-N.md`,
  `turn-N.json`, `result.json`).

## Reproduction

```sh
cd /home/deforest/Documents/DEV/Marathon
.marathon/venv/bin/python scripts/evals/logit_penalty_screen2.py \
  --output .marathon/diagnostics/logit-penalty-2b-20260928/screen-a
.marathon/venv/bin/python scripts/evals/logit_penalty_screen2.py \
  --output .marathon/diagnostics/logit-penalty-2b-20260928/screen-a --summarize
.marathon/venv/bin/python scripts/evals/logit_penalty_marathon_eval.py \
  --output .marathon/diagnostics/logit-penalty-2b-20260928/marathon-b \
  --arms-file .marathon/diagnostics/logit-penalty-2b-20260928/finalists.json --repeats 2
.marathon/venv/bin/python scripts/evals/logit_penalty_marathon_eval.py \
  --output .marathon/diagnostics/logit-penalty-2b-20260928/marathon-b --summarize
.marathon/venv/bin/python scripts/evals/logit_penalty_regrade.py \
  --output .marathon/diagnostics/logit-penalty-2b-20260928/marathon-b
```

Each run refuses to overwrite an existing `protocol.json`, so new blocks need new
output directories. Graders self-test against correct and broken references
before any model call; that receipt is `marathon-b/grader-self-check.json`.

## State after the experiment

Production defaults, weights, prompts and OMP settings are unchanged, and the
llama-swap config hash matched before and after every stage. The Stage A
containers were stopped, the Stage B and C instances were stopped, worker leases
were released, and GPUs 1-3 returned to 15 MiB. Retained: `protocol.md`, the two
runner scripts, the shim, the regrade script,`finalists.json`, per-stage
`summary.json`, `pooled-analysis.json`, `regrade.json`, and all case receipts.
