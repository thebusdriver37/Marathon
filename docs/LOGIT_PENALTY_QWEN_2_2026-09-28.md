# Logit-penalty experiment 2 (2026-09-28)

Follow-up to `docs/LOGIT_PENALTY_QWEN_2026-09-28.md`. Two stages: a bounded
request-level screen on the promoted Swift/uncensored Qwen 3.8 27B IQ4_XS merge,
then held-out validation through the real `marathon exec` CLI. Protocol:
`.marathon/diagnostics/logit-penalty-2b-20260928/protocol.md`.

## Bottom line

- Recommendation: keep the current no-bias default. `broad2-v2` is promising but
  unproven, and is the only candidate worth an optional, reversible user trial
  (`.marathon/diagnostics/logit-penalty-2b-20260928/trial/README.md`).
- Primary held-out set (blocks B, C, E; 22 matched pairs per candidate, 21 after
  excluding infrastructure-affected runs): `broad2-v2` response time -16.8%
  [-43.1, +26.6] on all pairs and -29.3% [-47.5, -7.8] with the one
  infrastructure-affected pair removed; output tokens -10.8% [-36.7, +27.1] and
  -19.2% [-41.5, +9.3]. Neither token interval excludes zero.
- The aggregate gains ride on one dominant pair. Dropping `marathon-b/ledger-r1`
  turns `broad2-v2` into tokens +2.5%, time -3.2%, i.e. noise level. So the token
  saving is not established, and the time saving is only weakly established.
- The one domain with a consistent direction is multi-file code work with scripted
  requirement changes and edge cases (median paired -31.7% tokens, -29.0% time over
  4 clean pairs). No other domain is settled; signs flip between blocks.
- `actually-2` (4 IDs) showed no reliable benefit and is not recommended.
- Quality: no detected difference. After the two documented instrument corrections
  every arm passes 10/10 in block B, 2/2 in block C and 10/10 or 9/10 in block E;
  the single corrected failure (`marathon-e/ledger-r2-broad-2`) is an
  infrastructure-flagged turn. The graders and sample size establish
  non-detection, not equivalence.
- This is not a claimed Marathon harness speedup, and no production default
  changed (llama-swap config sha `d14f3410e9ccc88e`, user catalog sha
  `dcd8936202da104e`, both identical before and after every stage).

## Audit of the three open points

1. Token-ID count. `broad2-v2` is 85 IDs at -2, not the prior 49; it is a
   different configuration that happens to share a name. Intersection 47 IDs,
   prior-only IDs `66073` (" alternatively") and `84485` (" hmm"), and 38 new IDs
   (mostly capitalized and no-leading-space variants plus words that were previously
   dropped). Both maps use strength -2 everywhere. Resolution rule is unchanged: a
   word is kept only when `/tokenize` with `add_special: false` yields exactly one
   ID and `/detokenize` returns the word with or without one leading space, so no
   multi-token fragment is ever penalized. 15 requested words stayed excluded as
   multi-token: `nonetheless, nevertheless, regardless, anyway, uncertain, unsure,
   reconsider, rethink, backtrack, recheck, revisit, doubt, confused, mistake`.
   Receipt: `screen-a/token-map.json`.
2. What "time" means. Stage A `wall_s` is the sum of per-request
   `time.monotonic` deltas around the HTTP calls, so it covers decoding only and
   excludes harness work. Stage B/C/D/E `wall_s` is the sum of per-turn wall time of
   the whole `marathon exec` process, so it is end-to-end for the turn and includes
   tool execution inside the agent loop, in-CLI retries and streaming. It excludes
   only ~1 s of process startup and lease handling between turns. Absolute seconds
   are comparable only inside a matched pair: blocks B/C ran 2 concurrent instances
   and blocks D/E ran 3, so cross-block totals differ by contention, not by model.
   Block E adds a per-request log from the same shim (110, 108 and 111
   `/v1/responses` calls for baseline, `broad2-v2` and `actually-2`), which is the
   closest thing to "everything you wait for" that this instrument produces.
3. Grading counts. There is one consistent story: per arm, block B has 10 runs
   (frozen 6/10, corrected 10/10), block C has 2 runs (frozen and corrected 2/2),
   block D has 10 runs (frozen 3/10 baseline, 6/10 `broad2-v2`, 4/10 `actually-2`;
   corrected 7/10, 10/10, 8/10) and block E has 10 runs (frozen 6/10, 5/10, 6/10;
   corrected 10/10, 9/10, 10/10). The pooled "8/12" figure is blocks B+C frozen
   (6 + 2) and the corrected "12/12" is the same 12 runs re-scored after the two
   instrument corrections below. Block D failures concentrate in turns with a
   nonzero exit, i.e. instrument events. Both corrections are applied to all arms
   identically from retained receipts, with the finalist maps frozen.

## Instrument corrections (documented, applied identically to all arms)

- `prose`: the frozen pass graded the final chat message, so every run failed the
  word and bullet count. Scoring the requested artifact `release.md` gives 6/6 ok
  in each of blocks B and E, with all required tokens and no em dash.
- `research`: `notes/delta.json` ships in the initial workspace, so the authoritative
  ttl is 240 from turn 1, not the frozen 300. With the corrected expectation all
  runs pass every turn, including `disagreeing_sources: 3` and non-empty
  uncertainty lists.
- Receipts: `marathon-b/regrade.json`, `marathon-e/regrade.json`, and the
  pre-run grader self-test against correct and deliberately broken references in
  `marathon-*/grader-self-check.json`.

## Candidate configurations

Scope: one per-request JSON `logit_bias` map on every request of the session,
covering reasoning and final output alike. No prompt classification, no
task-dependent routing, no profile switching.

- `broad2-v2`: 85 IDs at -2.
  `269 or`, `466  or`, `694  but`, `815 error`, `1362  could`, `1412  error`,
  `1921 But`, `1990  still`, `2086  different`, `2126 Or`, `2361  another`,
  `2441  Or`, `2493  might`, `2892  either`, `3222  hold`, `3315  whether`,
  `3384  though`, `3404  actually`, `3482  yet`, `3655  wait`, `3850 But`,
  `4213  However`, `4370  instead`, `4387 though`, `4598  rather`, `4611  however`,
  `4808  wrong`, `5752  otherwise`, `6084 hold`, `6970  maybe`, `7014  unless`,
  `7643  although`, `7834 but`, `8106  perhaps`, `10179  doubt`, `10380  Maybe`,
  `10451  possibly`, `10883 However`, `11158 wait`, `11746  Instead`,
  `13264  anyway`, `13428  Wait`, `13784 Wait`, `14673  incorrect`,
  `15029  regardless`, `16036  mistake`, `20734 Maybe`, `21143  confused`,
  `21979  retry`, `27167 could`, `29781 Instead`, `32645  Actually`, `33713 wrong`,
  `33955  uncertain`, `34946 although`, `35542 maybe`, `35999  nevertheless`,
  `36563  nonetheless`, `37201  Alternatively`, `37488 unless`, `37781  reconsider`,
  `40576 another`, `41484  unsure`, `41933 still`, `43355 retry`, `43468 might`,
  `44846 yet`, `47132 whether`, `48219 either`, `50821 Actually`, `51343 possibly`,
  `57913 otherwise`, `59869 incorrect`, `61501 different`, `62586  revisit`,
  `62643 instead`, `63068 perhaps`, `70546 actually`, `70714 rather`,
  `73071  rethink`, `77264 Hmm`, `85152  Hmm`, `88842 Alternatively`,
  `94459 however`, `95500  backtrack`.
  Machine-readable: `trial/broad2-v2-bias.json`, identical to `finalists.json`.
- `actually-2`: 4 IDs at -2: `3404` " actually", `32645` " Actually", `50821`
  "Actually", `70546` "actually".
- `baseline`: empty map (zero penalty), the reference arm.

## Stage A: direct API screen (not the app harness)

Runner: `scripts/evals/logit_penalty_screen2.py`. Receipts:
`.marathon/diagnostics/logit-penalty-2b-20260928/screen-a/`. Private copies of the
promoted worker commands on GPUs 1-3, 65,536-token capacity, same model, drafter,
KV types, batch, threads and speculation as the 2026-09-28 controlled run.
Temperature 0.6, top-p 0.95, top-k 20, min-p 0, repeat-penalty 1.0, xhigh
reasoning, `cache_prompt: false`, 8,192-token limit, 2 seeds (17, 29), 13 dev
cases, 12 configurations, 312 case-runs, no infrastructure errors. Bias application
was smoke-verified per worker on `/v1/chat/completions` and `/v1/responses`.

| Arm | Output tokens | Case-cluster 95% | Response time | Case-cluster 95% | Passes |
| --- | ---: | --- | ---: | --- | ---: |
| baseline | 23,049 | - | 295.6 s | - | 18/26 |
| broad2-v2 | 16,904 [-26.7%] | [-32.7, -9.1] | 207.2 s [-29.9%] | [-36.7, -11.9] | 18/26 |
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

- Only `broad2-v2` had intervals excluding zero on both metrics; `actually-2`
  reached it on time only, with its token interval ending exactly at zero.
- Groups are hypotheses, not mechanisms: the 14-word hedge set at -2 scored worse
  than its `actually` subset alone (+8.8% versus -17.6%), and -3 was worse than -2
  and -1. Strengths are not monotone and groups are not additive.
- Every arm had identical pass counts and zero discordant case-runs, so the screen
  measures work generated, not answer quality. Dominant-case sensitivity after
  dropping the two largest token cases: `broad2-v2` -16.3%, hedge+connector-2
  -14.8%, connector-2 -11.7%, `actually-2` -6.6%, all others within +-2.9%.
- Dropped before validation: the two group-pair arms, all single-token arms, the
  -1 and -3 strengths, and the zero-penalty duplicate role of `wait-2`.

## Stage B, C, D, E: real Marathon CLI validation

Runner: `scripts/evals/logit_penalty_marathon_eval.py`, receipts in
`.marathon/diagnostics/logit-penalty-2b-20260928/marathon-b/` (10 pairs),
`.../marathon-c/` (bounded repeat, 2 pairs), `.../marathon-d/` (10 pairs) and
`.../marathon-e/` (10 pairs).

- Interface: `marathon --instance <name> exec --json --sandbox workspace-write -c
  approval_policy="never" -c model_reasoning_effort="xhigh" -C <workspace>`, turns
  2+ through `exec resume <session-id>` so each arm keeps its own history. Normal
  tool schemas, isolated per-arm catalog and `XDG_*` homes, production pool workers
  at 196,000 context, catalog temperature 1.0, `MARATHON_MAX_OUTPUT_TOKENS=32768`,
  slot snapshots off, matched pairs on the same GPU, arm order alternated per
  repeat.
- The only runtime difference is `scripts/evals/logit_bias_shim.py`, one loopback
  forwarder that adds `logit_bias`; the baseline arm uses the same shim with `{}`.
  Bias reaches the app: `smoke2/state-r1-forced-*` forced `{"13428": 100}` and the
  CLI answered `"Wait Wait Wait ..."`.
- Five held-out families: `ledger` implement, revised requirements, introduced
  edge case; `pager` diagnose and repair with hidden executable checks; `research`
  synthesize conflicting fixed local sources; `prose` write and revise under
  retained constraints; `state` correct earlier facts and keep unrelated details
  across four turns. Fixed local source material, so no live-web confound.

Paired totals per block (all runs kept):

| Block | Pairs | Arm | Output tokens | 95% | Time | 95% |
| --- | ---: | --- | ---: | --- | ---: | --- |
| B | 10 | broad2-v2 | -13.0% | [-50.3, +47.2] | -34.6% | [-60.8, +4.3] |
| B | 10 | actually-2 | -21.4% | [-52.1, +22.8] | -7.6% | [-44.0, +63.0] |
| C | 2 | broad2-v2 | -23.2% | [-42.1, -8.8] | -20.7% | [-27.4, -14.5] |
| C | 2 | actually-2 | +23.1% | [+2.4, +50.4] | +91.1% | [-4.7, +194.8] |
| D | 10 | broad2-v2 | +46.9% | [-7.1, +122.7] | -72.2% | [-93.8, +186.2] |
| D | 10 | actually-2 | +88.6% | [-1.7, +177.4] | -64.8% | [-93.1, +244.4] |
| E | 10 | broad2-v2 | -3.5% | [-40.2, +69.4] | +16.6% | [-30.5, +112.3] |
| E | 10 | actually-2 | -1.9% | [-24.2, +34.6] | -4.9% | [-13.9, +6.6] |

Block D is instrument-degraded: 5 of its 30 runs carry a nonzero turn exit and one
2400 s timeout chain from a stale pool lease plus Marathon instance-name
collisions between overlapping waves. The runner now gives every wave its own
instance name and its own port pair, which is why block E is clean (1 flagged run).
Block D is reported but the primary pooled set is B + C + E.

Primary pooled set, blocks B + C + E, 22 pairs:

| Arm | Output tokens | 95% | Median | Improved | Time | 95% | Median | Improved |
| --- | ---: | --- | ---: | --- | ---: | --- | ---: | --- |
| broad2-v2 | -10.8% | [-36.7, +27.1] | +3.2% | 10/22 | -16.8% | [-43.1, +26.6] | -9.1% | 14/22 |
| broad2-v2 (infra excluded, 21) | -19.2% | [-41.5, +9.3] | - | 10/21 | -29.3% | [-47.5, -7.8] | - | 14/21 |
| actually-2 | -9.1% | [-31.6, +18.4] | +0.9% | 11/22 | +3.4% | [-27.6, +47.4] | -2.4% | 12/22 |
| actually-2 (infra excluded, 20) | +4.9% | [-10.1, +25.2] | - | 9/20 | +3.4% | [-17.8, +39.7] | - | 11/20 |

Dominant-outlier sensitivity, same 22-pair set:

| Arm | Drop top-1 time pair | Drop top-2 | Drop top-3 |
| --- | --- | --- | --- |
| broad2-v2 tokens | +2.5% | +2.7% | -3.7% |
| broad2-v2 time | -3.2% | -2.9% | -1.4% |
| actually-2 tokens | +4.4% | +6.2% | +6.8% |
| actually-2 time | +17.2% | +23.9% | +33.7% |

Per-task median paired change, clean pairs only (so the aggregate does not hide
task-level behaviour):

| Task | Pairs | broad2-v2 tokens | broad2-v2 time | actually-2 tokens | actually-2 time |
| --- | ---: | ---: | ---: | ---: | ---: |
| ledger | 4 | -31.7% | -29.0% | +0.0% | -10.2% |
| pager | 4-5 | +16.3% | -27.4% | +35.7% | -1.1% |
| prose | 4 | -13.8% | -10.3% | +4.2% | -10.4% |
| research | 4 | +1.9% | +7.9% | +14.9% | +11.5% |
| state | 4 | +33.0% | +12.0% | +1.7% | +2.0% |

Reading: only `ledger` is directionally consistent for `broad2-v2` on both metrics.
`pager` mixes token growth with time reduction, the small `state` task is slightly
worse for both penalties, and `research` and `prose` flip signs between blocks.
No domain is settled enough to justify routing, and none was tuned to.

Infrastructure, kept separate from model scores: block B 2 nonzero exits
(`ledger-r1-actually-2:t0` after a stale lease, `pager-r2-actually-2:t1`), 2 invalid
event lines, 0 timeouts. Block C none. Block D 5 runs with nonzero exits and 4
timeouts. Block E 1 run (`ledger-r2-broad-2`) with a nonzero exit. No lease wait
failed in any block, and the production config hash was asserted unchanged after
every single run.

## Concrete receipts

- Improvement pair: `marathon-b/ledger-r1-baseline-w...-3/` 31,662 tokens and
  347.4 s across 13 commands versus `marathon-b/ledger-r1-broad-2-w...-2/` 9,082
  tokens and 79.2 s across 8 commands; both `result.json` show `all_tests_pass:
  true` and `tests_preserved: true`. Reproduced in `marathon-e/ledger-r1-*`:
  21,339 tokens / 149.7 s versus 9,700 / 84.7 s.
- Token regression, same arm: `marathon-b/pager-r1-*`, 8,810 to 15,824 tokens with
  hidden checks passing in both.
- Second regression direction: `marathon-b/state-r2-*`, 1,109 to 2,384 tokens and
  12.7 to 18.0 s, both passing.
- Turn-level detail, commands, prompts, stderr and Codex event streams are in every
  trial directory (`command-N.json`, `events-N.jsonl`, `answer-N.md`,
  `turn-N.json`, `result.json`); block E also has per-request shim logs
  (`requests-*.jsonl`).

## Reproduction

```sh
cd /home/deforest/Documents/DEV/Marathon
R=.marathon/diagnostics/logit-penalty-2b-20260928
.marathon/venv/bin/python scripts/evals/logit_penalty_screen2.py --output $R/screen-a
.marathon/venv/bin/python scripts/evals/logit_penalty_screen2.py --output $R/screen-a --summarize
.marathon/venv/bin/python scripts/evals/logit_penalty_marathon_eval.py \
  --output $R/marathon-b --arms-file $R/finalists.json --repeats 2 --gpus 1,2,3
.marathon/venv/bin/python scripts/evals/logit_penalty_regrade.py --output $R/marathon-b
# one or more blocks, corrected grading applied where a regrade exists
.marathon/venv/bin/python scripts/evals/logit_penalty_pool.py \
  --blocks $R/marathon-b $R/marathon-c $R/marathon-e --output $R/pooled-bce.json
```

Each new block needs a new output directory: runners refuse to overwrite an
existing `protocol.json`. Graders self-test against correct and broken references
before any model call (`marathon-*/grader-self-check.json`).

## State after the experiment

Production defaults, weights, prompts and OMP settings are unchanged; both config
hashes matched before and after every stage. Stage A containers stopped, Stage B/C/D/E
instances stopped, shims killed, pool leases released, and the pool workers left to
their normal idle TTL. Retained: `protocol.md`, the four runner and analysis
scripts, `finalists.json`, per-stage `summary.json`, `pooled-analysis.json`,
`pooled-bce.json`, `pooled-bcde.json`, `pooled-e.json`, the two `regrade.json`
files, `trial/` and all case receipts.
