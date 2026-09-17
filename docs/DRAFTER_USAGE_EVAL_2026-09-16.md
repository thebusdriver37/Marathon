# Representative serving evaluation of the DFlash2 drafter, 2026-09-16

## Decision

Do not pursue large-scale synthetic corpus generation for the drafter on the present evidence.
The selected rank-32 checkpoint wins decisively on synthetic structured-agent prompts, and on replayed real Marathon usage it is statistically indistinguishable from the stock drafter.
The measured workloads where a drafter could still pay off are not the ones the existing corpus teaches, and the dominant cost in real usage is carried input context rather than generation.

## Phase 1, how representative the evaluation is

Filter is sessions whose `turn_context.model` matches `qwen3[.]8-27b-iq4-xs` from 2026-09 onward, which is the deployment this drafter serves.
73 conversations contribute 7,978 causally reconstructed requests, averaging 109.29 per conversation and peaking at 1,930.

- By the item immediately preceding the request: `function_call_output` 7,467, `custom_tool_call_output` 203, none 73. Real traffic is overwhelmingly continuation after a tool result.
- From 5,396 recorded usage events: input tokens p50 80,138, p90 151,561, p99 174,156, and output tokens p50 457, p90 2,221, p99 4,830.
- Aggregate input to output token ratio is 103.1 to 1.
- Cache hit share, defined as summed cached input tokens over summed input tokens, is 0.9803, so prefix reuse is already doing its work.
- One usage event records 1,225,458 input tokens, which exceeds any configured window, so that record is excluded from interpretation and reported only as a data-quality observation.

Limits that bound every conclusion here.
The broker runs with `captureBuffer: 0` and no artifact retains wire requests, so exact backend requests were not captured and are reconstructed instead.
`token_usage_record` carries no response id, so a usage event cannot be joined to a specific request boundary.
Sampling parameters are not retained anywhere, and the client configuration sets none, so all measured numbers are greedy and are labelled so.
Replaying a fixture generates the next response only, and no tool call recorded in a trace is ever executed.
73 requests, exactly the opening request of each conversation, were excluded because the client prompt is not stored as a message item and the template refuses a request with no user turn.

## Phase 2, serving comparison

Protocol: stock, candidate, candidate, stock on GPU 3 through the lease wrapper, identical target, runtime, context allocation, KV types, six speculative proposals, and greedy sampling, with warmups excluded.
164 requests completed and `summarize.py --require-abba` accepted the run.

| Measure | Stock | Rank-32 candidate | Change |
| --- | ---: | ---: | ---: |
| Measured requests | 80 | 80 | |
| Proposed tokens | 15,616 | 15,532 | |
| Accepted tokens | 9,226 | 9,232 | |
| Acceptance rate | 59.0804% | 59.4386% | +0.358 pp |
| Accepted per verification | 3.4973 | 3.5210 | |
| Decode tokens per second | 87.7199 | 88.2918 | +0.652% |
| Total request seconds | 3,468.30 | 3,460.99 | -0.211% |
| Decode seconds | 135.659 | 134.780 | |

40 fixtures were paired across 33 conversation groups.
The paired acceptance difference is +0.437 percentage points with a group-bootstrap CI95 of -1.041 to +1.701, 18 fixtures improving and 13 worsening, and a median difference of exactly 0.000.
The paired decode-throughput difference is +0.5975 tokens per second with CI95 -1.086 to +2.099.
39 of 40 fixtures produced identical normalised output hashes in all four stages, so the comparison is not driven by differing generations.
Decode accounts for about 3.9% of measured request time, which is a cold-cache replay figure and therefore understates the share that the sequential decode path owns in live use.

## Phase 3, whether a bounded pilot is justified

No bounded pilot was run, because the gate it would have to pass is the one that failed.
A pilot is warranted when a frequent regime shows remaining headroom that the current corpus plausibly does not cover.
Here the checkpoint shows no measurable advantage in any frequent real-usage regime, so there is no gap for a corpus to fill, and generating one would repeat the earlier null result at roughly 500 GB of cost.
The only demonstrated advantage lives on the synthetic agent-state distribution, which is precisely the distribution the corpus already over-represents.

## Recommendation

Attack carried context volume before drafter training.
Replay the live agent loop with a warm cache, tune compaction and tool-output retention so median input tokens per turn falls materially below 80,138, and hold the task pass rate.
Measure drafter acceptance on the identical warm-cache replay before and after, and promote only a change that improves both, because shorter context is also the most plausible route to the 61% acceptance headroom.

## Reproduction

From `.marathon/drafter-training/`:

- `venv/bin/python local-eval/inventory_traces.py` for the population and cache statistics.
- `venv/bin/python local-eval/extract_fixtures.py --holdout 40 --dev 16` for the fixture set and its manifest.
- `venv/bin/python lease_run.py 3 venv/bin/python benchmark.py <candidate> --name <fresh-dir> --local-fixtures local-eval/fixtures.jsonl --fixture-split holdout` for the run.
- `python3 summarize.py <fresh-dir> --require-abba` and `venv/bin/python local-eval/analyze_usage_eval.py <fresh-dir>` for validation and statistics.

Fixture files are private and stay local. Shareable identifiers are opaque.
Experiment workers were released and GPU 3 was left idle.
