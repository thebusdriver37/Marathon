# Context retention experiment, 2026-09-16

## Decision

Stop pursuing the current broad tool-output retention policy for production use.
The warm-cache speed result is strong and repeatable.
The independent completed-task oracle exposed a retention-related behavior regression that two refinements did not repair.

This decision does not rule out a narrower experiment later.
It does rule out adopting the current policy, including the v5 variant, without a different mechanism and broader task validation.

## Scope

The experiment compared the original retained history against deterministic reductions of older and repeated tool output.
It used eight opaque conversation chains, four turns per chain, and 32 measured requests.
Both target model, drafter, runtime, GPU allocation, context allocation, key-value types, and speculative settings were fixed.
The experiments used the registered GPU 3 slot through the lease wrapper.
The registered production worker on GPU 1 was not interrupted.

## Warm-cache baseline

The server directly reported prompt processing, cached-token counts, prompt time, decode time, and generated-token counts.
Wall time was measured by the client.
Client preparation and transport were inferred as wall time minus server prompt and decode time.

Aggregate measurements were:

- Total wall time: 1,074,055.927 ms.
- Prompt time: 1,025,785.143 ms, or 95.506 percent of wall time.
- Decode time: 40,028.434 ms, or 3.727 percent of wall time.
- Inferred client overhead: 8,242.350 ms, or 0.767 percent of wall time.
- Speculative acceptance: 57.7845 percent.
- Accepted proposals per draft event: 3.396717.
- All 24 continuation turns showed server-reported prefix reuse.
- No continuation turn missed the expected reuse.

The dominant cost is newly processed prompt context, not decode.

## Performance result

The initial reduced policy produced a meaningful and statistically supported speed win.

- Wall time fell by 255,461.023 ms, or 23.785 percent.
- Conversation-group bootstrap CI95 for wall time: -434,834.309 to -126,007.353 ms.
- Newly processed prompt tokens fell by 134,943, or 19.511 percent.
- Bootstrap CI95 for new prompt tokens: -213,256 to -72,122.
- Server prompt time fell by 255,431.320 ms, or 24.901 percent.
- Acceptance rose from 57.7845 percent to 61.5894 percent.
- Thirty of 32 paired cases improved and two regressed.

The refined duplicate-excerpt policy produced an almost identical speed profile.

- Wall time fell by 253,810.114 ms, or 23.631 percent.
- New prompt tokens fell by 134,337, or 19.424 percent.
- Server prompt time fell by 254,405.626 ms, or 24.801 percent.
- All 24 continuation turns still showed server-reported prefix reuse.
- Two paired requests regressed by 79.3 to 794.1 ms because generations diverged.

The 8-chain reduction itself took 0.298 seconds of wall time.
That is about 0.12 percent of the wall time saved by the v4 comparison.
It is therefore not a plausible explanation for the net win.

## Quality gate

The paired quality gate contained eight cases per condition and four ABBA-style measured requests per case.
Independent oracles checked executable code, SQL results, structured JSON, prior error evidence, prior markers, and a user correction.

Baseline pass rate:

- v3 baseline: 12 of 16 measured requests, or 75 percent.
- v3 reduced: 6 of 16, or 37.5 percent.
- v4 and v5 reduced: 8 of 16, or 50 percent.

Failures shared by both baseline and reduced conditions are treated as model or oracle defects, not retention defects:

- Exact arithmetic failed identically in both conditions.
- The user-correction oracle failed identically in both conditions.

The deliberate marker-retrieval failure is expected because the policy intentionally omitted two salient markers from one retained older output.
That failure demonstrates a hard limit: an agent cannot recover exact omitted strings unless the policy retains them or can retrieve the original record.

One unexplained regression remained in both v4 and v5:

- The extraction task failed only under reduction.
- The final task message and the required log lines were byte-identical after reduction.
- The reduction changed only unrelated historical tool output.
- The model nevertheless returned an active account that the unchanged log marked inactive.

The initial policy also changed failed-tool behavior into an attempted `refresh-cache` tool call.
That particular failure was repaired by preserving a visible excerpt of duplicate output.
This repair did not transfer to the extraction case.

## Interpretation

The performance mechanism is credible.
Reduced histories materially reduce newly processed prompt tokens and the measured prompt path.

The policy is not safe as written.
The extraction failure proves that apparently irrelevant history can affect model answers.
The marker failure proves that exact historical values cannot be inferred once omitted.
A single-turn oracle also cannot establish whether a full agent loop repeats commands or recovers correctly after compaction.

## Recommendation

Do not deploy the current policy.
Do not run broader validation of the same policy at this scope.

A future experiment should start from a different mechanism:

- Treat original histories as immutable and make the reduction a retrieval view.
- Keep exact identifiers and required markers out of omission regions.
- Start with polling or repeated command output, not broad age-based history compaction.
- Require matched answers on a larger held-out completed-task suite before measuring speed as a benefit.
- Compare completed-task wall time, turns, repeated commands, and recovery, not single-request latency alone.

## Artifacts

Experiment root:

- `.marathon/drafter-training/local-eval/`

Key artifacts:

- `warm-baseline-v2-20260916-142951/`
- `warm-reduced-full-v3-20260916-150307/`
- `warm-reduced-full-v4-20260916-160125/`
- `warm-comparison-v1.json`
- `warm-comparison-v4.json`
- `task-quality-v1-20260916-152518/`
- `task-quality-v4-20260916-161548/`
- `task-quality-v5-20260916-164639/`
- `BUDGETS.md`

No private prompts, responses, fixture paths, tool arguments, or generated histories are included in this report.
Experiment processes exited and GPU 3 was released.
