# Pi versus Marathon: matched small coding benchmark

Pi used less input context, but the matched comparison showed essentially equal total completion time and no correctness difference.
Both harnesses passed all four independent checks.
This does not establish a general quality ranking or show that Codex makes the model regress.

## Primary results: matched personal instructions

| Metric across two jobs and their follow-ups | Marathon | Pi |
| --- | ---: | ---: |
| Independent checks passed | 4/4 | 4/4 |
| Total completion time, excluding backend startup | 120.45 s | 120.44 s |
| Uncached input tokens processed | 32314 | 15150 |
| Generated tokens, including reasoning | 5736 | 8176 |
| Input tokens summed over requests, including cached history | 254269 | 169970 |
| Cached input tokens summed over requests | 221955 | 154820 |
| Inference requests | 21 | 30 |

Pi processed 53.1% fewer uncached input tokens and generated 42.5% more tokens.
Its total submitted input, including repeated cached history, was 33.2% lower.
Do not interpret summed input history as unique context or fresh prefill work.
The total times are effectively tied; the fractional-second difference has no significance.

| Task | Marathon | Pi | Independent checks |
| --- | ---: | ---: | --- |
| Repair record parser | 36.16 s | 27.37 s | Both pass |
| Add age filtering on follow-up | 20.28 s | 30.14 s | Both pass |
| Repair interval merging | 34.58 s | 27.02 s | Both pass |
| Add optional touching-interval merge | 29.42 s | 35.91 s | Both pass |

Pi was faster on the initial fixes; Marathon was faster on both follow-ups.
The first parser request contained 10377 prompt tokens for Marathon and 3671 for Pi.
The native tool sets contained 12 Marathon tools versus Pi's four tools: read, bash, edit, and write.
Their different system prompts, tool descriptions, history management, and tool execution are part of the harness comparison.
The result does not isolate any one of these mechanisms.

## Controls

- Same deployed Swift/uncensored Qwen 3.8 27B IQ4_XS target, Marathon-tuned DFlash2 drafter, production backend image, Q8 target cache, 196000 context capacity, and inference arguments.
- Same RTX 3090 on GPU 3 at 250 W, one request stream at a time under the existing GPU lease guard.
- Temperature zero, seed 8123, top-p 1, top-k 0, min-p 0, repeat penalty 1, medium reasoning, and 4096 output tokens per request, verified in captured backend requests.
- Identical user task text and initial Python source/test files in fresh Git workspaces.
- Each follow-up resumed its own harness's first turn and resulting files.
- Backend restarted before each two-turn session so neither harness inherited the other's cache; cache reuse within a session remained enabled.
- Backend loading time excluded; CLI startup, inference, file editing, and test execution included.
- Order was Marathon then Pi for records, Pi then Marathon for intervals.
- Same Papi AGENTS instructions and exact optional skill-description block supplied to both harnesses in the primary pass.

Marathon used its normal Codex-based CLI, version `Marathon 0.4.0`, and its Responses API adapter.
Pi used `@mariozechner/pi-coding-agent` 0.73.1, installed in an isolated diagnostic directory, with its Chat Completions integration.
The pre-existing Pi 0.57.1 bundled inside OpenClaw was discovered but not changed or used.
Pi's extension, theme, prompt-template, context-file, and skill auto-discovery were disabled; the matched personal instructions and skill catalog were supplied explicitly with `--append-system-prompt`.
Marathon loaded those same user resources through its usual isolated-home preparation.
Native system prompts, API adapters, tool schemas, and sandbox implementations were not made identical; this is a whole-harness comparison, not an isolated prompt ablation.
Both tasks prohibited network use and delegation.

Parser checks covered invalid input, strict errors, whitespace, numeric conversion, order preservation, empty input, filtering, and strict validation with filtering.
Interval checks covered nesting, duplicates, overlapping versus touching ranges, input immutability, invalid ranges, empty input, and touching chains.
The independent checks were not shown to either model during task execution.

## Fairness correction

An initial exploratory pass accidentally compared Marathon's inherited user instructions and skills against a clean Pi configuration without those resources.
That pass showed Pi taking 87.11 seconds versus Marathon's 117.16 seconds, with both passing 4/4.
Those numbers are not the primary fair comparison and must not be quoted as a demonstrated Pi speedup under matched instructions.
The mismatch was detected by inspecting actual request bodies, then both harnesses were rerun on fresh workspaces with shared personal context.
All eight primary-pass requests groups completed without harness errors, and all returned outputs passed the checks.

## Limits, evidence, and cleanup

This is one matched pass across two small Python tasks, not a statistically powered benchmark or a large-repository agent evaluation.
It used greedy sampling for control, so results may differ from normal temperature-1 sessions.
No compaction, lengthy conversation, vision, browsing, or production project was evaluated.
The evidence supports lighter input context for Pi, but neither an overall speed advantage nor a quality regression from Marathon on these tasks.

Primary requests, original requests before sampling normalization, event streams, generated code, server timings, checks, launch commands, and summaries are in `.marathon/diagnostics/pi-marathon-20260921/matched/`.
The parent directory retains the exploratory run, pinned isolated Pi installation, and repeatable benchmark scripts.
Synthetic workspaces remain under `/tmp/pi-marathon-20260921*` for inspection; none are production repositories.
The isolated `marathon-pi-bench` container was stopped, GPU 3 returned to 15 MiB, and the central GPU configuration was byte-checked unchanged.
No private conversations or production application settings were modified.

CLI/API setup was checked against [Pi's custom-model documentation](https://pi.dev/docs/latest/models), [Pi's CLI documentation](https://pi.dev/docs/latest/usage), and [Codex non-interactive documentation](https://learn.chatgpt.com/docs/non-interactive-mode).
Benchmark findings above are local measurements, not claims from those documentation pages.
