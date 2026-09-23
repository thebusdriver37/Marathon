# Spark fictional-user comparison

## Bottom line

NVFP4 was faster overall in this small screen, while both deployments preserved most of Joe's changing details and produced working code.
The strongest NVFP4-specific observations were one verbatim duplicated diary, an incorrect practical recommendation in a sourced answer, and an unsolicited timing assumption carried forward as fact.
The diary duplication did not recur in two exact-request replays after restarting NVFP4, so recurrence and root cause are unproven.
GGUF also made a basic initial time-arithmetic error and ignored some output constraints.
There is no established overall accuracy winner and no basis here to fine-tune or automatically replace the current deployment.

## Scope

The user's suspected problem was withheld throughout testing.
This was a hypothesis-uninformed comparison, not a model-label-blinded study.
No Hermes application, conversations, memory, configuration or personal records were inspected or used.
The shared Spark inference service was accessed directly through its existing loopback SSH tunnel.

The reusable runner is `scripts/evals/spark_joe_probe.py`.
It launched real Marathon CLI sessions, resuming the same thread across ten scripted turns for each deployment.
The user prompts, medium reasoning, temperature 1.0, seed 41, 4,096-token per-request output ceiling and available tools were held constant.
The fictional Joe Simmons conversation covered scheduling, corrections, groceries, an SMS draft, practical advice, executable Python, official-document research, conflicting fictional notices, final recall and constrained prose.
All main turns completed without hitting the time or output ceiling.
User messages were identical, but model answers, tool results and subsequent histories naturally differed.
Separate workspace names and runtime chat templates also introduced small input differences.
Identical seeds do not imply identical sampling across different engines.

## Deployments

- NVFP4: `iSkye/Qwen3.8-Flash-Next-NVFP4-ablit-a070`, snapshot `91c3e3d4daf14f8e9389b95f43112410f06ed3d5`.
- Its existing vLLM image was `vllm/vllm-openai@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8`, with the installed QSA overlay and V2 runner.
- NVFP4 retained 262,144 context, FP8 KV and three-token MTP with its configured draft vocabulary.
- GGUF: `Qwen3.8-Flash-Next-UD-IQ4_XS-nopple.gguf`, its PLE sidecar and `mtp-shared-Q8_0.gguf`.
- GGUF used `/home/deforest/Ai/llama.cpp-spark-exp/build-cuda-exp/bin/llama-server`, the rollback service's arguments, F16 KV, two-token MTP, 262,144 context and 16 GiB prompt cache.
- GGUF ran as an isolated transient unit, `joe-gguf-probe.service`, while both permanent GGUF and NVFP4 units were stopped.
- The only intentional GGUF launch change was binding to `127.0.0.1` instead of the rollback unit's `0.0.0.0`.

This compares usable deployments, not quantization in isolation: weights, ablation, engines, templates, cache formats and speculation differ.
The GPU-management skill guided the reversible service switch and protection of other workloads.

## Main-session results

Times are actual CLI completion times, including thinking and tools but excluding model loading.

| Turn | NVFP4 seconds | GGUF seconds |
| --- | ---: | ---: |
| Initial day plan | 19.4 | 26.5 |
| Appointment correction | 10.3 | 9.2 |
| Grocery arithmetic | 13.8 | 9.7 |
| Short SMS | 6.0 | 5.6 |
| Practical opinion | 9.4 | 5.2 |
| Build and test budget app | 45.6 | 69.0 |
| Official-document web research | 16.5 | 34.2 |
| Conflicting notices | 6.8 | 14.7 |
| Final memory handoff | 11.8 | 15.5 |
| Diary prose | 14.5 | 28.7 |
| Total | 154.4 | 218.2 |

NVFP4 completed this particular sequence in approximately 29% less time.
It was not faster on every turn, and this is not a repeatable throughput guarantee.
GGUF's cold model load took approximately 138 seconds, excluded above.
GGUF spent about 23.7 seconds before the diary's first visible prose, versus 3.4 seconds for NVFP4.
The research paths returned different amounts of source text, so that timing difference is partly a tool/context difference.

### Shared successes

Both selected the same $39.50 basket, retained coffee/rice/eggs as already owned, kept detergent unscented, and remembered the corrected Wednesday 7:10pm pickup by Eli.
Both preserved the cat's turkey-only diet and Eli's cilantro restriction.
Both returned six bullets in the final handoff and identified the postponed repair-cafe date.
Correction after re-reading the saved answers: NVFP4 explicitly recommended bringing the microwave despite the unchanged prohibition.
The original report incorrectly counted that restriction as preserved; its answer in `nvfp4-main/turn-08/answer.txt` proves otherwise.
Both used actual command execution and actual web tools rather than merely claiming to do so.
Both final budget helpers passed independent checks for an empty basket, duplicate purchases, exact cents, the $39.50 basket, and rejection of negative, non-finite, malformed and sub-cent inputs.
NVFP4 initially used an invalid Decimal method keyword, encountered a real test failure, corrected it and reran passing tests.
GGUF generated a larger implementation and fourteen passing self-tests.

### Specific misses

NVFP4's diary repeated the same 129-word passage verbatim, producing 258 words against a 100-130-word instruction.
The repeated text appears in the actual CLI answer and in the captured streaming text count, not just in this report.
GGUF's main diary was 114 words without that duplication.

NVFP4 correctly explained that Decimal constructed from a float retains its approximation, then contradicted the practical point by suggesting `Decimal.from_float` as an alternative alongside strings.
An independent Python check confirmed `Decimal.from_float(0.1) == Decimal(0.1)`, both retaining the binary approximation; `Decimal('0.1')` does not.
GGUF's main sourced answer correctly recommended a string instead.

NVFP4 added an unsolicited five-minute parking buffer, changing the requested Thursday departure calculation from 8:45 to 8:40, then carried its own addition into the final handoff despite the instruction to use supplied facts only.
That is an added assumption, not evidence it cannot subtract 35 minutes.
GGUF's initial departure was arithmetically wrong: it wrote 7:25 for an 8:40 appointment with 25 minutes of travel and 10 minutes early arrival, where 8:05 was correct.
GGUF did calculate 8:45 correctly after the appointment changed to 9:20 on Thursday.

GGUF's SMS was 38 whitespace-delimited words despite the under-35 instruction; NVFP4's was 31.
GGUF offered to write the plan to a file despite Joe's request for no follow-up pitches.
Both initial plans added details such as assumed shopping times and appointment-end expectations that were not supplied.
Suggested schedule slots are not automatically errors, but confidently assuming an unknown appointment duration is not warranted.

## Targeted follow-up and limitations

The original NVFP4 diary request was replayed directly to GGUF, preserving its exact preceding transcript and tool context rather than substituting GGUF's history.
GGUF returned a single diary without verbatim duplication.
After restoring NVFP4, two exact-request replays also returned single diaries, of 114 and 133 whitespace-delimited words respectively.
The latter slightly exceeded the 130-word ceiling but did not repeat the passage.
Thus duplication was observed in one of three NVFP4 diary attempts, versus neither of two GGUF attempts, with different original histories and replay conditions limiting any frequency comparison.
These are not sufficient samples to estimate a failure rate.
The NVFP4 replays took 32.0 seconds cold-prefix and 14.2 seconds warm-prefix, with 832 versus 232 reported reasoning tokens despite the same seed and request.
That demonstrates observed response variability, not a proven cache defect or a controlled causal measurement of reasoning overhead.
The first original diary request had only 51 reported reasoning tokens, so one timing sample should not be treated as a stable reasoning-cost estimate.
This direct replay is an inference diagnostic, not a replacement agent harness.
A direct replay of NVFP4's last research request to GGUF requested another source fetch; without executing that extra tool it has no scored final answer.

The main screen used one conversation per deployment, not a statistical benchmark.
Occupied inputs were approximately 10K-20K tokens, not near the maximum context window.
Neither a single factual miss nor absence of a miss proves a persistent model trait.
The user's application-specific issue could still involve an application prompt, memory, rendering or integration not exercised here.
The findings do not justify attributing every difference to ablation or NVFP4 quantization, or automatically changing the user's default.

## Local evidence

## Bounded mitigation screen, September 22

A fresh actual Marathon CLI repair-cafe turn correctly rejected the microwave, unlike the original longer conversation.
The original captured request contained the prohibition, so that observed miss was not caused by dropping the user's instruction before inference.
Inspection of the loaded chat template and Responses conversion did not establish a broken tool-call sequence or missing instruction.
This is not proof that the entire runtime is defect-free.

Twenty-four direct Responses requests compared unchanged settings, `preserve_thinking=false`, and a short general constraint-checking system instruction.
All retained enabled medium reasoning on the same loaded iSkye NVFP4 service.
Each arm received four cases at two seeds: the captured cafe and departure histories, plus two new structured constraint tasks inserted into the same earlier history.
These were matched inference replays, not twenty-four new end-to-end agent sessions.

Neither candidate reliably eliminated the carried-forward parking assumption.
The unchanged arm and each candidate produced one 8:40-only departure answer and one answer giving the user-fact-derived 8:45 departure, sometimes with an optional parking suggestion.
The phrase "same buffers" in the follow-up makes that replay less clean than the original unsupported addition: the model may interpret it as adopting the prior assistant suggestion.
The new pickup task explicitly prohibited extra buffers and passed in all six attempts.
The pottery task had correct factual fields in all six attempts, but unchanged settings wrapped one answer in Markdown despite the JSON-only request.
Both candidates passed both strict JSON checks for that task; two attempts per arm cannot establish a reliable improvement.
All six cafe replays ultimately excluded microwaves, although the guarded seed-73 answer first asserted eligibility and then corrected itself in the visible response.
Some answers also loosely described the postponed Saturday event as no longer being "Saturday," rather than distinguishing September 26 from October 3.

The captured histories did not contain prior reasoning items, so disabling thinking preservation primarily changed the template's empty historical thinking wrappers in these cases.
It did not test the effect of removing a substantial saved reasoning history.
No candidate earned a production change, and no service restart, weight change, or application-default change was performed for this screen.
The evidence confirms user-visible reliability problems in this deployment, but does not isolate quantization, weights, kernels, or sampling as their cause.
Artifacts: `.marathon/diagnostics/spark-constraints-20260922/`, with requests, raw SSE, answers, usage and strict structured-task results.
Runner: `scripts/evals/spark_instruction_probe.py`.

## Original comparison artifacts

## Extra-high reasoning follow-up

Twenty additional matched Responses requests compared medium with xhigh on five cases at seeds 41 and 73, reversing arm order on the second seed.
The same loaded model, original history, temperature and tools were retained without the previous constraint guard or thinking-preservation change.
Both `reasoning.effort` and `chat_template_kwargs.reasoning_effort` were set consistently.
The loaded template explicitly supports xhigh and adds an instruction to validate assumptions, consider alternatives and prioritize correctness.
Both arms received an equal 8192-token output allowance, with the custom `thinking_budget_tokens` field removed from both; all twenty requests completed.
These are inference replays, not a new end-to-end tool-use comparison.

Extra-high did not eliminate constraint failures: its seed-41 cafe answer explicitly said the prohibited microwave was "fine as the one bring-item."
Its second attempt excluded microwaves; both medium attempts excluded them.
Extra-high gave 8:45 for departure twice, compared with one 8:40 and one 8:45 from medium, but both efforts still produced an 8:45 explanation mentioning an additional parking buffer inconsistent with that arithmetic.
The departure replay's "same buffers" ambiguity remains a limitation.
All eight structured pottery/pickup answers passed strict JSON and factual checks.
Extra-high diaries were 123 words each, within the requested 100-130; medium produced 135 and 128 words.
None repeated a whole passage.
Medium's first diary also excluded everyone else from the yellow mug despite Mara's permission, while its second carried forward the 8:40 departure.
Extra-high's first diary expanded the no-calls-before-eleven rule into no phone buzzing before eleven, so word-count success should not be treated as flawless factual fidelity.

Total request wall time was 107.5 seconds for medium and 128.3 seconds for xhigh, approximately 19% longer for this small set.
Reported reasoning-token totals were 3077 and 3637 respectively.
The first xhigh request had a cold prefix while the first medium request was cached, limiting interpretation of the timing difference.
These results show mixed improvements, not a dependable cure or a statistically established ranking.
No default or service configuration was changed.
Artifacts: `.marathon/diagnostics/spark-xhigh-20260922/`.
Reproduction: use `scripts/evals/spark_instruction_probe.py --reasoning-comparison --output NEW_DIRECTORY` while the same worker and source captures are available.

## Original comparison artifact paths

## Matched local merged 27B follow-up

Ran the same twenty requests on `Swift-Qwen3.8-27B-Uncensored-Merge-IQ4_XS.gguf`, the personal local Marathon model, not the unmerged original 27B.
All twenty saved request bodies were verified identical to the Flash Next medium/xhigh comparison after excluding only `model`.
This includes the original Flash Next history, both seeds, temperature, output allowance, tools and reasoning settings.
The local template supports both reasoning levels, and raw streams contained reasoning events.
Different runtime defaults, tokenization, kernels and cache behavior remain deployment-level confounds; this is not weight-only isolation.
The runner reserved Marathon worker 2 through its normal pool lease and broker, and unloaded that worker after testing without changing defaults or interrupting worker 1.

Cafe: medium explicitly allowed the microwave in both seeds.
Extra-high seed 41 did not address the microwave prohibition and asserted the user was not registered without evidence.
Extra-high seed 73 explicitly allowed the microwave, inventing an interpretation that "no microwaves" prohibited using one rather than bringing a broken one, and returned 74 words against the under-65 request.
Thus three of four answers explicitly contradicted the microwave exclusion; the fourth omitted it rather than demonstrating correct handling.
In the matched Flash Next run, one of four explicitly allowed the microwave and three excluded it.

Departure: all four local answers eventually gave 8:45, though medium seed 41 initially said 8:50 before correcting itself and invented a coffee-budget saving of approximately $3-5.
Structured pottery/pickup: seven answers contained the correct factual fields, but four used Markdown fences despite JSON-only instructions.
The remaining pickup answer, extra-high seed 41, emitted `exec_command` with arguments `{"cmd":"echo '{}'"}` despite "Do not use tools," and supplied no final visible answer.
The raw completed response confirms the tool request; this was not merely a missing text capture.
The diagnostic did not execute that requested tool.
Strict structured-task passes were 3/8 locally versus 8/8 in the matched Flash Next screen; distinguish formatting misses from incorrect facts.

All four local diaries met the 100-130 word requirement: 120, 107, 127 and 122 words, without whole-passage repetition.
However, medium seed 73 wrote "Tomorrow I go to the dentist; the day after, Eli gets Pickle," reversing the supplied Wednesday pickup and Thursday dentist order.
It also described the yellow mug as "mine and no one else's," despite Mara's permission.
Passing the length check therefore does not mean passing factual fidelity.

These results show that the local merge also exhibits the same failure categories under identical history, and do not support treating them as unique to Flash Next.
They do not establish a general model ranking from five small cases or invalidate the user's broader preference for the local model.
Artifacts: `.marathon/diagnostics/merge27b-constraints-20260922/`.
Reproduction: add `--local-merge` to the reasoning-comparison runner, with a new output directory.

## Original comparison files

Artifacts are under `.marathon/diagnostics/spark-joe-20260922/`.
`nvfp4-main` and `gguf-main` hold answers, CLI events, captured requests, per-request timing/usage and generated code.
`prompts.json` records the fixed user sequence.
`nvfp4` was a setup-only routing failure with no model request and is excluded from quality results.
Stream capture was added before GGUF and targeted replays; the initial NVFP4 run retains streaming counts, requests and CLI text, not full raw SSE.

## Handoff

The temporary GGUF worker was stopped, and the original NVFP4 service was restored, active, enabled and healthy on loopback port 8000.
Discovery again identifies the iSkye abliterated checkpoint.
The permanent GGUF service remains disabled, as it was before testing.
No model files, permanent service configuration or application defaults were changed or deleted.
Restoring NVFP4 required a lengthy cold initialization: its main weight-loading stage alone reported 530.64 seconds, separate from conversation timing.
The work consisted of twenty real CLI turns and four targeted single-request diagnostics, one of which requested an additional tool and therefore had no scored final answer.
