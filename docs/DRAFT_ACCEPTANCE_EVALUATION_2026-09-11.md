# Draft acceptance investigation

Date: 2026-09-11.

## Conclusion

Keep the current official Q4_K_M DFlash2 drafter, six-token window, zero confidence cutoff, and Q4 draft KV.
No tested candidate established a 10% gain while retaining the current target and context capacity.
Higher acceptance alone was often slower, and the small Q8 draft-KV decode gain did not produce a meaningful completed-task latency improvement.
This supports the existing configuration within the tested search, not a claim that all future drafting research is exhausted.

## Question and constraints

Can Marathon gain meaningful speed by improving speculative drafting while retaining the current target model, reasoning behavior, and context capacity?
The target IQ4_XS weights, medium reasoning, `Minimize thinking`, target Q8 KV, runtime image, and 196,000-token requested context allocation were held fixed unless explicitly stated otherwise.
No production configuration or model was replaced.
This is a search among concrete feasible candidates, not proof of a global Pareto optimum.

## Research findings

The current drafter is the official Inco AI DFlash2 checkpoint, quantized to Q4_K_M.
The [z-lab model card](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) explicitly identifies that repository as a mirror of Inco AI, not a newer model.
The official examples propose seven tokens, but their performance figures use an H200, a different target precision, and xhigh reasoning.
Those figures cannot be directly substituted for this RTX 3090 setup.

The newest [official GGUF repository](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF) revision inspected was `51962825493a48b846b40126d35c799ac4093ad0`.
Its Q4 file differs from the installed revision by 64 bytes.
All 81 weight tensors are byte-identical; the only added model metadata is `dflash.rope.dimension_sections = [64, 0, 0, 0]`.
This is not a new trained drafter.

[DSpark v2](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark) is a distinct candidate with published improvements over DSpark v1.
Its published evaluation uses NVFP4/FP8 targets, so a local test is necessary.
The Q4 conversion tested comes from [Anbeeld](https://huggingface.co/Anbeeld/Qwen3.8-27B-DSpark-GGUF), pinned to `77843de41a67edb1c6294d641ea38fa78f9c8124` and verified against its published SHA-256.

An [uncensored DFlash2 upload](https://huggingface.co/taniii1234/Qwen3.8-27B-Uncensored-DFlash2) has no model card or documented training/evaluation procedure.
Its architecture and tensor shapes match the official drafter, while sampled tensor payload ranges differ.
It was treated as an experimental candidate, not assumed to be trained for Marathon's exact target.
Revision `bd03640a3e49a6eff782e82efafc1ab78e01fedf` was downloaded and its SHA-256 verified.

Marathon's target is itself modified from the standard Qwen model.
Its [publisher documents changes to attention output and MLP down-projection weights](https://huggingface.co/JonathanColetti/Qwen3.8-27B-Uncensored-GGUF).
A mismatch between the standard drafter's training target and this target is a plausible contributor to acceptance, but these experiments do not isolate or quantify that contribution.

## Method

Each comparison uses the same physical GPU, with production controls before and after where the run completes.
Workers acquire Marathon's existing exclusive backend lease.
GPU 1 was already leased, so no experiment was launched there.
GPU power limits were left unchanged.

The initial screen uses three fixed source-plus-coding prompts at 7,073, 14,069, and 17,070 input tokens, two sampled seeds, and a greedy setting.
Each timed response allows 512 generated tokens, with separate warm-ups.
These frequently end during reasoning and measure decoding, not completed-task quality.
Sampled output changes are recorded and do not by themselves prove a quality regression or improvement.
All requests retain temperature 1, top-k 20, top-p 0.95, min-p 0.05 except the explicit greedy checks.

Speed comparisons use paired geometric means of server decode-rate ratios.
Acceptance percentage is accepted draft tokens divided by proposed draft tokens.
It is not the optimization objective: a shorter draft can improve that percentage while producing fewer useful tokens per verification pass and slowing the response.

## Feasibility findings

Seven-token drafting loaded at 196K but failed with CUDA out of memory during the first fixture's greedy request.
Its two completed sampled measurements are an incomplete, selected screen and do not establish a deployable improvement.
The crash log and partial results were retained, and the remaining width tests resumed in a separate run.

The official Q8-weight DFlash2 and Q4 DSpark candidates also failed to load at the full allocation.
These failures establish a capacity constraint, not an acceptance or speed comparison.

## Initial full-capacity screen

| Candidate | Decode-rate change | Aggregate draft acceptance | Measurements |
|---|---:|---:|---:|
| Current six-token drafting | Reference | 45.0% | 9 per baseline stage |
| Two-token drafting | -26.8% | 74.6% | 9 |
| Four-token drafting | -9.2% | 55.7% | 9 |
| Five-token drafting | -2.1% | 51.4% | 9 |
| Confidence cutoff 0.2 | -2.9% | 43.8% | 9 |
| Confidence cutoff 0.5 | -13.0% | 60.4% | 9 |
| Q8 draft KV | +1.9% | 47.1% | 9 |
| Latest official Q4 metadata | +0.3% | 45.0% | 9 |

Four-token drafting uses the initial control only because the later seven-token stage crashed before a return control.
Other rows use their completed same-card before-and-after controls.
Different sampled trajectories occur for several variants, so these are workload screens rather than exact-output microbenchmarks.
Only five of nine Q8 draft-KV outputs matched the controls; all nine latest-metadata outputs and their draft counts matched.
The two-token result demonstrates why maximizing acceptance percentage is the wrong objective.

An interruption stopped the first alternate-weight and completed-task clients during generation.
Their partial evidence remains in `uncensored-screen` and `kv8-quality`, and idle workers were explicitly cleaned up before continuing.
The partial alternate-weight result was 4.7% slower over five measurements and is not treated as a completed comparison.

## Completed-task check for Q8 draft KV

Five workloads with two seeds each cover Python implementation, Python debugging, SQL, structured extraction, and tool calls.
The existing speculative-acceptance checker was reused, with medium reasoning, `Minimize thinking`, fixed production sampling, and a 4,096-token cap.
Python functions were executed against bounded functional checks, SQL against expected SQLite results, and tool calls against their expected arguments.
Tool calls were validated rather than executed.

Production, Q8 draft KV, and the production return control each passed all ten checks, with no output truncated at the cap.
Both production runs reproduced all ten answer messages.
The candidate matched nine messages; one debugging response generated 765 tokens rather than the baseline's 572, while still passing the functional checks.

| Measure | Q8 draft KV versus paired production controls |
|---|---:|
| Decode rate, geometric mean | +1.12% |
| Whole-answer latency, geometric mean | +2.20% slower |
| Latency on nine matching-output pairs | -0.37% faster |
| Functional checks | 10/10 for all three stages |

Total request time was 33.84 seconds for initial production, 35.34 seconds for the candidate, and 33.73 seconds for the return control.
These small samples do not establish a general quality regression, but they do not justify deployment as a useful speed improvement.
Reasoning changes were not requested or induced through different prompt settings; changing draft behavior can change the sampled trajectory.

## Equal reduced-capacity comparison

To distinguish capacity failures from poor speed, production and three candidates were rerun at the same 131,072-token allocation on GPU 2.
Each candidate received the 7K and 14K fixtures with two sampled seeds and greedy decoding, for six timed measurements.
Production controls bracketed the complete run.

| Candidate | Decode-rate change | Matching outputs out of six |
|---|---:|---:|
| Official Q8 draft weights | -3.34% | 3 |
| DSpark Q4 | -20.06% | 1 |
| Seven-token DFlash2 | -0.18% | 3 |

The Q8-weight candidate also contains the official rotary metadata addition, so this is a candidate-package comparison rather than a perfectly isolated quantization experiment.
That metadata alone had identical text outputs and acceptance counts in its separate screen.
The seven-token result varies substantially by prompt and seed; its overall result does not justify losing context capacity.
DSpark was screened at six proposals, not exhaustively tuned across all of its runtime and confidence settings.
These losing screens were not promoted to broader completed-task evaluation.

## Alternate uncensored drafter

The alternate BF16 safetensors checkpoint was converted using llama.cpp's DFlash tensor name map and the installed drafter's metadata.
All 81 tensor shapes were checked against production, BF16 values were converted to F16 or F32 as appropriate, and written tensors were verified.
The existing llama-quantize executable then produced Q4_K_M weights, and every resulting tensor's shape and quantization type matched the production drafter.
No checkpoint-provided Python code was executed.

The completed same-card, full-capacity comparison found the alternate drafter 1.34% slower over nine timed measurements, with 44.3% aggregate draft acceptance versus 45.0% for production.
Five outputs matched the controls.
This result does not establish any advantage from the alternate weights, and the upload lacks evidence that it was trained against Marathon's exact target.
It was not promoted or subjected to broader quality testing after this losing screen.

## Remaining research possibilities

A drafter trained or calibrated against the exact modified and quantized target could potentially improve acceptance, but this investigation provides no measured benefit or credible percentage estimate for that work.
The current greedy draft selection and target verification implementation also differs from engines that sample a draft distribution and use rejection sampling.
Changing that algorithm would require careful distributional correctness and performance validation; it is not an established free optimization.
Neither changing the target sampler nor increasing reasoning effort is justified as a way to improve the acceptance statistic.

## Reproduction

Runtime image: `sha256:443d87c87fe673faf379b29dd11bfa22e03c76e4d2fe14e9ce9503c810daf34c`.
Raw requests, responses, model revision metadata, downloads, conversion recipe, worker commands, GPU memory snapshots, and failure logs are under `.marathon/diagnostics/draft-policy-20260911/`.
The initial width and policy processes used the same runner logic with their original benchmark container names; later runs use `bench-draft-policy-2` and `bench-draft-policy-3`.
Exact launch commands are saved for every stage.
Experiments use isolated native server endpoints, not complete Marathon UI or agent trajectories.

## Final state

All experiment workers were stopped and removed, releasing GPUs 2 and 3.
The broker health endpoint returned OK.
The central configuration retained SHA-256 `9da896acf2d0686bd7e523b895a44a4ba05686d5249245e36f74ca81e4dd534d`.
The target, production drafter, runtime source, and production settings were unchanged.
The new report is the only addition to tracked project scope from this investigation; the earlier llamAmpere report was preserved.
Model candidates and reproducibility evidence remain in the ignored diagnostics directory for any follow-up, with hashes in the saved manifests.
