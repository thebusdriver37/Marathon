# Spark FP8 versus BF16 KV: small matched screen

Both settings passed all four objective tests.
No quality advantage or consistent decode-speed improvement was demonstrated for BF16 KV.
Retain FP8 pending evidence from a task that reproduces the reported quality problem.
This screen does not prove that FP8 has unchanged quality on difficult production work.

## Method

The live `iSkye/Qwen3.8-Flash-Next-NVFP4-ablit-a070` checkpoint and patched vLLM backend ran on the single Spark GB10.
Only `KV_CACHE_DTYPE` changed from `fp8` to `auto`, selecting BF16.
The model revision, custom QSA overlay, medium reasoning, MTP3, 47172-token draft vocabulary, BF16 recurrent state, 262144 context limit, and 0.786 GPU-memory budget remained fixed.
Each arm received identical synthetic user prompts through the existing Spark tunnel and `/v1/chat/completions` endpoint.
Sampling was temperature 1.0, top-p 0.95, top-k 20, seed 8123, presence penalty 0, repetition penalty 1, and a 3072-token output cap.
All requests ended normally without hitting the output cap.
Requests were sent sequentially after idle checks; no private conversations were inspected.
These were direct backend requests, not a complete Marathon agent workflow.

The four cases checked all valid job schedules under constraints, invoice rounding, Python mutable-list aliasing, and extraction plus corrected arithmetic from a 36472-token synthetic document.
Answers were scored against explicit expected JSON values.
Markdown fences were tolerated by the answer parser; instruction-format fidelity beyond JSON content was not separately graded.

## Results

| Case | FP8 pass | BF16 pass | FP8 decode | BF16 decode | FP8 response time | BF16 response time |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Job scheduling | Yes | Yes | 50.58 tok/s | 48.84 tok/s | 20.78 s | 28.02 s |
| Invoice rounding | Yes | Yes | 57.77 tok/s | 58.63 tok/s | 18.25 s | 23.23 s |
| Python aliasing | Yes | Yes | 55.18 tok/s | 56.90 tok/s | 11.72 s | 10.63 s |
| 36K document | Yes | Yes | 60.21 tok/s | 59.19 tok/s | 23.33 s | 25.95 s |

Decode rate includes reasoning and final-answer tokens and is measured as `(completion_tokens - 1) / (last streamed token time - first streamed token time)`.
Response time includes prompt processing and generation but excludes model startup.
FP8 generated 3043 tokens across the four requests; BF16 generated 3643.
Different reasoning lengths explain part of the response-time variation and are not evidence of slower kernels.
The document's first-token latency was 17.75 seconds with FP8 and 19.19 seconds with BF16.

The BF16 backend reported a 16.42 GiB cache pool with capacity for 599741 tokens, or 2.29 full 262144-token sequences.
Thus the configured single-conversation limit still fit, while aggregate cache capacity was reduced.
This was a capacity report, not a fully occupied 262K test.

## Limitations and restoration

There was one run per task per precision, without randomized ordering or repeated restarts.
FP8 started warm, while BF16 had just restarted and logged a sampling-kernel JIT compilation on the first task.
Small speed differences are therefore not established gains or regressions.
Both arms reaching 4/4 means this screen cannot rank their reasoning quality or reproduce the user's subjective complaint.
The recurrent-state precision and uncensoring modification were not varied.

The original `.env` bytes were backed up privately and restored automatically after the candidate tests.
Final FP8 readiness passed, and a request through the normal authenticated `spark/Qwen3.8-Flash-Next` broker route correctly returned `42` for `19 + 23`.
The restored `.env` matches its original bytes exactly, SHA-256 `abff4899e425a6369be2ec78b6d064c65c813a142a7b785138aaa39aaae0fe56`.
The captured candidate and restored container arguments differ only in `--kv-cache-dtype auto` versus `fp8`.
Raw requests, answers including synthetic reasoning, per-request metrics, launch arguments, startup logs, and the repeatable probe are in `.marathon/diagnostics/spark-kv-ab-20260920/`.
No model weights, credentials, or raw conversation artifacts are added to version control.
