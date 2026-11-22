# Inkling 2 Mini with Maya

On AiGuy, Inkling is a selectable model in Marathon's `/model` picker and Majesty's
Marathon model list. Qwen remains the remembered default. Choose `Inkling 2 Mini`
(262144 context) or `Inkling 2 Mini 344K` (344064 context). A one-off terminal launch
can use `marathon codex -m inkling-2-mini`.

The deployment uses Maya c520fd14de plus attention-split.patch, attention split 80,
UD-Q4_K_XL target weights, FP8 KV, fixed width-4 DSpark and a 16384-token draft
context. Prefill chunks are 2048 for 262K and 512 for 344K. DSpark changed some
complete greedy outputs in the audit; the owner approved this configuration based
on measured agent quality. These results do not claim lossless equivalence.

The pinned runtime lives in gpu-control's `runtime/inkling-maya-c520fd14-attn80/`.
It packages the binary, CUDA libraries, tokenizer and full Python serving stack;
weights stay under `~/AI/models`. No experiment or NInfer directory is a runtime
dependency. Its README records image and file hashes, launch settings and rollback.
The native guard refuses CPU expert fallback before the API becomes ready.

Two machine-local `llama_swap_pool` backends, `inkling-maya` and
`inkling-maya-large`, map each worker to its card's original Qwen lock using
`pool_lease_backend` and `pool_lease_models`. Every worker remains loopback-only.
Marathon checks GPU occupancy before leasing; the launcher checks again before
loading. A worker receives a duplicate of the same flock descriptor through an
owner-only Unix socket, keeping its card reserved if its client dies during cold
load. Switching model pools unloads the old worker before releasing its lease.
Normal Inkling router exit unloads its worker; an abandoned broker worker expires
at the same 900-second idle TTL as Qwen. No card is dedicated to Inkling.

The portable catalog only declares backend placeholders. Configure both pools in
`~/.config/marathon/catalog.toml` before selecting its GGUF in setup; it has no slow
llama.cpp fallback. External model entries expose `inkling-2-mini` and
`inkling-2-mini-344k` through the same shared pools in the Codex model picker.
The Qwen worker IDs, launch flags, pinned image, caches and default are unchanged.

## Protocol and limits

Use Maya's full streaming Responses server, with `reasoning_transport = "responses"`
and slot APIs disabled. The smaller benchmark `maya-adapter.py` is not the deployed
interface. The full server canonicalizes OpenAI JSON-string tool arguments into
objects before rendering Inkling's template. Multi-turn tool histories work.
Maya maps the requested effort into the template. Switching effort changes an early
prompt prefix and can require a full prefill; keep effort stable for large contexts.

The router also supports `chat_template_effort` for separately configured llama.cpp
servers: it puts `none` through `xhigh` in `chat_template_kwargs.reasoning_effort`,
removing conflicting OpenAI effort/enable_thinking fields. This avoids llama.cpp
#32753, which otherwise turns `none` into the template's high-effort default.
That compatibility path is tested but is not used by the deployed Maya runtime.

Inkling currently lacks Qwen's slot-file cache and cross-process snapshot resume,
fast-mode LoRA, and GPU vision. Its model entries accept text only. Maya can reuse a
live prefix internally; that is not a durable slot snapshot. Normal conversation
resume reconstructs the prompt. The frontend's 1,048,576-character turn-input limit
still applies even when the model's token window has room. Maya's sampler considers
at most 64 candidates, so equal API settings do not imply identical Qwen sampling.

`fit_max_tokens` reduces the output allowance to the remaining window without
truncating the prompt. Sampling defaults are in the config's `sampling` object,
not a top-level seed (which Maya ignores). The deployed defaults are temperature
1.0, top-p .95, top-k 64, min-p 0, repetition penalty 1.0.

## Choosing the default later

Use `marathon setup` to remember Inkling, or replace the selection document
`~/.config/marathon/selection.json` with this single line when the owner chooses it:

```json
{"schema":1,"model":"inkling-2-mini-ud-q4-k-xl","profile":"maya-262k","frontend":"codex","profile_policy":"explicit"}
```

This rollout does not make that change. Existing Majesty conversations keep their
selected model. Its model list refreshes when Majesty starts, using Marathon's
`model/list`; use Majesty's supported idle restart if new entries are missing.

## Evidence and rollback

The public synthetic evaluation is in
`~/AI/experiments/inkling2-mini-20261121/integration/REPORT.md` on AiGuy, with deployment
checks in `go-live/`. Two 150K+ tasks passed for Inkling, one for Qwen. A dense
309628-token Inkling input also passed; the same material exceeded Qwen's window.
The matched short cohort scored 8/10 versus 9/10, with median time ratio 0.68 across
joint successes. These are small synthetic cohorts, not proof of superiority on
arbitrary projects.

To roll back, wait until all affected workers and leases are idle, then restore
`go-live/config.before.yaml` to the watched broker YAML and
`go-live/catalog.before.toml` to the machine catalog. Qwen stays selected throughout.
Do not restart an active broker worker or delete shared lease files. The packaged
runtime README records code rollback separately; preserve unrelated local edits.


## Initial eager-cache candidate (superseded by lazy revision)

The native cache candidate lives in gpu-control branch `inkling/cache`. It restores
matching token prefixes inside the Maya engine. Marathon continues sending full
Responses history with `supports_slots = false`; do not enable Qwen slot APIs or
change the model catalog for this candidate. The Responses usage already carries
`input_tokens_details.cached_tokens`, and the runtime's timing wrapper carries
`timings.cache_n`. Marathon preserves both and counts only `prompt_n` as newly
prefilled tokens. Router lineage reconstructs full history from a known
`previous_response_id`; a new client can replay its full conversation directly.

Run the isolated probe from this checkout, against the candidate worker whose
GPU lease is held by the benchmark controller:

```sh
PYTHONPATH=. python3 scripts/evals/inkling_cache_probe.py \
  --upstream http://127.0.0.1:18393 \
  --output /home/deforest/AI/experiments/inkling2-mini-20261121/parity-cache/marathon \
  --prompt-file /path/to/synthetic-benchmark.txt
```

The output directory must be new. After `cold-done.json` appears, the controller
stops and restarts its own candidate worker, waits for readiness, then creates
`resume-now` in that output directory. The probe records a full replay and a
lineage continuation in `results.json`, including TTFT, engine timings, usage,
and generated answers. It creates its own temporary catalog, authenticated router,
and state directory; it never reads the installed catalog or private sessions,
and never starts or stops GPU workers. Stop the probe to remove its router.
The test `tests/test_inkling_cache_probe.py` checks these protocol guarantees with
a synthetic backend; native cache correctness and performance require live GPU
measurements. Runtime deployment and rollback belong to gpu-control; no Marathon
production code or catalog migration is required.


The isolated real Marathon probe on November 22 passed against the v2 native
candidate after its worker was stopped and restarted. The 111,122-token synthetic
request fell from 49.887 seconds to first token to 1.853 seconds (26.9x), restoring
all 111,122 tokens. Its `previous_response_id` continuation reused 111,293 of
111,305 tokens and reached first token in 1.627 seconds. Cold, restored, and
continuation answers were identical and correct: three hidden codes and the
integer sequence 1 through 64. Evidence:
`~/AI/experiments/inkling2-mini-20261121/parity-cache/marathon/results.json`.
The router subprocess was removed afterward. This proves the Marathon transport
path; broader native quality, disk limits, and promotion remain the runtime
candidate's separate qualification. The probe plus router context/accounting
suite passes 144 tests.


A controlled 200,980-token speed comparison produced the same correct 428-token
JSON answer on production and v2. Live compute-only prefill measured 1,940.5 vs
1,945.3 tok/s; decode measured 172.2 vs 173.4 tok/s. That matched-output sample
showed no compute-throughput regression. Persistence still has a first-use cost:
TTFT increased 103.895 -> 105.894 seconds (1.9%), and full response completion
increased 106.381 -> 113.364 seconds (6.6%) because completion waits for disk
persistence. The API's prefill timing includes state staging, unlike the live
compute-only counter. This is a small paired measurement, not a zero-overhead claim.

A separate 512-token prose pair measured 74.1 vs 70.4 tok/s but generated different
text, so it cannot isolate a runtime speed change. Restoring v2's prose reused all
200,979 tokens, reached first token in 3.228 seconds, and reproduced its cold
answer exactly at 70.5 tok/s. Speed evidence lives under
`~/AI/experiments/inkling2-mini-20261121/parity-cache/marathon-speed/`, including
`comparison-production-cache-v2.json` and `production/sequence-live.json`;
the candidate compute counter is in `parity-cache/v2-sequence-prefill-progress.json`.

## Lazy-persistence revision benchmark

**Combined candidate qualified (November 22):** gpu-control `inkling/cache` is rebased
onto `51741e5` and image `sha256:0f0cc4011268894c909a35100ae1b436ae421d7233a2191d64fd89dd7a0ea656`
(tag `marathon-inkling-maya:cache-combined`) keeps copy drafting, draft context 4096 and
the 344K preset. Against live `0f778b24`: 20 hot tool turns wrote 0 bytes (+9.7 ms
median), resume reused 98.33% at 15.4x faster TTFT, decode/prefill within 0.4% with
identical text, copy edit 1.65x, 344K VRAM unchanged, gate 60/63 both. The hold
below is resolved pending owner promotion; see
`~/AI/experiments/inkling2-mini-20261121/parity-cache/COMBINED-REPORT.md`.

**Promotion hold (November 22):** live gpu-control advanced independently to
`51741e5`, image `sha256:0f778b24ff07e8ebf82682075bfa026693163d1bfab93a8a481521321a59baa9`.
It now includes `copy-draft.patch`, `IK2_SUFFIX_BLOCK=1`, and a 4096-token draft
context (`d479239`), plus the revised 344K preset (`54a614b`). The cache image
`d9208d345bcf...` below was qualified against the supplied `b8ab` baseline with
16384-token draft context. **Do not promote it over current live.** Preserve the
new live features, build a combined pinned image, and qualify it against the actual
current production baseline before promotion. This work made no live changes.


`scripts/evals/inkling_agentic_cache_probe.py` measures twenty actual model-generated
`read_archive_chunk` function calls through an isolated Marathon router. Stable
tool definitions and instructions accompany every request. Synthetic tool results
grow context from about 100K to 150K; `previous_response_id` carries lineage.
The probe validates each call's name, integer argument, and call ID before returning
the next synthetic tool result. No private sessions or installed catalog are read.

Use `prepare --corpus <synthetic.json>` once against the isolated worker's CPU
`/tokenize` endpoint, then `run --corpus <synthetic.json> --output <new-phase-dir>
--engine-pid <native-host-pid> --cache-dir <dedicated-cache-directory>` for each
image. The parent benchmark controller holds GPU3's lease and owns worker lifecycle.
At `ready-to-unload.json`, stop and restart that worker, wait for readiness, then
write its new native host PID to `resume-now`. The probe sends the next tool-result
continuation through its still-running router and exits. Production's restarted
continuation supplies the cold comparator; the candidate must reuse at least 90%
and improve first-generation latency by at least 5x. `compare --baseline <dir>
--candidate <dir> --output <comparison.json> --unload-json <controller-measurement.json>` reports per-turn and median overhead.

Each turn records monotonic request boundaries, first visible generation-event
latency (`ttft_s`), completion latency, engine timings, and cached tokens. Marathon
buffers executable tool events until validation, so `first_activity_kind` identifies
when the first event is a completed function call rather than a token delta. Do not
label that measurement a direct native first-token timestamp. `/proc/PID/io`
`write_bytes`, `cancelled_write_bytes`, and logical `wchar` are separate counters;
reads never count as writes. Cache directory occupancy is recorded separately and
must not be summed as write traffic. The controller must capture unload-time writes
before the old PID exits; turn counters alone exclude shutdown persistence.

Marathon's HTTP client already allows 3600 seconds for the broker unload request.
The supervisor now allows at least 150 seconds for router shutdown when its pool
shares a lease namespace with a handoff worker, exceeding the broker's 120-second
unload timeout. This also covers starting on Qwen and later switching to Inkling.
Other backend and child-process timeouts remain unchanged. Rollback of this grace
is a code revert; no installed catalog change is required. The runtime's matching
95-second container-stop, 40-second active-request wrap, and 45-second native-QUIT grace must be deployed together.

A router started without a managed pool cannot acquire one merely by changing
model ID: its pool membership and lease namespace are established at startup.
It can forward an externally configured route, but owns no worker lease and does
not unload that worker on exit. The broker's TTL then controls persistence. The
longer supervisor grace therefore applies only to managed shared-pool lifecycles.


### Final lazy revision measurements

The final pinned image (including protection against eviction of an incoming
conversation during a full-store switch and selective re-save if its backing
snapshot is later evicted) passed a fresh 20-turn run on the isolated GPU3 worker with
stable tools and real function-call/tool-result lineage. All twenty calls and the
post-restart twenty-first call had the correct function and index. Context grew
from 100,726 to 148,826 tokens, then to 151,357 on the resumed continuation.

| Measurement | Production | Final lazy candidate |
| --- | ---: | ---: |
| First request completion | 43.380 s | 43.473 s |
| Hot-turn median completion | 1.863 s | 1.902 s |
| Kernel disk writes across 20 turns | 0 bytes | 0 bytes |
| Snapshots before unload | 0 | 0 |
| Next turn after worker restart | 72.621 s cold | 4.833 s restored |
| Tokens reused after restart | 0 | 148,826 / 151,357 (98.33%) |

Median **paired per-turn** overhead was 43.7 ms for first generation activity
and 43.9 ms for completion; first-use overhead was 92.8 ms (0.21%). The difference between the
two column medians is a different statistic. Hot overhead is small, not literally
zero. Function-call events are buffered for validation, as explained above.

The controller's actual Docker SIGTERM unload exited cleanly in 9.506 seconds and
created exactly one 2,611,378,536-byte snapshot. Native kernel `write_bytes` grew by
2,611,380,224 bytes during unload and by zero during all twenty turns and resume.
Reads were excluded. Snapshot writes occur **at unload**, not at request completion.
The restarted continuation was 15.03x faster than its production cold comparator.
No snapshots were rewritten during that continuation. This replaces the initial
eager revision's per-request staging and persistence costs.

Evidence: `~/AI/experiments/inkling2-mini-20261121/parity-cache/lazy/agentic/`, with
`production/results.json`, `lazy-qualified/results.json`, `qualified-unload.json`, and
`comparison-qualified.json`. The isolated router was removed after completion. The
shared-pool shutdown, twenty-turn protocol/write accounting, cache probe, router
context/accounting, and pool ownership suites pass 155 tests. Production settings
were untouched; promotion and rollback remain in the gpu-control runtime worktree.

Qualification image: `sha256:d9208d345bcf1632a334cdb91a0d5aa849e60d39fefdff1dc76841da08c92f86`.
The native binary revision is identified by SHA256 prefix `c88e6fcc`; gpu-control
records its complete hash and reproducible patch. Earlier lazy runs remain as
historical evidence, but the figures above use this final pinned image.
