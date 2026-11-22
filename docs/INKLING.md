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
