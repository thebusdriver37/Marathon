# Inkling 2 Mini (experimental)

This branch adds an Inkling model family to Marathon's GGUF discovery and model picker.
Place target weights under the configured model root and choose Inkling in setup.
The DSpark drafter remains a helper and is excluded from the picker.

Fast agentic evaluation uses Maya's full Python server through an external model
entry, with `model = "inkling-2-mini"`, `reasoning_transport = "responses"`, and
the backend slot API disabled. Its streaming Responses API works with the
ordinary Marathon frontend and router; Maya handles prefix reuse internally.
The standalone `runtime/maya-adapter.py` benchmark service only implements
nonstreaming chat and is a different interface. The unapplied external entry is
`integration/rollout/marathon-maya-entry.draft.toml` in the evaluation folder.
The fast engine/weights/configuration and DSpark correctness limits are recorded
in `integration/FAST-PROTOCOL.md` and `integration/REPORT.md` there. Keep its
physical GPU's existing Marathon pool lease through server cleanup.
The team's DSpark audit found changed complete greedy streams. Agentic results
with that drafter describe the selected deployment, not plain-Maya equivalence.
Enable Maya's `fit_max_tokens` so the output allowance fits the remaining
window without truncating the prompt. Put sampling defaults in its `sampling`
block or `.shared-settings.json`; a top-level `seed` is ignored. Verify `/props`
and native request headers. Maya's API `top_k=0` selects its widest 64-token
candidate shortlist, so matching API settings does not make its sampler
identical to Qwen's.

The installed frontend also has a 1,048,576-character limit on turn input.
A 302,858-token repository snapshot exceeded that character limit before model
inference, despite fitting the native Maya window. Context-token capacity alone
does not guarantee that an arbitrary full snapshot can enter the frontend.
Large text snapshot ingestion and cross-process snapshot/resume need separate
qualification before production use.

The local llama.cpp profiles below remain integration references; the fast
evaluation uses the external Maya server above. The `inkling2` backend expects
a native llama.cpp build at
`~/AI/backends/llama.cpp-inkling-29b52d39/build/bin/llama-server`.
This revision recognizes Inkling's Jinja template and native PEG tool/reasoning parser.
An older Qwen runtime cannot substitute for this build. Building or installing that
backend and obtaining the weights remain separate operator steps.
Use ggml-org IQ4_NL weights with that stock revision. Unsloth Mainline_Support
weights need the loop-pass expert-scale fix in
[llama.cpp #32704](https://github.com/ggml-org/llama.cpp/pull/32704), or the
qualified Unsloth build recorded in the evaluation. Do not mix those pairs.

The default `inkling-32k` profile uses 32768 context tokens, Q8 KV, one slot,
1024/64 batch/microbatch, and natural reasoning. `inkling-262k` uses 262144
context tokens and Q4 KV on a 24 GB card. Both are experimental; the 1M model
claim is not an advertised Marathon context setting. Use explicit device placement
and the normal pool lease before a manual backend launch.

The ordinary Codex frontend, structured patch tools, and base coding prompt are
unchanged. Marathon forwards Responses requests to llama.cpp. With `--jinja`,
llama.cpp parses `<|content_thinking|>` and `<|content_invoke_tool_json|>` into
structured reasoning and tool output. It also parses JSON-string tool arguments
into objects before rendering tool history. Do not hand-render Inkling tool markup
or replace the native template with a Qwen template.
Direct optional/required tool calls and a multi-turn replay of JSON-string API
arguments passed on both evaluated builds. General structured output is a
separate gate: use `response_format` with `json_schema`, since raw `grammar`
also constrains Inkling's message header
([#32745](https://github.com/ggml-org/llama.cpp/issues/32745)).

`reasoning_transport = "chat_template_effort"` forwards every advertised effort
through `chat_template_kwargs.reasoning_effort`. It preserves `none`: Inkling maps
that to 0, while omission defaults to 0.9. The template ignores `enable_thinking`. The router also removes the conflicting
OpenAI effort field: llama.cpp otherwise translates `none` into a boolean and
erases the template value, or overrides recovery with the original effort.
Lowest effort can still emit reasoning. Other template levels are minimal=.1,
low=.2, medium=.7, high=.9, and xhigh=.99. Recovery requests the lowest effort
using the same transport. There is no post-tool reasoning cutoff for these profiles.
The explicit template transport avoids
[llama.cpp #32753](https://github.com/ggml-org/llama.cpp/issues/32753);
rendered-prefix probes verified effort 0 for none and 0.7 for medium.
Changing effort changes the early system prefix and can require full prompt
prefill again. Keep effort stable during long conversations where possible;
lowest-effort stalled recovery may incur that cost.

For an existing external server, a machine-local `external_models` entry can use
the same `reasoning_transport`, explicit reasoning levels/default, and actual
server context size. Local family settings reach the router through
`MARATHON_MODEL_REASONING_TRANSPORT`. Image support needs a compatible projector
and separate validation; these text agent trials do not certify vision.

No production pool entry, default model, installed frontend, or broker is changed.
The isolated evaluation and un-applied rollout draft are under
`~/AI/experiments/inkling2-mini-20261121/integration/` on AiGuy.
