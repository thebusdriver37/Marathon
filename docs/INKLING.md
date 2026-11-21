# Inkling 2 Mini (experimental)

This branch adds an Inkling model family to Marathon's GGUF discovery and model picker.
Place target weights under the configured model root and choose Inkling in setup.
The DSpark drafter remains a helper and is excluded from the picker.

The `inkling2` backend expects a native llama.cpp build at
`~/AI/backends/llama.cpp-inkling-29b52d39/build/bin/llama-server`.
This revision recognizes Inkling's Jinja template and native PEG tool/reasoning parser.
An older Qwen runtime cannot substitute for this build. Building or installing that
backend and obtaining the weights remain separate operator steps.

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

`reasoning_transport = "chat_template_effort"` forwards every advertised effort
through `chat_template_kwargs.reasoning_effort`. It preserves `none`: Inkling maps
that to 0, while omission defaults to 0.9. The template ignores `enable_thinking`.
Lowest effort can still emit reasoning. Other template levels are minimal=.1,
low=.2, medium=.7, high=.9, and xhigh=.99. Recovery requests the lowest effort
using the same transport. There is no post-tool reasoning cutoff for these profiles.

For an existing external server, a machine-local `external_models` entry can use
the same `reasoning_transport`, explicit reasoning levels/default, and actual
server context size. Local family settings reach the router through
`MARATHON_MODEL_REASONING_TRANSPORT`. Image support needs a compatible projector
and separate validation; these text agent trials do not certify vision.

No production pool entry, default model, installed frontend, or broker is changed.
The isolated evaluation and un-applied rollout draft are under
`~/AI/experiments/inkling2-mini-20261121/integration/` on AiGuy.
