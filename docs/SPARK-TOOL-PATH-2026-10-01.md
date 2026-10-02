# Spark tool-path diagnosis, 2026-10-01

## Scope and result

Synthetic requests only; no private conversations read, generated commands executed,
production configuration changed, or live services restarted.

`scripts/evals/tool_path_probe.py` sent three fixed conversation checkpoints with
two seeds through three paths: TensorFold Chat Completions directly, the deployed
Responses shim, and an isolated instance of Marathon's actual router. Automatic
tool-protocol and stalled-response recovery were disabled in that test process.
All 18 requests returned structured `exec_command` calls, without leaked tool
markup or reported failures. Each Marathon request made one upstream request.

This is a transport/router probe with a small exec_command/write_stdin catalog,
not a full TUI session or a reconstruction of the user's long conversation.
Histories contain synthetic fixed tool results; tools are never executed. The
probe does not establish code correctness or eliminate long-context failures.

Receipts: `.marathon/diagnostics/spark-tool-path-20261001/`, including exact requests,
raw SSE responses, forwarded router payloads, upstream events, and summary.json.
The inspected checkpoint's forwarded payload differs from its shim control only
in model routing alias and stream=False; the router returns SSE to its client.

## Reproduced server-side fallback

The active server imports its parser from `tensorfold.cuda.reply_text`, not the
separate generic parser in `tensorfold.server.tools`.

Direct synthetic calls to the installed CUDA parser on Spark, repeated twice:

| Generated envelope | Structured call | Final content contains tool markup |
| --- | --- | --- |
| Complete offered exec_command | Yes | No |
| Complete unoffered write | No | Yes |
| Truncated exec_command without closing tags | No | Yes |

Mechanism in installed TensorFold source:

1. `cuda/reply_text.py:hide_tool_calls` hides envelopes during streaming.
2. `parse_tool_calls` matches complete envelopes only; unknown names remain text,
   and an unclosed envelope remains unmatched text.
3. Default `ToolCallPolicy` permits parallel calls and leaves that text unchanged.
4. `cuda/server.py:run` puts the leftover content into its final content delta.

Thus malformed tool text can be surfaced by TensorFold itself, before the shim
or Marathon. This reproduces the presentation mechanism, NOT the cause of the
model generating the malformed/truncated sequence in a real conversation.
The tests did not induce that generation spontaneously. Changing parallel-call
policy merely to hide malformed output would not establish or fix its cause.

## Runtime identity

Spark currently serves `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` via
`tensorfold-qwen38:v0.3.6.3`, with int8 KV, six MTP drafts, thinking enabled, and
262144 context. It is not the former iSkye NVFP4/vLLM stack, despite the catalog's
NVFP4 display label. Checkpoint, quantization and serving implementation changed
together, so attributing the change solely to Marathon is not supported.

Local and deployed Responses shim sources matched SHA256
`2c63a011a333488d9e168879f592264735e6ca9015712896410be2198ed47e0e`.

## Next diagnostic boundary

Extend this same synthetic probe toward longer histories and longer tool arguments,
capturing finish reasons, output limits and exact raw calls. If a malformed call is
generated, replay its exact request directly against TensorFold before changing
Marathon. Separate token-limit truncation, model/template mistakes, and parser
fallback; do not treat all three as one bug or add another alias-based repair.
