# Future updates

This file records worthwhile follow-up work that is intentionally outside the
current iteration. Items here are not active runtime behavior.

## Deeper GGUF launch planning

Marathon now reads and caches each GGUF's embedded name, architecture, and
trained context before matching its catalog family.
The remaining work is to let an uncataloged model get a fully derived Marathon
and Dyno baseline without adding a hand-written family entry first.

The deeper planning pass should also:

- identify quantization, layer count, dense or MoE layout, chat template, and
  shard completeness;
- select a compatible generic or architecture-specific backend and reject
  unsupported architectures with a useful explanation;
- derive conservative context, batch, micro-batch, KV-cache, and GPU-placement
  defaults from the model metadata and detected machine resources;
- validate that the requested frontend is compatible, including Codex context
  requirements and the presence of a usable chat template;
- feed those facts into Dyno candidate generation while keeping failed-load and
  quality gates deterministic;
- cache derived launch plans by model and backend fingerprint.

Do not infer model quality from metadata, silently rewrite weights, or claim a
context length beyond the model's declared training context.
