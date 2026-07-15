# Future updates

This file records worthwhile follow-up work that is intentionally outside the
current iteration. Items here are not active runtime behavior.

## Automatic GGUF metadata inspection

Allow an uncataloged GGUF model to get a safe, architecture-aware Marathon and
Dyno baseline without adding a hand-written family entry first.

The inspection pass should read metadata from the first GGUF shard and:

- identify architecture, quantization, trained context, layer count, dense or
  MoE layout, chat template, and shard completeness;
- select a compatible llama.cpp backend and reject unsupported architectures
  with a useful explanation;
- derive conservative context, batch, micro-batch, KV-cache, and GPU-placement
  defaults from the model metadata and detected machine resources;
- validate that the requested frontend is compatible, including Codex context
  requirements and the presence of a usable chat template;
- feed those facts into Dyno candidate generation while keeping failed-load and
  quality gates deterministic;
- cache the inspection by model fingerprint and invalidate it when a shard or
  backend changes.

Do not infer model quality from metadata, silently rewrite weights, or claim a
context length beyond the model's declared training context.
