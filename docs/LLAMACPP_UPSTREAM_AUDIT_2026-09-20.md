# Bounded llama.cpp upstream compatibility audit

Subsequent work completed a separate candidate build and a small real-Marathon comparison; see [the comparison report](LLAMACPP_MARATHON_COMPARISON_2026-09-20.md).
The statements below describe the earlier audit stage.

Production remains unchanged.
The upstream review and patch-compatibility probe are complete; no updated runtime was built or deployed.
A full port needs manual CUDA and server conflict resolution, which exceeds the requested short audit.
Three draft backports are prepared for separate qualification instead of silently replacing working runtime behavior.

## Revisions and actual rebase attempt

Production descends from upstream `9723942adc518b43c4b95dc4dce6906903eb5e09` dated August 31, followed by seven private commits through `ca461b488709b74fb7a59211fa4c75b8643e5601` and later runtime overlays.
Upstream `ce8caa6e60a03093351d6016a818720e0d46f0fb` dated September 20 is 354 commits ahead of that upstream base.
The separate general-backend pin in `config/llamacpp.ref` does not identify this production runtime.
The currently deployed image remains `sha256:8af4fa77e493f6765b7c66d4f6cbfb673e0add0919b470c11526e08d54365e04`.

A separate local Git checkout was created under `.marathon/diagnostics/upstream-audit-20260920/source` and fetched the exact upstream revision.
No shared source checkout was reset or modified.
Patch 001, the IQ4_XS crossover change, applied cleanly and was committed in that disposable audit checkout.
Patch 002 failed both the ordinary application check and an actual three-way application, leaving an attention-header conflict in the isolated checkout.
Its MMVQ and top-k portions applied in the three-way attempt.
The checkout is intentionally an unfinished audit candidate, not a buildable runtime or production source.

Additional forward checks found overlaps in DFlash injection, attention swizzling/dequantization, scheduler reuse, checkpoints, and media verification.
Those later checks were performed after the patch-002 dependency stopped the sequence and must not be interpreted as independent proven semantic conflicts.
Some mechanisms have upstream equivalents, so blindly retaining every local patch would also risk duplication.
The portable stack is not by itself a byte-for-byte recipe for every later production overlay; the deployed common/server libraries and compact Q8 translation unit must be accounted for separately.

## Relevant upstream changes

| Change | Finding in local deployed-source records | Action |
| --- | --- | --- |
| [GDN normalization, #28068](https://github.com/ggml-org/llama.cpp/pull/28068) | Qwen35 graph still calls the old `ggml_l2_norm` implementation | Highest-priority correctness candidate; draft limited to the local Qwen35 graph |
| [Checkpoint eviction, #28302](https://github.com/ggml-org/llama.cpp/pull/28302) | Recorded production server source applies spacing eviction before the list is full | Draft adapted to preserve our checkpoint-buffer reuse |
| [Convergent attention barrier, #27870](https://github.com/ggml-org/llama.cpp/pull/27870) | Local compact header retains separate branch-local barriers | Draft uses upstream's single convergent barrier structure |
| [Speculation after images, #28715](https://github.com/ggml-org/llama.cpp/pull/28715) | Production server already supplies `prompt.tokens.pos_next()` to drafting | Existing local fix overlaps the upstream mechanism; do not count it as a new speedup |

The local attention code already contains an alternate-branch synchronization workaround.
Its structure differs from the upstream convergent fix, but this audit did not reproduce a synchronization failure on our current workload.
Absence of that upstream patch is not proof that our measured configuration is broken.
Likewise, the checkpoint source finding does not establish that every historical slow resume had this cause.

## Draft backports

Draft files are in `scripts/experiments/upstream-backports-20260920/`.
They are deliberately outside the automatic runtime patch stack.

- `01-qwen-gdn-normalization.patch` adopts upstream's RMS-based formulation for the Qwen35 graph only.
- `02-checkpoint-eviction.patch` gates spacing eviction on checkpoint capacity and replaces duplicate positions through the existing reusable-buffer pool.
- `03-attention-convergent-barrier.patch` transplants the upstream metadata-combine barrier block into the qualified compact Q8 header, retaining compact staging elsewhere.

All three pass `git apply --check` against isolated copies of their recorded local source files.
The copied inputs, source hashes, forward-check results, and partial three-way rebase remain in the ignored audit directory.
These are source applicability checks only, not compile, sanitizer, inference, or performance validation.
No GPU workload, broker configuration, model, runtime image, or live session was changed.

## Recommended next qualification

Test the normalization correction separately before investing in a complete upstream port.
It intentionally changes arithmetic, so bit-identical replies to the old implementation are not a valid correctness requirement for this particular fix.
Use a trusted reference or task-quality evaluation and matched performance measurements, retaining the same weights and context capacity.
Use fresh target/draft cache state and a separate cache identity during qualification; do not restore old recurrent/KV snapshots across changed model arithmetic.

Checkpoint qualification should first reproduce short-prompt rewind eviction, then verify retained checkpoints, buffer reuse, snapshot restoration, and bounded memory.
Attention qualification should include synchronization checking on an `np > 1` case plus existing numerical and throughput fixtures.
None of these source changes is currently a demonstrated speed improvement.
A complete upstream update remains a separate manual port-and-validation task, not a safe one-command merge.
