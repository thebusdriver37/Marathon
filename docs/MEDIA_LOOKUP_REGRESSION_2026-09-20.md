# Image-history speed regression, 2026-09-20

The affected conversation slowed from 69 to 21 tokens/second when image content first entered its history at 67866 input tokens.
The combined lookup/DFlash media-capability check disabled all drafting; subsequent turns remained near 19-21 tokens/second with zero drafted tokens.
The defect was reproduced with a synthetic image, repaired, and deployed at an idle request boundary.
At 73K occupied tokens, the isolated candidate produced image descriptions at 56-65 tokens/second, while all 12 existing quality tasks passed.
A fresh image request through Marathon's normal route verified the deployment at 62.29 tokens/second with active speculation.
Old speculative snapshots rebuild once through existing restore-error recovery to avoid stale drafter state; conversation history is retained.
The full source, validation, migration, and rollback record is [the central deployment document](/home/deforest/Documents/DEV/gpu-control/runtime/qwen38-media-lookup-deployment.md).
Local artifacts are in `.marathon/diagnostics/media-fallback-20260920`.
No private conversation text was copied or transmitted; the retained affected-session record contains numerical metadata only.
