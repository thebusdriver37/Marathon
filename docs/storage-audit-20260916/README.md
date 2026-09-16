# Storage retention audit, 2026-09-16

## Policy

Preserve research evidence, trained parameters, captured target features, source modifications, and the shared Codex compiler cache.
Move only historical inference caches, verified duplicate plugin checkouts, and unused experimental build outputs to desktop Trash.
Trash is on the same filesystem, so these moves do not reclaim free space until Trash is emptied.
No inference service or container is restarted or removed.

## Cleanup manifest

[manifest.json](manifest.json) lists each exact original path, size, reason, and completion status.
The original-path totals are 33.75 GiB of GPU Control benchmark inference snapshots, 5.61 GiB of duplicate plugin checkouts, and 0.90 GiB of unused llama.cpp builds.
The 97 entries total approximately 40.27 GiB.
Retained plugin checkout locations and exact revisions are recorded in the manifest.
The benchmark snapshots are runtime KV/recurrent-state caches, not trained model checkpoints or recorded benchmark measurements.
Removing them prevents exact cache-state restoration for those historical runs; logs, configuration, results, and cache metadata remain available to interpret the measurements and rerun the experiments.

## Research retained

GPU Control benchmark scripts, fixtures, summaries, raw measurements, logs, provenance, and slot metadata remain in their original locations.
Marathon diagnostics retain reports, verification results, model variants, logs, and generated test workspaces.
No conversation contents were reviewed as part of this cleanup.
[preserved-files.json](preserved-files.json) inventories retained files inside affected experiment directories and records their sizes and modification times, which were checked after cleanup.
This is a metadata preservation check, not a content checksum verification.

The approximately 192 GiB drafter-training tree remains intact except for any duplicate plugin cache entries explicitly listed in the manifest.
It contains roughly 87 GiB across the main captured feature sets, learned checkpoint weights, adapters, corpus work, and the training runtime.
These are products of paid compute, and prose summaries cannot reproduce their exact values.
They should not be deleted merely because a results document exists.

## Source and build retention

[sources.json](sources.json) records local experimental source revisions and untracked-file inventories.
Sibling binary-capable Git patch files preserve tracked modifications relative to each recorded HEAD.
Original source trees remain available, including untracked files; these patches are supplemental provenance, not a complete replacement for the trees.
Small source worktrees were retained because their savings are minor and they preserve the exact code behind past experiments.
The IQ4 llama.cpp source is also mounted by an existing build container.

The removed builds are `llama.cpp-mtp-build`, `llama.cpp-next-build`, and `llama.cpp-checkpoint-opt-build` under Marathon's `.marathon` directory.
Each has its CMake cache and available compile commands/version file archived in a sibling folder here.
Their corresponding source trees remain in place.
Rebuilding requires the recorded compiler/CUDA environment and configuring the retained source with the recorded CMake options; the CMake cache is provenance, not a portable executable build recipe.
The `.marathon/llama.cpp-build` directory remains because repository launcher scripts use it as a fallback.
Configured backends under `~/AI/backends`, Docker images, live inference caches, installed Codex binaries, and the rollback executable are untouched.
The 61 GiB shared Codex compiler cache is retained at the user's explicit request.
The five old Codex target directories moved to Trash in the preceding cleanup are not included in this audit's 40.27 GiB total.

## Restore

Use the desktop Trash Restore action for the original path listed in the manifest.
For duplicate plugin checkouts, the manifest also identifies a retained equivalent revision.
Do not restore over a newly created build or cache directory without checking its contents first.
The audit deliberately does not empty Trash.
