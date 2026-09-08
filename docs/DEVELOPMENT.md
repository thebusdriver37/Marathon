# Development

Use the same checkout as normal users.
Machine-specific overrides belong in `~/.config/marathon/catalog.toml`, not in the shipped catalog.

## Checks

```bash
./bin/marathon setup-deps
.marathon/venv/bin/python -m unittest discover -s tests -v
```

The CI workflow runs shell syntax checks and the Python suite on Python 3.10 and 3.12.
Set `MARATHON_NETWORK_TESTS=1` to also test real pip bootstrap and interrupted-environment recovery; CI enables this check.
Native runtime patching and GPU benchmarks have separate requirements described in [runtime packaging](RUNTIME.md).
Python test success alone does not establish GPU correctness or speed.
Normal frontend installation builds and smoke-tests the binary without requiring developer test runners.
To run the native patch suite as part of a build, install a recent `just` and `cargo-nextest`, then run `MARATHON_CODEX_RUN_TESTS=1 ./bin/marathon build-codex`.
Both runtime patchers preserve upstream sources and existing worktree edits, and reuse unchanged patch stacks.

## Cached frontend builds

`marathon build-codex` reuses a managed source directory for each upstream revision and stores Cargo artifacts in `.marathon/codex-target-cache` by default.
Patch changes update only affected files, preserving unchanged dependencies and their timestamps.
The first managed build adopts the existing patched directory when possible.
Builds are serialized, and patch updates refuse to overwrite staged, unstaged, or untracked user edits.
A new upstream revision gets a separate source directory.
Release optimization and linking still take time after native code changes; a cold build still needs to compile dependencies.
`CARGO_TARGET_DIR` or `MARATHON_CODEX_TARGET_DIR` can override the artifact directory.
Direct calls to `scripts/apply_codex_patches.sh` retain immutable patch worktrees unless `MARATHON_PATCH_UPDATE=1` is set.

## Live user workflows

```bash
./bin/marathon eval smoke --run-gpu
# Also check allocation of all three workers and rejection of a fourth session:
./bin/marathon eval smoke --run-gpu --workers 3
```

This opt-in suite uses the installed frontend and remembered model through a real terminal and real inference.
It checks conversation memory and prompt-cache reuse, file editing and tests, background command cancellation and recovery, the printed resume command, and headless configuration overrides with shell tools.
The cache check requires the configured backend's slot-cache support.
The default uses one worker; the capacity check requires three available workers.
Existing workloads are never evicted.
Transcripts, session logs, router timings, fixture files, and `results.json` remain in the printed temporary directory for inspection.
Use `--output-dir /path/to/new-directory` to choose the evidence location.
The suite closes its sessions; inference workers follow their normal idle policy.
This suite is separate from the normal Python checks and needs no additional Python packages.

Keep first-run setup in the README and advanced configuration in [advanced usage](ADVANCED_USAGE.md).
Document actual tested support separately from expected hardware compatibility.
Preserve existing user selections and never change power caps or stop unrelated inference services during setup.
