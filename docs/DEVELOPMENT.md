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

Keep first-run setup in the README and advanced configuration in [advanced usage](ADVANCED_USAGE.md).
Document actual tested support separately from expected hardware compatibility.
Preserve existing user selections and never change power caps or stop unrelated inference services during setup.
