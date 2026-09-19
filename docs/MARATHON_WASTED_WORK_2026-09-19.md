# Merge-only wasted-work probe

## Confirmed restricted-sandbox defect

The saved Swift merge parser and graph sessions repeatedly received a file-level `test failed` from `node --test`, without individual assertion details.
Replaying the same files through the installed frontend's actual sandbox reproduced the missing output; unrestricted execution showed all individual tests.
This is a restricted-network Linux sandbox defect, not evidence that the merge needs fine-tuning.
Marathon's default unrestricted frontend mode is not demonstrated to suffer this defect.

The problem also affects ordinary Node child processes: a child can exit successfully while both captured stdout and stderr are empty.
The restricted seccomp filter denied `getsockname` and `getsockopt(SOL_SOCKET, SO_TYPE)`, which libuv uses to classify socketpair-backed child standard streams.
System-call tracing identified both denials.
Allowing only `getsockname` was insufficient; the first candidate still failed the output regressions.

Patch `027-sandbox-child-output.patch` permits descriptor address inspection and only the `SO_TYPE` socket-option query.
Other socket-option queries remain denied, along with the existing restrictions on IP socket creation, connections, binding, and filesystem writes.
It does not disable sandboxing, alter commands, inject model hints, or change the model.

## Real Marathon CLI reproduction

The existing `scripts/evals/marathon_failure_probe.py` runner now includes a tiny `diagnostics` task.
It creates three Node tests, including one deliberately incorrect expectation, and asks the agent to report the exact failure without changing files.
The expected facts are two passing tests, one failure named `invoice total`, actual 42, expected 43.
Model interaction uses the installed `marathon exec` path, normal instructions and tools, and the Swift+uncensored IQ4_XS merge with the production inference settings copied into an isolated worker.
No baseline-model trials or training runs were used.

Before the patch, both fresh sessions eventually reported the correct facts but needed five shell calls each.
Both ran the suite, tried a different reporter, and finally executed the test file directly to recover the missing details.
Wall times were 36.0 and 19.0 seconds, including session startup; these are not steady-state benchmarks.
One response incorrectly taught that the missing details were normal Node reporter behavior, illustrating how broken tool evidence can lead to a misleading explanation.

Artifacts are under `/home/deforest/AI/experiments/marathon-wasted-work-20260919/`.
The fixture is intentionally small to isolate tool-output fidelity, not to measure general coding intelligence.

Two matched fresh sessions using the rebuilt candidate frontend both returned the correct facts with three shell calls each instead of five.
Both obtained the individual test details from their first `node --test` invocation, with no alternate-reporter or direct-file recovery calls.
Both preserved the fixture and created no additional files.
Wall times were 34.0 and 13.0 seconds; startup, unseeded sampling, concurrent native compilation, and the tiny sample prevent treating these timings as a general speed benchmark.
The defensible improvement is removal of two unnecessary recovery calls per session in this test.

Successful post-fix evidence is under `after-fix-ready/swift/` and records an explicit candidate frontend path while still launching through `marathon exec`.
The preceding `after-fix/` attempts stopped at launch because the candidate lacked its required feature-metadata sidecar; they made no model calls and are not counted as model trials.
The sidecar was supplied only after verifying that the candidate was built from Marathon's hardened patched source.
All experiment workers were released and their disposable slot snapshots removed; source, transcripts, and results remain.

## Verification

The three checks in `tests/test_sandbox_diagnostics.py` exercise a real native sandbox, not mocked tool results.
They cover individual Node test details, captured child stdout/stderr, and successful descriptor inspection while network connections and outside-workspace writes remain denied.
All three fail against the old installed frontend and pass against both the patched standalone sandbox binary and the rebuilt full frontend.
The ordinary Python suite ran 430 tests successfully with 10 optional checks skipped; the three new native checks were also run explicitly.
The native sandbox library suite passed all 110 tests using `just test --release -p codex-linux-sandbox --lib`.
The full frontend was rebuilt and installed through the normal atomic installer, with the previous binary and metadata preserved for rollback.
The installed build is `3d2ee51ca2d5db578f328aa75e20aa22c0197c9a:3a019cbf7a33331e8585ea8a07c04dc12a12d2e9`.
The previous patch manifest matched the current patch stack excluding patch 027, confirming that the native comparison changes only this patch.
Existing sessions retain their running frontend; newly started Marathon sessions use the installed fix.

To run the native checks against an installed frontend:

```bash
MARATHON_SANDBOX_TEST_BIN=/absolute/path/to/codex \
  python3 -m unittest discover -s tests -p test_sandbox_diagnostics.py -v
```

Developer builds with `MARATHON_CODEX_RUN_TESTS=1` also run these checks against the candidate frontend before installing it.
Node-dependent checks skip when Node is unavailable; the descriptor and security check still runs on Linux.

## Build-time waste

During the rebuild, Cargo also compiled and optimized the CLI package's `logs_client` developer executable.
Marathon installs only `codex`, so `scripts/build_codex.sh` now selects `--bin codex` explicitly.
This removes an unnecessary binary build without reducing optimization or changing the installed frontend's dependencies.
No build-time percentage improvement is claimed; a matched timing comparison was not performed.

## Remaining lead

The older merge graph session also spent multiple calls searching for shell web programs before using the already-available managed web tools.
That remains a separate discovery/instruction-following lead, not a confirmed router defect or a reason to change model weights.
This probe deliberately concentrates on the reproducible output-loss defect instead of adding broad stop-early prompts or hiding failed commands.

Review also corrected the earlier graph report: the merge fixed the CLI's 49-versus-48 service discrepancy before its timeout.
Independent execution of the saved final CLI reports 48, so that discrepancy is no longer an unresolved final-code failure.
