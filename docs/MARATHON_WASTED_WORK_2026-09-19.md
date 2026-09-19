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

## Second pass: mixed web and frontend tool ordering

Four fresh merge-only real CLI sessions tested official-documentation lookup with local execution: two Node assertion questions and two Python TaskGroup debugging questions.
The earlier detour through shell web utilities did not recur in this screen; managed web tools were available and used.
Instead, the mixed-tool sessions exposed a router sequencing defect.
When a model response requested both a router-managed web action and a frontend-executed shell command, `_run_responses_loop` executed the web action and continued inference before the frontend's shell result could enter the next model request.
Streaming could show a completed shell command to the user while the model still lacked its result.
The first Python session issued six shell calls before their results appeared in its rollout, repeatedly asking to read the same file and claiming output was missing.
That session ultimately used ten shell calls; the second Python session used six.
The two Node sessions used four and two shell calls.
All four eventually completed with the core answers correct, so this is avoidable recovery work rather than four failed tasks.

The Python router now completes managed web actions and yields to the frontend whenever that response also contains frontend work.
It does not request another model response until the frontend returns its tool results through the normal continuation path.
The same rule applies when the web budget is exhausted; budget handling must not force a premature final answer while shell results remain pending.
Web-only continuation, bounded web retries, and reconnect caching remain in place.
No tool permissions, sampling settings, model weights, or native frontend code were changed in this pass.

The new regression test failed in all four streaming/non-streaming and normal/exhausted-budget combinations before the fix.
It now verifies that no premature inference occurs, completed web results and frontend calls survive reconnect replay, and the subsequent request contains both web evidence and the actual local result.
All 140 router tests pass; the complete Python suite ran 431 tests successfully with 10 optional checks skipped.
Artifacts, exact prompts, actual CLI transcripts, web results, and router hashes are retained under `/home/deforest/AI/experiments/marathon-web-discovery-20260919/`.

Four matched fresh post-fix sessions all completed and answered the core questions correctly, with unchanged repository fixtures and no extra repository files.
The Node task used one shell call in each repetition, down from four and two.
The Python task used two shell calls in each repetition, down from ten and six.
Each post-fix rollout had at most one pending shell result, compared with peaks of two, two, six, and three before the fix.
The repeated file-read recovery sequence did not recur.
Post-fix times were 49.1/31.0 seconds for Node and 37.0/24.0 seconds for Python, compared with 113.1/56.1 and 136.2/47.1 before.
These are small unseeded samples with live network retrieval, differing documentation-fetch choices, and cold-start overhead; they do not establish a universal speedup percentage.

One post-fix answer included an inaccurate extra explanation saying ValueError is an Exception but not a BaseException.
It inherits from both; the answer correctly identified ExceptionGroup as the container for the tested ValueError scenario.
This isolated wording error remains a model-quality observation, not a demonstrated recurring defect or justification for training.

The production GPU configuration remained unchanged, the isolated merge worker was unloaded, and disposable experiment snapshots were removed while retaining the evidence.
New Marathon router processes load this Python fix; existing running routers need a restart to use it.

## Whole-router review: four reproduced transport/recovery defects

`tests/test_router_transport.py` uses real local HTTP/WebSocket sockets, the actual router handlers, and a scripted upstream instead of GPU inference.
Before implementation, each defect reproduced in two different scenarios, each repeated with fresh servers twice: 16 failing subcases in total.
These prove reachable router failures, not their frequency with the Swift merge or a model-quality improvement.

| Defect | Independent scenarios | Fix |
| --- | --- | --- |
| Recovery published calls it subsequently forgot | Valid shell then malformed patch; valid patch then malformed patch | Hold executable tool events until the backend attempt completes and passes protocol validation; continue streaming text/reasoning. |
| Recovery kept forcing tools after success | Empty response recovery; malformed patch recovery, both followed by a successful web fetch | Restore the caller's original tool choice after the requested recovery action succeeds. |
| HTTP fallback lost web execution and patch translation | Non-streaming JSON; streaming SSE | Route Responses through the shared generation/tool/history implementation and translate buffered patches too. |
| Keepalives corrupted upstream SSE | Pause inside a JSON string; pause between two data lines of one event | Buffer incomplete upstream frames and insert keepalives only between complete frames. |

All 16 subcases pass after the fixes.
The installed Marathon native CLI was additionally tested over HTTP in two isolated temporary workspaces, with its normal model catalog/tool definitions and workspace-write sandbox.
Both sessions fetched controlled web evidence, successfully created the requested file through native apply_patch, and returned the patch result on the subsequent model request.
This exposed a related HTTP history problem: the frontend strips web-item IDs when replaying its full history.
The router now restores matching managed-web evidence from its existing model/thread-scoped lineage.
ID-less markers are restored only when the action identifies an unambiguous result; ambiguous or foreign-thread history is never guessed.
Both native CLI tests also assert that the next model request retains the fetched evidence.

Header-based HTTP compaction, string input, and thread-isolated history restoration have additional regression coverage.
All 147 router tests pass with installed-CLI coverage enabled.
The complete Python suite runs 438 tests successfully with 10 unrelated optional checks skipped.
To repeat the router checks, run `MARATHON_ROUTER_TEST_BIN=/home/deforest/.local/share/marathon/bin/codex .marathon/venv/bin/python -m unittest discover -s tests -p 'test_router*.py'`.
No GPU services, model weights, sandbox permissions, or native binaries were changed.
Restart an existing Marathon session to load the Python fixes; a frontend rebuild is unnecessary.
