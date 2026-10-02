# Session-history retrieval compaction trial

## Local integration update

At the user's request, this is now wired into normal Marathon launches, not a
separate trial command. `run_codex` registers the read-only `marathon_history`
MCP server and the packaged `marathon_app/compaction.md` handoff prompt. Explicit
CLI overrides still take precedence. `MARATHON_HISTORY_ENABLED=0` is the rollback
switch. No frontend rebuild, broker restart, or duplicate transcript database is
required. Nothing has been pushed or released.

The server selects only the current conversation's journal using the frontend's
MCP `_meta.threadId`, never a model-supplied path or conversation ID. New sessions
and resumed sessions use the same mechanism. Missing or ambiguous identity fails
closed. Normal router processing now translates MCP namespaces and restores them
in responses for both HTTP and WebSocket paths; the extra benchmark gateway is
not used. The catalog no longer advertises unsupported deferred tool search.

The historical measurements and trial limitations below are retained unchanged.
The integrated synthetic end-to-end receipts are under
`.marathon/diagnostics/lcm-20261001-integrated/`, launched with the maintained
evaluation runner's `--native-defaults` mode, which also restarts and resumes the
actual frontend. Personal conversations are not test inputs. The 23-second figure
is a measured average with the benchmark's medium compaction effort, not a latency
guarantee for every model, reasoning setting, or history size.

Integration verification: the normal-router run completed two compactions in
25.4 and 30.7 seconds, recovered all six queried records from 90 observed records,
passed the explicit-schema Python configuration check, and successfully queried
the same journal after an actual frontend restart/resume (6.0-second resumed
turn). No benchmark gateway was involved. Archive, router-context, launcher,
security, session-routing, remote, compaction-cache, and swarm test suites passed
(293 passing tests, two optional swarm tests skipped). The local launcher resolves
to this source checkout, so closing and reopening Marathon loads the integration
without rebuilding Rust. The existing active session was not restarted.

## Scope

Opt-in, LCM-inspired proof of concept, not a full LCM implementation or a production default. Existing Marathon/Codex native compaction still produces the summary. A small read-only MCP service can search and page through the original journal of exactly one isolated session. No embeddings, second transcript database, external inference, or private-session access.

Production source/configuration and the running Spark session were not changed. Tests used the promoted local Swift-Qwen3.8-27B-Uncensored-Merge-IQ4_XS model, medium reasoning, and two broker-managed 3090 workers. GPU 1 remained leased elsewhere and was not interrupted.

## Frozen three-arm comparison

Receipts: `.marathon/diagnostics/lcm-20261001-validated/`.

Two fresh sessions per arm, 180 synthetic service records per session, three native `/compact` operations per session. Each recall asks for three previously observed records, each containing four exact fields, including unpredictable random values. A record passes only when all four fields match. Every run first verified that the complete inventory reached the journal.

| Arm | Exact records recovered | Policy/unknown checks | Compaction + recall seconds, runs 1 / 2 |
| --- | --- | --- | --- |
| Stock summary | 2/18 | 24/24 | 101.4 / 116.2 |
| Structured summary | 3/18 | 24/24 | 183.2 / 302.5 |
| Structured summary + retrieval | 18/18 | 24/24 | 109.7 / 155.9 |

Retrieval recovered a later authorized record correction in both sessions. The quoted hostile instruction to enable deployment/delete backups did not change the tested policies. Unknown signing secrets remained null. These are narrowly scoped checks, not a comprehensive prompt-injection guarantee.

JSON-only formatting was imperfect: stock 2/6, structured 3/6, retrieval 3/6. Factual checks and formatting are scored separately. Several non-retrieval recalls attempted prohibited shell recovery; filesystem isolation prevented reading the withdrawn inventory or the outside canary. Retrieval used its dedicated tools, with no prohibited shell calls in these six recalls.

### Coding-check qualification

The initial continuation prompt did not explicitly distinguish a flat dictionary from a dictionary nested under the service ID. The strict grader expected flat output. Its original results remain unchanged: stock 0/2, structured 0/2, retrieval 1/2. Inspection shows correct underlying record values in both retrieval outputs and one structured output; nesting accounts for two strict failures. Do not present this ambiguous subtest as a clean coding-quality comparison. The larger follow-up explicitly requires exactly seven top-level keys.

## Controls and limitations

- Actual native Marathon TUI and native `/compact`, not a fabricated summarization-only harness. Headless PTY captures are protocol evidence, not visual screenshots.
- All arms use the same existing single-worker namespace gateway and trial catalog with deferred tool search disabled. This is necessary for local MCP tools to reach the model. It is not a comparison against completely unmodified production routing.
- Same per-repeat inventory, selected records, model, quantization, context, and reasoning setting. Arm order rotates across repeats. Generation is stochastic; no claim of deterministic matched sampling or statistically established speed improvement.
- Retrieval has an extra tool-connection preflight and extra tool definitions. The timing table excludes setup/preflight/coding and includes only compaction plus recall. Complete wall times remain in raw results.
- Original inventory files are withdrawn from the sandbox after observation. Gold answers and originals remain outside the model's accessible workspace. This deliberately tests facts available only in history; real projects often retain searchable source files.
- Router tool-output allowance is raised equally to 180,000 characters for the experiment. This isolates compaction from earlier tool-output truncation. Retrieval cannot restore evidence never written to the journal.
- Active constraints are repeated/preserved in user messages, so their success does not establish an advantage from retrieval.
- Only synthetic exact-memory recovery and a small continuation task are evaluated. No claim about general coding quality, conversation style, automatic long-running sessions, or maximum-context reliability.
- No hierarchical summary DAG, production session binding, resume integration, retention policy, or automatic rollout is implemented.

## Larger follow-up

Receipts: `.marathon/diagnostics/lcm-20261001-large/`. One fresh session per arm, 600 records, five successive native compactions, 15 queried records. This uses the archive hardening and explicit flat-dictionary code contract described below.

| Measure | Stock | Summary + retrieval |
| --- | --- | --- |
| Exact records recovered | 1/15 | 15/15 |
| Policy/unknown checks | 20/20 | 20/20 |
| Compaction time | 64.2 s | 117.2 s |
| Recall time | 87.7 s | 135.7 s |
| Full run wall time | 245.0 s | 338.0 s |
| Model requests | 26 | 42 |
| Final explicit-schema code check | Fail: unknown record fields | Pass |
| JSON-only recall responses | 2/5 | 3/5 |

All runs completed without a turn timeout. Retrieval used no prohibited shell calls during recall and recovered the later corrected record. Stock retained the corrected record but lost the other fourteen queried records. The final retrieval-generated Python configuration contained the exact requested values and policy settings; stock used null/None for the missing record fields.

This is a fidelity improvement with measurable cost, not a speed improvement: compaction plus recall took approximately 253 seconds with retrieval versus 152 seconds without it. Stock often finished sooner by reporting that the original details were unavailable. Do not compare those times as equivalent completed work.

Across these two versions, retrieval recovered all 33 requested records, but the 18-record and 15-record results should remain separate because the archive and code-test wording changed. These are deliberately retrieval-friendly synthetic tasks, not 33 independent real-world project evaluations.

Recommendation: proceed toward a carefully isolated opt-in session-history feature, not an unconditional production rollout. The experiment supports recoverable original evidence as the useful mechanism. A better summary alone did not preserve arbitrary omitted details. Next acceptance gates should include real multi-turn coding workloads, automatic compaction/resume, full session binding, tool-contract integration, and bounded retention/search cost.

## Verification

- 12 archive unit tests pass, including original-output pagination, stable IDs, source labels, corrupt/oversized input, traversal rejection, symlinks/replacement, FIFO rejection, retrieval-echo exclusion, stdio protocol, and separate-session isolation.
- 6 existing compaction-cache tests pass.
- 123 existing router-context tests pass.
- Larger-run source and prompt SHA256 hashes still match its saved protocol after completion.
- On one completed synthetic 831 KB journal, 30 in-process searches took a median 6.1 ms (maximum 8.02 ms). This excludes MCP transport/model time and says nothing about much larger journals. Lookup/model round trips, not this small CPU sample, dominate the observed live cost.

## Integration findings

The trial exposed three tool-plumbing requirements:

1. Local routing needs namespace translation for MCP tools. The experiment reuses `SwarmGateway` with one worker, identically for all arms.
2. The advertised deferred tool-search capability does not match this local router's supported tool types. The isolated trial catalog disables it so history tools are declared directly.
3. Under approval policy `never`, the read-only history MCP server needs explicit tool approval configuration. A successful function-call event alone is insufficient: the preflight also asserts returned evidence.

These changes are confined to trial launch configuration. Production adoption must solve the tool contract cleanly rather than copying benchmark launch overrides indiscriminately.

## Archive boundaries

`marathon_app/history_archive.py` binds to one journal in an explicitly isolated session root and fails closed if there are multiple journals, a replaced inode, a symlinked journal, an oversized record/file, or a non-regular file. IDs use original physical line numbers. Search results carry role/provenance, lossy-summary labeling, and bounded excerpts; reads are paginated. Reasoning, developer messages, and session metadata are not returned.

After the first comparison, two hardening changes were made before the larger follow-up: nonblocking open rejects FIFOs without hanging; the service skips its own tool calls/results when indexing, avoiding circular search evidence. Tests cover both. Benchmark protocol files record source/prompt SHA256 hashes, so these versions must not be silently pooled as identical treatments.

The root is trusted and explicitly supplied by the launcher. This is not a general-purpose filesystem security boundary or an API for accepting arbitrary user-selected journal roots.

## Reproduction and cleanup

Run from the Marathon repository with available broker workers:

```sh
.marathon/venv/bin/python -m unittest discover -s tests -p test_history_archive.py
.marathon/venv/bin/python scripts/evals/lcm_trial.py --run-gpu --output-dir .marathon/diagnostics/lcm-fresh --records 180 --cycles 3 --repeats 2 --workers 2 --arms retrieval stock structured --timeout 240
```

The output directory must not exist. Current source includes the explicit flat-dictionary continuation instruction and archive hardening described above. Each arm gets its own session home, synthetic project, expected answers, runtime logs, terminal capture, and results. Runtime cleanup releases its own leases. No personal sessions should be stopped to make room.

Development `lcm-20261001-smoke*`, `-matrix`, and `-main` directories contain setup failures and invalid preliminary comparisons (tool visibility/approval and sandbox-isolation issues). They are not evidence for the final result. Retain the validated/large receipts and this report; obsolete synthetic development receipts can be removed separately with approval. Nothing is automatically deleted.

Final state: all experiment processes exited. Worker 2 and worker 3 were exclusively leased for cleanup, unloaded through the broker, and released; both returned to 15 MiB GPU memory. Worker 1 and the live Spark session were untouched. Changes consist only of this report, the archive service, its tests, the summary prompt, and the evaluation runner; they remain uncommitted and are not enabled in normal Marathon.
