# Experimental local swarm

This branch adds `marathon swarm` for one lead agent and two persistent helpers using the same local Qwen model.
Codex V2 owns agent spawning, messaging, waiting, and conversation state.
Marathon leases GPU workers and routes each Codex thread to a stable worker.
No Codex Rust changes or broker configuration changes are required.

## Try it

Select your existing llama-swap pool profile in Marathon on Linux first.
The default experiment requires three available pool leases.
It never takes over existing Marathon sessions.

```bash
marathon swarm
marathon swarm exec "Split this task between two helpers, integrate their work, and run tests."
```

Run these commands from the project you want to work on.
The lead assigns disjoint files or separate Git worktrees before parallel edits.
Worktrees are not created automatically, so this is still a shared-workspace experiment unless the agents explicitly create them.
The normal approval and sandbox configuration remains in effect.

To compare the same three-agent workflow on one GPU, or try it while only one pool lease is free:

```bash
marathon swarm --workers 1 exec "Your task"
```

`--agents 2` selects a lead and one helper instead.
Worker count defaults to agent count.
Each command starts a new team with an isolated Codex home.
Reuse the helpers through messages and follow-up tasks; the gateway does not recycle thread assignments during a session.
Attaching existing conversations and resuming a saved swarm are not implemented.

## Validation

```bash
.marathon/venv/bin/python -m unittest discover -s tests -p test_swarm.py -v
.marathon/venv/bin/python scripts/evals/swarm.py --run-gpu --workers 1
.marathon/venv/bin/python scripts/evals/swarm.py --run-gpu --workers 3
```

The coding evaluation assigns two modules to helpers and an integration module to the lead.
It checks the resulting code independently, verifies three distinct agent threads, and reports concurrent requests across workers.
Concurrent requests establish routing concurrency, not simultaneous GPU kernel execution or a guaranteed speedup.
Compare elapsed time only across successful runs, and repeat runs before drawing performance conclusions.

The initial live baseline passed with three native Codex agents sharing one Qwen worker in 115.32 seconds, including startup and cleanup.
All three independent coding tests passed and the original test file was preserved.
The Python suite passed 333 tests with three optional skips.
The three-GPU coding trial passed in 56.29 seconds with three distinct threads, requests overlapping across all three workers, and all three independent tests passing.
GPU utilization samples from the initial attempt also confirmed simultaneous activity on GPUs 1, 2, and 3.
The first three-GPU attempt exposed helpers confusing inherited lead context with their own identity; explicit per-thread helper identity fixed this on the rerun.
The 56.29-second result is about twice as fast as the earlier baseline, but the identity fix and single-run variability mean this is not a controlled GPU-only speedup measurement.

## Implementation and cleanup

The loopback gateway uses a fresh bearer token and forwards requests with each worker's own router credentials.
It translates namespaced tools and local V2 agent messages into the formats llama.cpp accepts, then restores namespaces in streamed tool calls.
The `encrypted_content` field in locally generated agent messages contains Qwen's literal message text; this experiment does not decrypt cloud conversations.
Codex uses HTTP streaming to the gateway, which forwards inference through each Marathon router's WebSocket conversation engine.
This retains Marathon's existing patch conversion, tool execution, and prompt-cache handling.
Normal Marathon compaction remains on the agent's assigned worker.
Each agent gets its own stable prompt-cache key so helpers cannot supersede each other's requests when sharing a worker in the baseline.

Exit closes the gateway and releases its runtime processes and pool leases, including partially started teams.
The broker may keep model weights warm under its existing idle TTL.
Other Marathon instances and the broker configuration remain untouched.
Session history and routing evidence are retained under `.marathon/swarms/<run-id>` in the working directory, or a new directory supplied with `--output-dir`.
`events.jsonl` records thread assignments and request timings without recording router tokens.
