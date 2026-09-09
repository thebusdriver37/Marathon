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
Marathon defaults to Codex's workspace-write sandbox with network access enabled.
On Linux, each shell command runs in a separate PID namespace, so broad process-name kills cannot reach the frontend, supervisor, or another command's processes.
Run from your project directory and use Codex's additional writable directories when needed.
Explicitly disabling the sandbox also disables this process protection.

To compare the same three-agent workflow on one GPU, or try it while only one pool lease is free:

```bash
marathon swarm --workers 1 exec "Your task"
```

`--agents 2` selects a lead and one helper instead.
Worker count defaults to agent count.
A new swarm starts with an isolated Codex home; resume retains that home and its native Codex agent graph.
Reuse the helpers through messages and follow-up tasks; the gateway does not recycle thread assignments during a session.
Resume the saved team using its home and the complete lead session ID:

```bash
MARATHON_CODEX_HOME=/absolute/path/to/run/codex-home marathon resume SESSION_UUID
```

Ordinary `marathon resume` and `marathon exec resume` recognize a saved swarm and restore its gateway, worker count, and collaboration tools automatically.
To override the number of GPU workers for this launch, use `marathon swarm --workers 1 resume SESSION_UUID` with the same `MARATHON_CODEX_HOME`.
Resume requires an explicit lead UUID; the swarm picker and `--last` are not supported.
The lead can reuse its saved helpers with `followup_task`; `list_agents` may show only currently running agents until an idle helper is resumed.
The lead is offered `followup_task` for helper communication, which starts an idle helper or delivers a message to a running one.
The redundant `send_message` option is hidden from the lead because it does not start idle helpers; helpers retain it for progress messages to the lead.
New runs persist their agent and worker counts, while older runs recover worker count from their event log and use the original three-agent default.
The normal router translates saved local collaboration messages during both inference and compaction.
The saved rollout is not rewritten by that translation.

Swarm requests omit long runs of literal echo-only commands and their associated reasoning from the model's view of old history, while preserving the saved rollout and useful commands.
For a recovered lead, the first inference request exposes only the read-only helper-list tool; subsequent requests restore the normal tools.
This breaks the observed pattern of announcing helper calls while merely printing shell markers.
If an agent produces eight consecutive echo-only commands in a turn, Marathon stops that turn with an explanatory error instead of allowing the loop to continue indefinitely.
Send a new message to retry after that error.

## Validation

```bash
.marathon/venv/bin/python -m unittest discover -s tests -p test_swarm.py -v
MARATHON_TEST_CODEX_BIN=~/.local/share/marathon/bin/codex .marathon/venv/bin/python -m unittest discover -s tests -p test_process_isolation.py -v
.marathon/venv/bin/python scripts/evals/swarm.py --run-gpu --workers 1 --check-resume
.marathon/venv/bin/python scripts/evals/swarm.py --run-gpu --workers 3
.marathon/venv/bin/python scripts/evals/swarm_recovery.py --run-gpu --source-home /path/to/swarm/codex-home --session LEAD_UUID
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

Recovery validation reproduced the llama.cpp `Cannot determine type of 'item'` error by resuming a copy of an interrupted real swarm rollout.
With shared history normalization, that same copy compacted successfully and answered the diagnostic prompt, while the original rollout remained byte-for-byte unchanged.
The real Codex recovery test checks rejection of a second live writer, forcibly kills its disposable launcher, then successfully resumes the conversation without removing lock files.
The optional resume evaluation shuts down a fresh team, resumes through the ordinary headless command, and verifies that the same three thread IDs run again without changing the completed files.
The recovery evaluation copies a real three-agent history and its SQLite state into a private temporary directory, asks both saved helpers for a diagnostic reply, and checks their identities, the absence of new shell calls, and that the original history is unchanged.
Its transcript and results remain in the printed evidence directory for inspection.
The echo recovery change passed a real-history replay with both original helpers replying and zero new shell commands, plus a fresh three-GPU coding trial in 38.69 seconds with all three independent tests passing.
A separate replay still showed a helper continuing its old task despite receiving the new brief correctly; the echo guard does not guarantee that the model follows every instruction.
The process-isolation test verifies a separate PID namespace before running the original broad `pkill` pattern and checks that a matching host sentinel survives.
The post-fix three-agent coding evaluation passed on one worker in 98.15 seconds, including independent verification of all three coding tests.

## Implementation and cleanup

The loopback gateway uses a fresh bearer token and forwards requests with each worker's own router credentials.
It translates namespaced tools and local V2 agent messages into the formats llama.cpp accepts, then restores namespaces in streamed tool calls.
The `encrypted_content` field in locally generated agent messages contains Qwen's literal message text; this experiment does not decrypt cloud conversations.
Codex uses HTTP streaming to the gateway, which forwards inference through each Marathon router's WebSocket conversation engine.
This retains Marathon's existing patch conversion, tool execution, and prompt-cache handling.
Normal Marathon compaction remains on the agent's assigned worker.
Each agent gets its own stable prompt-cache key so helpers cannot supersede each other's requests when sharing a worker in the baseline.

Exit closes the gateway and releases its runtime processes and pool leases, including partially started teams.
On Linux, the frontend receives SIGTERM if its launcher dies, including a launcher SIGKILL, so an orphan cannot retain the session writer lock indefinitely.
The launcher check also covers death before the child installs its parent-death signal.
A second frontend still cannot resume a conversation with a live writer; close that frontend first instead of deleting lock files or automatically killing a process based on a session ID.
The broker may keep model weights warm under its existing idle TTL.
Other Marathon instances and the broker configuration remain untouched.
Session history and routing evidence are retained under `.marathon/swarms/<run-id>` in the working directory, or a new directory supplied with `--output-dir`.
`events.jsonl` records thread assignments and request timings without recording router tokens.
