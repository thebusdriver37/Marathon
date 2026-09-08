#!/usr/bin/env python3
"""Opt-in compaction benchmark through the installed Marathon terminal and GPU.

Retains transcripts, summaries, recall answers, and router traces as evidence.
Facts originate in tool output so retained user messages cannot satisfy recall.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
import subprocess
import tempfile
import time

from e2e_smoke import LAUNCHER, Terminal, read_events

FACTS = {
    "release": "harbor-731",
    "port": "18473",
    "retry_ms": "375",
    "max_attempts": "7",
    "checkpoint": "delta-59",
    "checksum": "6ac91f02",
    "migration": "042_reconcile.sql",
    "rollback": "restore-cobalt",
    "owner": "Mira Solano",
    "pending": "validate replica lag",
    "failure": "test_replay_order",
    "constraint": "never rewrite archived events",
}


def completed(session):
    return [e["payload"] for e in read_events(session)
            if e.get("type") == "event_msg" and e.get("payload", {}).get("type") == "task_complete"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-gpu", action="store_true")
    parser.add_argument("--efforts", nargs="+", default=["medium", "low", "none"])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--coding-effort", default="medium",
                        help="coding effort held fixed while varying compaction (default: medium)")
    parser.add_argument("--filler-lines", type=int, default=240)
    parser.add_argument("--tool-output-max-chars", type=int,
                        help="test-only router output limit for long-context measurements")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--auto-compact-token-limit", type=int,
                        help="also require automatic compaction during fixture setup")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not args.run_gpu:
        parser.error("live inference is opt-in; pass --run-gpu")
    if min(args.repeats, args.cycles, args.timeout) < 1 or args.filler_lines < 0:
        parser.error("repeats, cycles, and timeout must be positive; filler lines must be nonnegative")
    # Import the router's exact truncation policy only for the opt-in live run.
    import sys
    sys.path.insert(0, str(LAUNCHER.parent.parent))
    sys.path.insert(0, str(LAUNCHER.parent.parent / "scripts/routers"))
    from codex_local_router import _bound_tool_output_item, DEFAULT_TOOL_OUTPUT_MAX_CHARS
    output = args.output_dir.resolve() if args.output_dir else Path(tempfile.mkdtemp(prefix="marathon-compaction-"))
    if args.output_dir:
        output.mkdir(parents=True, exist_ok=False)
    print(f"Evidence: {output}", flush=True)
    results = []
    for repeat in range(args.repeats):
        for effort in args.efforts:
            trial = output / f"{effort}-{repeat}"
            project = trial / "project"
            project.mkdir(parents=True)
            subprocess.run(["git", "init", "-q", str(project)], check=True)
            lines = [f"Historical observation {i:04d}: completed routine reconciliation; no outstanding action."
                     for i in range(args.filler_lines)]
            for index, (key, value) in enumerate(FACTS.items()):
                lines.insert(index * max(1, len(lines) // len(FACTS)), f"ACTIVE HANDOFF FACT {key}: {value}")
            (project / "handoff.txt").write_text("\n".join(lines) + "\n")
            instance = f"compact-{os.getpid()}"
            environment = dict(os.environ, TERM="xterm-256color", CODEX_CLI_NAME="codex",
                               MARATHON_CLI_NAME=str(LAUNCHER),
                               MARATHON_CODEX_HOME=str(trial / "codex-home"),
                               MARATHON_RUNS_DIR=str(trial / "runs"))
            environment.pop("MARATHON_COMPACTION_REASONING_EFFORT", None)
            if args.tool_output_max_chars:
                environment["MARATHON_TOOL_OUTPUT_MAX_CHARS"] = str(args.tool_output_max_chars)
            if args.coding_effort:
                environment["MARATHON_COMPACTION_REASONING_EFFORT"] = effort
            sessions = trial / "codex-home/instances" / instance / "sessions"
            config_args = []
            if args.auto_compact_token_limit:
                config_args = ["-c", f"model_auto_compact_token_limit={args.auto_compact_token_limit}"]
            terminal = Terminal([
                str(LAUNCHER), "--instance", instance, "codex", "--no-alt-screen",
                "-s", "workspace-write", "-c", f'model_reasoning_effort="{args.coding_effort or effort}"',
                *config_args,
                "Run exactly this shell command: cat handoff.txt; rg 'ACTIVE HANDOFF FACT' handoff.txt. "
                f"Use max_output_tokens {max(20000, args.tool_output_max_chars or 0)}. "
                "The final rg output ensures the facts survive output truncation. "
                "Remember every ACTIVE HANDOFF FACT for our pending release task. "
                "Do not echo the facts, edit files, or delegate. Reply only READY after reading."],
                project, environment, trial / "terminal.txt")
            try:
                session = terminal.wait(lambda: next(iter(sessions.rglob("*.jsonl")), None), args.timeout)
                terminal.wait(lambda: completed(session), args.timeout)
                observed = "\n".join(str(_bound_tool_output_item(e["payload"], int(environment.get(
                                         "MARATHON_TOOL_OUTPUT_MAX_CHARS", DEFAULT_TOOL_OUTPUT_MAX_CHARS)))[0].get("output", ""))
                                     for e in read_events(session) if e.get("type") == "response_item"
                                     and e.get("payload", {}).get("type") == "function_call_output")
                missing = [key for key, value in FACTS.items() if value not in observed]
                if missing:
                    raise AssertionError(f"Fixture facts were not observed before compaction: {missing}")
                automatic = [e for e in read_events(session) if e.get("type") == "compacted"]
                if args.auto_compact_token_limit and not automatic:
                    raise AssertionError("Automatic compaction was requested but did not occur")
                if automatic:
                    for auto_index, event in enumerate(automatic):
                        auto_summary = event["payload"].get("message", "")
                        (trial / f"automatic-summary-{auto_index}.json").write_text(
                            json.dumps(event["payload"], indent=2))
                        missing = [key for key, value in FACTS.items() if value.lower() not in auto_summary.lower()]
                        if missing:
                            raise AssertionError(f"Automatic summary {auto_index} omitted fixture facts: {missing}")
                # Keep the original evidence but make accidental file-based recall impossible.
                (project / "handoff.txt").rename(trial / "handoff.source.txt")
                for cycle in range(args.cycles):
                    before = read_events(session)
                    compact_count = sum(e.get("type") == "compacted" for e in before)
                    task_count = len(completed(session))
                    started = time.monotonic()
                    terminal.prompt("/compact")
                    terminal.wait(lambda: len(completed(session)) > task_count, args.timeout)
                    elapsed = time.monotonic() - started
                    compactions = [e for e in read_events(session) if e.get("type") == "compacted"]
                    if len(compactions) != compact_count + 1:
                        raise AssertionError("Compaction finished without installing exactly one summary")
                    payload = compactions[-1]["payload"]
                    traces = [e for p in (trial / "runs").rglob("*.jsonl") for e in read_events(p)]
                    response = next((e["data"] for e in reversed(traces)
                                     if e.get("event") == "router.response.completed"
                                     and e["data"].get("response_id") == payload.get("compaction_response_id")), {})
                    summary = payload.get("message", "")
                    (trial / f"summary-{cycle}.json").write_text(json.dumps(payload, indent=2))
                    summary_hits = {key: bool(re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", summary, re.I))
                                    for key, value in FACTS.items()}
                    task_count = len(completed(session))
                    event_count = len(read_events(session))
                    terminal.prompt("Without tools or reading files, recall the ACTIVE HANDOFF FACT values. "
                                    "Reply with only a JSON object using these keys: " + ", ".join(FACTS) +
                                    ". If a value is unknown, use null. Do not guess.")
                    answer = terminal.wait(lambda: completed(session)[task_count:] or None, args.timeout)[0]
                    reply = answer.get("last_agent_message", "")
                    new_events = read_events(session)[event_count:]
                    used_tools = any(e.get("type") == "response_item" and e.get("payload", {}).get("type")
                                     in {"function_call", "custom_tool_call", "local_shell_call"} for e in new_events)
                    try:
                        recalled = json.loads(reply.strip().removeprefix("```json").removesuffix("```").strip())
                    except json.JSONDecodeError:
                        recalled = {}
                    if not isinstance(recalled, dict):
                        recalled = {}
                    hits = {key: str(recalled.get(key)).lower() == value.lower() for key, value in FACTS.items()}
                    (trial / f"recall-{cycle}.txt").write_text(reply)
                    row = dict(effort=effort, repeat=repeat, cycle=cycle, compaction_seconds=round(elapsed, 3),
                               prefix_cache=environment.get("MARATHON_COMPACTION_PREFIX_CACHE", "default"),
                               slot_prepare_mode=response.get("slot", {}).get("prepare_mode"),
                               automatic_compactions=len(automatic), coding_effort=args.coding_effort,
                               backend_timings=response.get("backend_timings"),
                               usage=response.get("backend", {}).get("usage"),
                               summary_chars=len(summary), summary_hits=summary_hits, recall_hits=hits,
                               recall_used_tools=used_tools, passed=all(hits.values()) and all(summary_hits.values()) and not used_tools)
                    results.append(row)
                    (output / "results.json").write_text(json.dumps(results, indent=2))
                    print(json.dumps(row), flush=True)
            except Exception as error:
                row = dict(effort=effort, repeat=repeat, passed=False, error=str(error))
                results.append(row)
                (output / "results.json").write_text(json.dumps(results, indent=2))
                print(json.dumps(row), flush=True)
            finally:
                terminal.close()
    return 0 if all(row["passed"] for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
