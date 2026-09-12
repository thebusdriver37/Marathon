#!/usr/bin/env python3
"""Measure the installed Marathon terminal from launch through its first reply.

The first run uses the worker's existing state; subsequent runs can reuse it.
This evaluator never unloads workers or changes their idle policy.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time

from e2e_smoke import LAUNCHER, Terminal, read_events


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-gpu", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--launcher", type=Path, default=LAUNCHER)
    parser.add_argument("--typing-delay", type=float, default=0.0)
    args = parser.parse_args()
    if not args.run_gpu:
        parser.error("live inference is opt-in; pass --run-gpu")
    if min(args.repeats, args.timeout) < 1:
        parser.error("repeats and timeout must be positive")
    if args.typing_delay < 0:
        parser.error("typing delay must be non-negative")
    launcher = args.launcher.resolve()
    output = args.output_dir.resolve() if args.output_dir else Path(
        tempfile.mkdtemp(prefix="marathon-startup-")
    )
    if args.output_dir:
        output.mkdir(parents=True, exist_ok=False)
    print(f"Evidence: {output}", flush=True)
    results = []
    for repeat in range(args.repeats):
        trial = output / str(repeat)
        trial.mkdir()
        environment = dict(
            os.environ,
            TERM="xterm-256color",
            MARATHON_RUNS_DIR=str(trial / "runs"),
            MARATHON_CODEX_HOME=str(trial / "codex-home"),
            MARATHON_STOCK_CODEX_HOME=str(trial / "stock-codex-home"),
            MARATHON_AUTO_INSTALL_CLI="0",
        )
        environment.pop("PYTHONPATH", None)
        # Use normal worker selection and session preparation, including an
        # automatic instance relaunch when other Marathon sessions are open.
        started = time.monotonic_ns()
        terminal = Terminal([str(launcher), "codex", "--no-alt-screen"],
                            launcher.parent.parent, environment, trial / "terminal.txt")

        def events():
            return [event for path in (trial / "runs").rglob("*.jsonl")
                    for event in read_events(path)]

        try:
            # The first title can say Ready while the model catalog is loading.
            # Also require the frontend's successful Responses connection probe.
            terminal.wait(lambda: "| Ready |" in terminal.text and any(
                event.get("event") == "router.response.completed" for event in events()
            ), args.timeout)
            ready = time.monotonic_ns()
            milestones = {}
            for event in events():
                if event.get("event") in {"run.started", "backend.model.ready", "runtime.ready", "frontend.started"}:
                    milestones[event["event"]] = round((event["mono_ns"] - started) / 1e9, 3)
            frontend = next(event for event in events() if event.get("event") == "frontend.started")
            sessions = Path(frontend["data"]["codex_home"]) / "sessions"
            before = {path: path.stat().st_size for path in sessions.rglob("*.jsonl")}
            expected = f"STARTUP_OK_{os.getpid()}_{repeat}"
            time.sleep(args.typing_delay)
            prompted = time.monotonic_ns()
            terminal.prompt(f"Reply only {expected}. Do not use tools.")

            def reply():
                for path in sessions.rglob("*.jsonl"):
                    with path.open("rb") as handle:
                        handle.seek(before.get(path, 0))
                        for line in handle:
                            try:
                                event = json.loads(line)
                            except (ValueError, UnicodeDecodeError):
                                continue
                            payload = event.get("payload", {})
                            if event.get("type") == "event_msg" and payload.get("type") == "task_complete":
                                # Other sessions may share this home. Only our
                                # deliberately unique reply completes this check.
                                if payload.get("last_agent_message", "").strip() == expected:
                                    return True
                return False

            terminal.wait(reply, args.timeout)
            completed = time.monotonic_ns()
            terminal.wait(
                lambda: expected in terminal.clean()
                and terminal.clean().rfind("Ask Marathon to do anything")
                > terminal.clean().rfind(expected),
                args.timeout,
            )
            row = dict(repeat=repeat, passed=True, ready_seconds=round((ready - started) / 1e9, 3),
                       first_reply_seconds=round((completed - started) / 1e9, 3),
                       prompt_reply_seconds=round((completed - prompted) / 1e9, 3),
                       typing_delay_seconds=args.typing_delay,
                       milestones=milestones)
        except Exception as error:
            row = dict(repeat=repeat, passed=False, error=str(error))
        finally:
            terminal.close()
        results.append(row)
        (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    return 0 if all(row["passed"] for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
