#!/usr/bin/env python3
"""Run one observable, sandboxed Codex scenario against an active Marathon runtime."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from marathon_app import runtime as runtime_module
from marathon_app.catalog import discover_models, find_model, find_profile
from marathon_app.codex_telemetry import snapshot_sessions, summarize_session_changes
from marathon_app.codex_home import codex_environment
from marathon_app.frontends import codex_command
from marathon_app.runtime import Runtime
from marathon_app.telemetry import EventWriter


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--workspace", type=Path, required=True)
    result.add_argument("--prompt-file", type=Path, required=True)
    result.add_argument("--name", required=True)
    result.add_argument("--timeout", type=int, default=900)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--use-user-config", action="store_true")
    result.add_argument("--model", default="qwen3.6-27b")
    return result


def active_runtime(model_query: str) -> tuple[Runtime, EventWriter]:
    session = json.loads(runtime_module.SESSION_FILE.read_text(encoding="utf-8"))
    model = find_model(model_query, discover_models())
    profile = find_profile(model, str(session["profile"]), "codex")
    runtime = Runtime(model, profile)
    runtime._context_window = int(session["context"])
    writer = EventWriter(Path(session["run_log"]), str(session["run_id"]), "eval")
    return runtime, writer


def terminate_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def main() -> int:
    args = parser().parse_args()
    workspace = args.workspace.resolve()
    prompt_file = args.prompt_file.resolve()
    if not (workspace / ".git").exists():
        raise SystemExit(f"workspace is not a Git repository: {workspace}")
    if workspace not in prompt_file.parents:
        raise SystemExit("prompt file must be contained in the sandbox workspace")

    runtime, writer = active_runtime(args.model)
    prompt = prompt_file.read_text(encoding="utf-8")
    output_dir = workspace / ".obstacle"
    output_dir.mkdir(exist_ok=True)
    event_log = output_dir / f"{args.name}.codex.jsonl"
    last_message = output_dir / f"{args.name}.last.md"
    environment, codex_home, shared_profile = codex_environment()
    before = snapshot_sessions(codex_home)

    base_command = codex_command(runtime, shared_profile=shared_profile)
    config_args = [] if args.use_user_config else ["--ignore-user-config"]
    if args.resume:
        command = [
            base_command[0], "exec", "resume", "--last", *config_args,
            *base_command[1:],
            "-c", 'approval_policy="never"',
            "-c", 'sandbox_mode="workspace-write"',
            "-c", "sandbox_workspace_write.network_access=false",
            "--json", "-o", str(last_message), prompt,
        ]
    else:
        command = [
            base_command[0], "exec", *config_args, *base_command[1:],
            "--sandbox", "workspace-write",
            "-C", str(workspace),
            "-c", 'approval_policy="never"',
            "-c", "sandbox_workspace_write.network_access=false",
            "--json",
            "-o", str(last_message),
            prompt,
        ]
    writer.emit(
        "eval.scenario.started",
        {
            "name": args.name,
            "workspace": str(workspace),
            "timeout_s": args.timeout,
            "resume": args.resume,
            "sandbox": "workspace-write",
            "shell_network_access": False,
        },
    )
    started = time.monotonic()
    returncode = 1
    timed_out = False
    with event_log.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            command,
            cwd=workspace,
            env=environment,
            text=True,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_group(process)
            returncode = 124

    summaries = summarize_session_changes(
        before,
        cwd=workspace,
        provider="marathon-local",
        codex_home=codex_home,
    )
    for summary in summaries:
        writer.emit("codex.session.completed", {"scenario": args.name, **summary})
    writer.emit(
        "eval.scenario.completed",
        {
            "name": args.name,
            "duration_ms": (time.monotonic() - started) * 1000.0,
            "returncode": returncode,
            "timed_out": timed_out,
            "codex_event_log": str(event_log),
            "last_message": str(last_message),
            "sessions_changed": len(summaries),
        },
        level="info" if returncode == 0 else "error",
    )
    print(json.dumps({
        "name": args.name,
        "returncode": returncode,
        "timed_out": timed_out,
        "duration_s": round(time.monotonic() - started, 3),
        "event_log": str(event_log),
        "last_message": str(last_message),
    }, indent=2))
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
