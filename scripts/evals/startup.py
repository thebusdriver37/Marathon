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
import statistics
import sys
import tempfile
import time
import urllib.request

from e2e_smoke import LAUNCHER, Terminal, read_events


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-gpu", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--launcher", type=Path, default=LAUNCHER)
    parser.add_argument("--model", help="model ID from Marathon's frontend catalog")
    parser.add_argument("--typing-delay", type=float, default=0.0)
    parser.add_argument("--catalog-requests", type=int, default=0,
                        help="measure live model-picker HTTP refreshes after readiness")
    parser.add_argument("--no-live-handoff", action="store_true",
                        help="exercise fresh conversations using disk starter caches only")
    parser.add_argument("--router-source", type=Path,
                        help="alternate router source for an isolated baseline comparison")
    args = parser.parse_args()
    if not args.run_gpu:
        parser.error("live inference is opt-in; pass --run-gpu")
    if min(args.repeats, args.timeout) < 1:
        parser.error("repeats and timeout must be positive")
    if args.typing_delay < 0:
        parser.error("typing delay must be non-negative")
    if args.catalog_requests < 0:
        parser.error("catalog requests must be non-negative")
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
        if args.no_live_handoff or args.router_source:
            router_source = (args.router_source.resolve() if args.router_source else
                             launcher.parent.parent / "scripts/routers/codex_local_router.py")
            wrapper = trial / "router-python"
            wrapper.write_text(f"#!{sys.executable}\n" + f'''
import sys
from pathlib import Path
sys.argv = sys.argv[1:]
__file__ = sys.argv[0]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[2]))
__name__ = "evaluation_router"
sys.modules[__name__] = sys.modules["__main__"]
exec(compile(Path({str(router_source)!r}).read_text(), __file__, "exec"), globals())
if {args.no_live_handoff!r}:
    async def no_handoff(self, *args):
        self.retained_slot_hint = None
        return False
    RouterState._adopt_retained_slot = no_handoff
main()
''')
            wrapper.chmod(0o700)
            environment["MARATHON_ROUTER_PYTHON"] = str(wrapper)
        # Use normal worker selection and session preparation, including an
        # automatic instance relaunch when other Marathon sessions are open.
        started = time.monotonic_ns()
        command = [str(launcher), "codex", "--no-alt-screen"]
        if args.model:
            command.extend(["--model", args.model])
        terminal = Terminal(command,
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
            catalog_ms = []
            if args.catalog_requests:
                sys.path.insert(0, str(launcher.parent.parent))
                from marathon_app.instance import resolve_instance
                from marathon_app.runtime import runtime_paths
                from marathon_app.router_security import open_api_request

                run = next(event for event in events() if event.get("event") == "run.started")
                instance = run["data"].get("instance")
                config = resolve_instance(instance).settings
                host = config.router_host
                if ":" in host:
                    host = f"[{host}]"
                token = runtime_paths(instance).session_file.with_name("router.token").read_text()
                request = urllib.request.Request(
                    f"http://{host}:{config.router_port}/v1/models",
                    headers={"Authorization": f"Bearer {token}"},
                )
                previous = None
                for _ in range(args.catalog_requests):
                    tick = time.perf_counter()
                    with open_api_request(request, timeout=5) as response:
                        payload = json.load(response)
                    catalog_ms.append((time.perf_counter() - tick) * 1000)
                    if not payload.get("models") or (previous is not None and payload != previous):
                        raise RuntimeError("model catalog empty or changed during benchmark")
                    previous = payload
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
                                    return payload
                return False

            completed_turn = terminal.wait(reply, args.timeout)
            completed = time.monotonic_ns()
            terminal.wait(
                # Incremental terminal rendering can reuse the existing input
                # prompt. Wait for Ready after this unique reply before sending
                # Ctrl-C, so cleanup cannot race the frontend's turn completion.
                lambda: expected in terminal.clean()
                and terminal.text.rfind("| Ready |") > terminal.text.rfind("| Working |"),
                args.timeout,
            )
            row = dict(repeat=repeat, passed=True, ready_seconds=round((ready - started) / 1e9, 3),
                       first_reply_seconds=round((completed - started) / 1e9, 3),
                       prompt_reply_seconds=round((completed - prompted) / 1e9, 3),
                       typing_delay_seconds=args.typing_delay,
                       milestones=milestones)
            first_token_ms = completed_turn.get("time_to_first_token_ms")
            if isinstance(first_token_ms, (int, float)):
                row["first_activity_seconds"] = first_token_ms / 1000
            trace = next((event["data"] for event in reversed(events())
                          if event.get("event") == "router.response.completed"
                          and event.get("data", {}).get("backend_timings")), None)
            if trace:
                row["backend_timings"] = trace["backend_timings"]
                row["slot_prepare_mode"] = trace.get("slot", {}).get("prepare_mode")
            row["live_handoff_disabled"] = args.no_live_handoff
            if catalog_ms:
                row["catalog_ms"] = catalog_ms
                row["catalog_median_ms"] = statistics.median(catalog_ms)
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
