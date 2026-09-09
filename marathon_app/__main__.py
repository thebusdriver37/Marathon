"""Command entry point for the unified Marathon runtime."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import NoReturn

from rich.console import Console
from rich.table import Table

from . import __version__
from .catalog import default_profile_id, discover_models, format_size
from .codex_home import marathon_codex_home, session_home_for_id
from .instance import normalize_instance_name, resolve_instance
from .model_library import register_model_root
from .runtime import RuntimeBusyError, automatic_launch_instance, request_stop, runtime_paths
from .router_security import open_api_request
from .telemetry import resolve_run, summarize_run
from .remote import run_remote_host_command
from .ui import (
    run_codex_default,
    run_dashboard,
    run_dyno_dashboard,
    run_remote_dashboard,
    run_setup_dashboard,
)


def _models(targets: list[str]) -> int:
    console = Console()
    if targets:
        if len(targets) != 2 or targets[0] != "add":
            console.print("[bold red]Usage:[/bold red] marathon models [add <folder>]")
            return 2
        try:
            root = register_model_root(Path(targets[1]))
        except ValueError as error:
            console.print(f"[bold red]{error}[/bold red]")
            return 2
        console.print(f"[green]Added model folder:[/green] {root}")
    models = discover_models()
    table = Table(title="Installed Marathon models")
    table.add_column("Model")
    table.add_column("Size", justify="right")
    table.add_column("Default profile")
    table.add_column("Path", style="dim")
    for model in models:
        table.add_row(model.display_name, format_size(model.size_bytes), default_profile_id(model), str(model.path))
    console.print(table)
    return 0 if models else 1


def _status(instance: str | None = None) -> int:
    console = Console()
    resolved = resolve_instance(instance)
    config = resolved.settings
    session_file = runtime_paths(resolved.name).session_file
    try:
        session = json.loads(session_file.read_text(encoding="utf-8"))
        token = session_file.with_name("router.token").read_text(encoding="utf-8")
        host = config.router_host
        if ":" in host:
            host = f"[{host}]"
        request = urllib.request.Request(
            f"http://{host}:{config.router_port}/health",
            headers={"Authorization": f"Bearer {token}"},
        )
        with open_api_request(request, timeout=2) as response:
            health = json.loads(response.read().decode("utf-8"))
    except (OSError, json.JSONDecodeError, urllib.error.URLError):
        if resolved.name:
            console.print(
                f"[dim]Marathon instance '{resolved.name}' is stopped.[/dim]"
            )
        else:
            console.print("[dim]Marathon is stopped.[/dim]")
        return 1
    label = f" instance '{resolved.name}'" if resolved.name else ""
    console.print(
        f"[green]● Marathon{label} running[/green] · "
        f"{session.get('model')} / {session.get('profile')}"
    )
    console.print(f"Supervisor PID {session.get('supervisor_pid')} · backend {health.get('backend_health') or 'ready'}")
    if session.get("run_log"):
        console.print(f"Trace: {session['run_log']}", style="dim")
    return 0


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{int(seconds // 60)}m {seconds % 60:.0f}s"


def _run_is_active(path: Path, instance: str | None = None) -> bool:
    session_file = runtime_paths(instance).session_file
    try:
        session = json.loads(session_file.read_text(encoding="utf-8"))
        pid = int(session["supervisor_pid"])
        run_log = Path(session["run_log"]).expanduser().resolve()
        os.kill(pid, 0)
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False
    return run_log == path.resolve()


def _report(target: str | None, instance: str | None = None) -> int:
    console = Console()
    try:
        path = resolve_run(target, instance)
        summary = summarize_run(path, live=_run_is_active(path, instance))
    except (OSError, ValueError) as error:
        console.print(f"[bold red]Cannot read run:[/bold red] {error}")
        return 2
    console.print(
        f"[bold magenta]Marathon run {summary['run_id']}[/bold magenta] · "
        f"{summary['model']} / {summary['profile']}"
    )
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="cyan")
    table.add_column()
    table.add_row("Trace", str(summary["path"]))
    if summary["complete"]:
        state = "complete"
    elif summary["active"]:
        state = "active"
    else:
        state = "incomplete / interrupted"
    table.add_row("State", state)
    table.add_row("Duration", _format_duration(float(summary["duration_s"])))
    table.add_row(
        "Events",
        f"{summary['event_count']:,} ({summary['errors']} runtime errors)",
    )
    active_label = (
        f" · {summary['active_codex_sessions']} active"
        if summary["active_codex_sessions"]
        else ""
    )
    table.add_row(
        "Activity",
        f"{summary['router_turns']} Codex responses · "
        f"{summary['chat_completion_requests']} chat API calls · "
        f"{summary['direct_turns']} direct turns · "
        f"{summary['codex_sessions']} Codex launches{active_label} · "
        f"{summary['hermes_sessions']} Hermes launches",
    )
    usage = summary["usage"]
    if usage:
        table.add_row(
            "Backend tokens",
            f"{usage.get('input_tokens', 0):,} input · {usage.get('output_tokens', 0):,} output · "
            f"{usage.get('total_tokens', 0):,} total",
        )
    codex_usage = summary["codex_usage"]
    if codex_usage:
        table.add_row(
            "Codex tokens",
            f"{codex_usage.get('input_tokens', 0):,} input · "
            f"{codex_usage.get('cached_input_tokens', 0):,} cached · "
            f"{codex_usage.get('output_tokens', 0):,} output · "
            f"{codex_usage.get('reasoning_output_tokens', 0):,} reasoning",
        )
    if summary["avg_backend_latency_ms"] is not None:
        table.add_row("Average backend latency", f"{summary['avg_backend_latency_ms'] / 1000:.2f}s")
    if summary["prompt_tps"] is not None or summary["decode_tps"] is not None:
        table.add_row(
            "Model throughput",
            f"{summary['prompt_tps'] or 0:.1f} prompt tok/s · {summary['decode_tps'] or 0:.1f} decode tok/s",
        )
    if summary["avg_direct_ttft_ms"] is not None:
        table.add_row("Direct Chat TTFT", f"{summary['avg_direct_ttft_ms'] / 1000:.2f}s average")
    if summary["gpu_samples"]:
        table.add_row(
            "GPU telemetry",
            f"{summary['gpu_samples']} samples · {summary['avg_gpu_power_w'] or 0:.1f}W average/card · "
            f"{summary['avg_gpu_utilization_pct'] or 0:.0f}% utilization · "
            f"{summary['peak_gpu_memory_mib'] or 0:.0f} MiB peak/card · "
            f"{summary['peak_gpu_temperature_c'] or 0:.0f}°C peak",
        )
        table.add_row("Estimated GPU energy", f"{summary['energy_wh']:.2f} Wh")
        if summary["estimated_gpu_energy_cost_usd"] is not None:
            table.add_row(
                "Estimated GPU energy cost",
                f"${summary['estimated_gpu_energy_cost_usd']:.4f}",
            )
    if summary["tool_calls"]:
        tools = ", ".join(f"{name} ×{count}" for name, count in sorted(summary["tool_calls"].items()))
        table.add_row("Codex tools", tools)
        if summary["avg_tool_duration_ms"] is not None:
            table.add_row("Average tool duration", f"{summary['avg_tool_duration_ms'] / 1000:.2f}s")
    if summary["tool_failures"]:
        failures = ", ".join(
            f"{name} ×{count}"
            for name, count in sorted(summary["failed_tools"].items())
        )
        table.add_row(
            "Tool failures",
            f"[bold red]{summary['tool_failures']}[/bold red] ({failures})",
        )
    if summary["reasoning_efforts"]:
        efforts = ", ".join(
            f"{name} ×{count}" for name, count in sorted(summary["reasoning_efforts"].items())
        )
        table.add_row("Reasoning effort", efforts)
    if summary["router_tool_calls"]:
        tools = ", ".join(
            f"{name} ×{count}" for name, count in sorted(summary["router_tool_calls"].items())
        )
        table.add_row("Model tool calls", tools)
    table.add_row("Dropped telemetry", str(summary["dropped_events"]))
    console.print(table)
    if summary["error_events"]:
        errors = Table(title="Recent errors", show_lines=False)
        errors.add_column("Time", style="dim")
        errors.add_column("Event", style="red")
        errors.add_column("Detail")
        for item in summary["error_events"]:
            errors.add_row(
                str(item.get("ts") or ""),
                str(item.get("event") or ""),
                str(item.get("message") or "")[:240],
            )
        console.print(errors)
    return 0


def _compare(targets: list[str], instance: str | None = None) -> int:
    console = Console()
    if len(targets) != 2:
        console.print("[bold red]Usage:[/bold red] marathon compare <run-a> <run-b>")
        return 2
    try:
        paths = [resolve_run(target, instance) for target in targets]
        left, right = (
            summarize_run(path, live=_run_is_active(path, instance))
            for path in paths
        )
    except (OSError, ValueError) as error:
        console.print(f"[bold red]Cannot compare runs:[/bold red] {error}")
        return 2
    table = Table(title="Marathon run comparison")
    table.add_column("Metric", style="cyan")
    table.add_column(str(left["run_id"]), justify="right")
    table.add_column(str(right["run_id"]), justify="right")
    rows = (
        ("Model", left["model"], right["model"]),
        ("Profile", left["profile"], right["profile"]),
        ("Duration", _format_duration(left["duration_s"]), _format_duration(right["duration_s"])),
        ("Backend turns", left["router_turns"], right["router_turns"]),
        ("Chat API calls", left["chat_completion_requests"], right["chat_completion_requests"]),
        ("Hermes launches", left["hermes_sessions"], right["hermes_sessions"]),
        ("Backend output tokens", left["usage"].get("output_tokens", 0), right["usage"].get("output_tokens", 0)),
        ("Average backend latency", f"{(left['avg_backend_latency_ms'] or 0) / 1000:.2f}s", f"{(right['avg_backend_latency_ms'] or 0) / 1000:.2f}s"),
        ("Prompt throughput", f"{left['prompt_tps'] or 0:.1f} tok/s", f"{right['prompt_tps'] or 0:.1f} tok/s"),
        ("Decode throughput", f"{left['decode_tps'] or 0:.1f} tok/s", f"{right['decode_tps'] or 0:.1f} tok/s"),
        ("Average GPU power/card", f"{left['avg_gpu_power_w'] or 0:.1f}W", f"{right['avg_gpu_power_w'] or 0:.1f}W"),
        ("Estimated GPU energy", f"{left['energy_wh']:.2f} Wh", f"{right['energy_wh']:.2f} Wh"),
        ("Peak GPU memory/card", f"{left['peak_gpu_memory_mib'] or 0:.0f} MiB", f"{right['peak_gpu_memory_mib'] or 0:.0f} MiB"),
        ("Runtime errors", left["errors"], right["errors"]),
        ("Tool failures", left["tool_failures"], right["tool_failures"]),
    )
    for metric, a, b in rows:
        table.add_row(str(metric), str(a), str(b))
    console.print(table)
    return 0


def _instance_name(value: str) -> str | None:
    try:
        return normalize_instance_name(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="marathon", description="One-command local AI runtime")
    parser.add_argument("--version", action="version", version=f"Marathon {__version__}")
    parser.add_argument(
        "--instance",
        type=_instance_name,
        metavar="NAME",
        help="use an independent named runtime instance",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=(
            "dashboard",
            "swarm",
            "codex",
            "exec",
            "hermes",
            "direct",
            "remote",
            "remote-host",
            "tune",
            "setup",
            "models",
            "status",
            "stop",
            "report",
            "compare",
            "resume",
            "fork",
        ),
        default="codex",
    )
    parser.add_argument("targets", nargs=argparse.REMAINDER)
    return parser


def _relaunch_as_instance(name: str, argv: list[str] | None, session_home: Path) -> NoReturn:
    """Replace this process with an explicitly identified named instance."""

    original_arguments = list(sys.argv[1:] if argv is None else argv)
    executable = sys.executable
    os.execve(
        executable,
        [
            executable,
            "-m",
            "marathon_app",
            "--instance",
            name,
            *original_arguments,
        ],
        dict(os.environ, _MARATHON_SESSION_HOME=str(session_home)),
    )
    raise RuntimeError("failed to relaunch Marathon")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inherited_home = os.environ.pop("_MARATHON_SESSION_HOME", None)
    session_home = Path(inherited_home) if inherited_home else None
    original_arguments = sys.argv[1:] if argv is None else argv
    explicit_instance = any(
        value == "--instance" or value.startswith("--instance=")
        for value in original_arguments
    )
    try:
        instance = resolve_instance(args.instance).name
    except ValueError as error:
        Console().print(f"[bold red]Invalid instance configuration:[/bold red] {error}")
        return 2
    resume_arguments = None
    if args.command == "resume" and args.targets and not args.targets[0].startswith("-"):
        resume_arguments = ["resume", *args.targets]
    elif (args.command == "exec" and args.targets[:1] == ["resume"]
          and len(args.targets) > 1 and not args.targets[1].startswith("-")):
        resume_arguments = ["exec", *args.targets]
    resuming = args.command == "resume" or (args.command == "exec" and args.targets[:1] == ["resume"])
    if resuming and resume_arguments is None and not explicit_instance:
        from .swarm_session import saved_swarm
        try:
            if saved_swarm(session_home or marathon_codex_home()) is not None:
                raise ValueError("Swarm resume requires the complete lead session UUID; --last and the picker are not supported.")
        except (RuntimeError, ValueError) as error:
            Console().print(f"[bold red]Marathon could not resume:[/bold red] {error}")
            return 2
    if resume_arguments is not None and not explicit_instance:
        try:
            from .swarm_session import resume_id, saved_swarm
            session_home = session_home or session_home_for_id(resume_id(resume_arguments))
            if saved_swarm(session_home) is not None:
                from .swarm import run_swarm
                return run_swarm(resume_arguments, session_home=session_home)
        except (RuntimeError, ValueError) as error:
            Console().print(f"[bold red]Marathon could not resume:[/bold red] {error}")
            return 2
    if not explicit_instance and args.command in {"codex", "exec", "resume", "fork"}:
        try:
            if args.command in {"resume", "fork"} and args.targets and not args.targets[0].startswith("-"):
                session_home = session_home_for_id(args.targets[0])
            automatic = automatic_launch_instance()
        except (RuntimeError, ValueError) as error:
            Console().print(f"[bold red]Marathon could not start:[/bold red] {error}")
            return 2
        if automatic is not None:
            return _relaunch_as_instance(automatic, argv, session_home or marathon_codex_home())

    def launch_codex(arguments: list[str]) -> int:
        try:
            if session_home is not None:
                return run_codex_default(arguments, instance, session_home=session_home)
            return run_codex_default(arguments) if instance is None else run_codex_default(arguments, instance)
        except RuntimeBusyError as error:
            if explicit_instance and inherited_home is None:
                Console().print(f"[bold red]{error}[/bold red]")
                return 2
            # Another simultaneous launch won the lock after our initial probe.
            # Retry selection after cleanup, retaining the conversation's home.
            try:
                automatic = automatic_launch_instance()
            except RuntimeError as exhausted:
                Console().print(f"[bold red]{exhausted}[/bold red]")
                return 2
            original = list(original_arguments[2:] if inherited_home else original_arguments)
            return _relaunch_as_instance(automatic or "default", original, session_home or marathon_codex_home())
    if args.command == "models":
        return _models(args.targets)
    if args.command == "status":
        return _status(instance)
    if args.command == "stop":
        stopped = request_stop(instance)
        if instance:
            Console().print(
                f"[green]Stop requested for Marathon instance '{instance}'.[/green]"
                if stopped
                else f"[dim]Marathon instance '{instance}' is already stopped.[/dim]"
            )
        else:
            Console().print(
                "[green]Stop requested.[/green]"
                if stopped
                else "[dim]Marathon is already stopped.[/dim]"
            )
        return 0
    if args.command == "report":
        return _report(args.targets[0] if args.targets else None, instance)
    if args.command == "compare":
        return _compare(args.targets, instance)
    if args.command == "remote-host":
        return run_remote_host_command(args.targets, instance)
    if args.command == "swarm":
        from .swarm import run_swarm
        try:
            return run_swarm(args.targets)
        except (RuntimeError, ValueError) as error:
            Console().print(f"[bold red]Swarm could not start:[/bold red] {error}")
            return 2
    if args.command == "remote":
        if len(args.targets) != 1:
            Console().print("[bold red]Usage:[/bold red] marathon remote <ssh-host>")
            return 2
        return run_remote_dashboard(args.targets[0], instance=instance)
    if args.command == "tune":
        return run_dyno_dashboard(instance)
    if args.command == "setup":
        return run_setup_dashboard(instance)
    if args.command in {"resume", "fork"}:
        codex_args = [args.command, *args.targets]
        return launch_codex(codex_args)
    if args.command == "exec":
        codex_args = ["exec", *args.targets]
        return launch_codex(codex_args)
    if args.command == "codex":
        return launch_codex(args.targets)
    frontend = args.command if args.command in {"hermes", "direct"} else None
    return (
        run_dashboard(frontend)
        if instance is None
        else run_dashboard(frontend, instance)
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
