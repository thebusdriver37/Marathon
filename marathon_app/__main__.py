"""Command entry point for the unified Marathon runtime."""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from rich.console import Console
from rich.table import Table

from . import __version__
from .catalog import discover_models, format_size, settings
from .model_library import register_model_root
from .runtime import SESSION_FILE, request_stop
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
        table.add_row(model.display_name, format_size(model.size_bytes), model.family.default_profile, str(model.path))
    console.print(table)
    return 0 if models else 1


def _status() -> int:
    console = Console()
    config = settings()
    try:
        session = json.loads(SESSION_FILE.read_text(encoding="utf-8"))
        with urllib.request.urlopen(
            f"http://{config.router_host}:{config.router_port}/health", timeout=2
        ) as response:
            health = json.loads(response.read().decode("utf-8"))
    except (OSError, json.JSONDecodeError, urllib.error.URLError):
        console.print("[dim]Marathon is stopped.[/dim]")
        return 1
    console.print(f"[green]● Marathon running[/green] · {session.get('model')} / {session.get('profile')}")
    console.print(f"Supervisor PID {session.get('supervisor_pid')} · backend {health.get('backend_health') or 'ready'}")
    if session.get("run_log"):
        console.print(f"Trace: {session['run_log']}", style="dim")
    return 0


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{int(seconds // 60)}m {seconds % 60:.0f}s"


def _run_is_active(path: Path) -> bool:
    try:
        session = json.loads(SESSION_FILE.read_text(encoding="utf-8"))
        pid = int(session["supervisor_pid"])
        run_log = Path(session["run_log"]).expanduser().resolve()
        os.kill(pid, 0)
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False
    return run_log == path.resolve()


def _report(target: str | None) -> int:
    console = Console()
    try:
        path = resolve_run(target)
        summary = summarize_run(path, live=_run_is_active(path))
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


def _compare(targets: list[str]) -> int:
    console = Console()
    if len(targets) != 2:
        console.print("[bold red]Usage:[/bold red] marathon compare <run-a> <run-b>")
        return 2
    try:
        paths = [resolve_run(target) for target in targets]
        left, right = (
            summarize_run(path, live=_run_is_active(path)) for path in paths
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="marathon", description="One-command local AI runtime")
    parser.add_argument("--version", action="version", version=f"Marathon {__version__}")
    parser.add_argument(
        "command", nargs="?", choices=("dashboard", "codex", "hermes", "direct", "remote", "remote-host", "tune", "setup", "models", "status", "stop", "report", "compare", "resume", "fork"),
        default="codex",
    )
    parser.add_argument("targets", nargs="*")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "models":
        return _models(args.targets)
    if args.command == "status":
        return _status()
    if args.command == "stop":
        stopped = request_stop()
        Console().print("[green]Stop requested.[/green]" if stopped else "[dim]Marathon is already stopped.[/dim]")
        return 0
    if args.command == "report":
        return _report(args.targets[0] if args.targets else None)
    if args.command == "compare":
        return _compare(args.targets)
    if args.command == "remote-host":
        return run_remote_host_command(args.targets)
    if args.command == "remote":
        if len(args.targets) != 1:
            Console().print("[bold red]Usage:[/bold red] marathon remote <ssh-host>")
            return 2
        return run_remote_dashboard(args.targets[0])
    if args.command == "tune":
        return run_dyno_dashboard()
    if args.command == "setup":
        return run_setup_dashboard()
    if args.command in {"resume", "fork"}:
        return run_codex_default([args.command, *args.targets])
    if args.command == "codex":
        return run_codex_default()
    return run_dashboard(
        args.command if args.command in {"hermes", "direct"} else None
    )


if __name__ == "__main__":
    raise SystemExit(main())
