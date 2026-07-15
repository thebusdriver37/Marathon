"""Command entry point for the unified Marathon runtime."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request

from rich.console import Console
from rich.table import Table

from . import __version__
from .catalog import discover_models, format_size, settings
from .runtime import SESSION_FILE, request_stop
from .ui import run_dashboard


def _models() -> int:
    console = Console()
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
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="marathon", description="One-command local AI runtime")
    parser.add_argument("--version", action="version", version=f"Marathon {__version__}")
    parser.add_argument(
        "command", nargs="?", choices=("dashboard", "codex", "direct", "models", "status", "stop"),
        default="dashboard",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "models":
        return _models()
    if args.command == "status":
        return _status()
    if args.command == "stop":
        stopped = request_stop()
        Console().print("[green]Stop requested.[/green]" if stopped else "[dim]Marathon is already stopped.[/dim]")
        return 0
    return run_dashboard(args.command if args.command in {"codex", "direct"} else None)


if __name__ == "__main__":
    raise SystemExit(main())
