"""Same-terminal Codex, Hermes, and direct-chat frontends."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from .codex_home import codex_environment
from .codex_telemetry import snapshot_sessions, summarize_session_changes
from .runtime import Runtime
from .router_security import open_api_request


MARATHON_STATUS_LINE = [
    "model-with-reasoning",
    "tokens-per-second",
    "context-remaining",
    "context-window-size",
    "context-tokens",
]


def _restore_sigint() -> None:
    signal.signal(signal.SIGINT, signal.SIG_DFL)


def _codex_binary() -> str:
    configured = os.environ.get("MARATHON_CODEX_BIN")
    if configured:
        return configured
    data_home = Path(
        os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")
    ).expanduser()
    patched = data_home / "marathon" / "bin" / "codex"
    return str(patched) if patched.is_file() and os.access(patched, os.X_OK) else "codex"


def _hermes_binary() -> str:
    return os.environ.get("MARATHON_HERMES_BIN") or "hermes"


def _marathon_cli_name(instance: str | None = None) -> str:
    configured = os.environ.get("CODEX_CLI_NAME")
    if configured and configured.strip():
        command = configured.strip()
    elif shutil.which("marathon"):
        command = "marathon"
    else:
        command = str(Path(__file__).resolve().parents[1] / "bin" / "marathon")
    return f"{command} --instance {instance}" if instance else command


def _codex_features(binary: str) -> set[str]:
    candidate = Path(binary).expanduser()
    if not candidate.is_absolute():
        resolved = shutil.which(binary)
        if not resolved:
            return set()
        candidate = Path(resolved)
    marker = Path(f"{candidate.resolve()}.features")
    try:
        return {
            line.strip()
            for line in marker.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    except OSError:
        return set()


def codex_command(
    runtime: Runtime,
    extra_args: list[str] | None = None,
    *,
    shared_profile: str | None = None,
) -> list[str]:
    binary = _codex_binary()
    provider = (
        'model_providers.marathon-local={ name = "Marathon Local", '
        f'base_url = "{runtime.router_url}/v1", wire_api = "responses", '
        "requires_openai_auth = false, supports_websockets = true, "
        'env_key = "MARATHON_ROUTER_TOKEN", '
        "stream_idle_timeout_ms = 900000 }"
    )
    command = [
        binary,
        "-c", provider,
        "-c", 'model_provider="marathon-local"',
        "-m", runtime.model.alias,
        "-c", f"model_catalog_json={json.dumps(str(runtime.catalog_file))}",
        "-c", 'web_search="cached"',
    ]
    if shared_profile:
        command.extend(["--profile", shared_profile])
    if "tokens-per-second" in _codex_features(binary):
        command.extend(
            ["-c", f"tui.status_line={json.dumps(MARATHON_STATUS_LINE)}"]
        )
    command.extend(extra_args or [])
    return command


def require_hardened_codex(binary: str) -> None:
    if "local-runtime-security" not in _codex_features(binary):
        raise RuntimeError("Marathon requires its hardened frontend; run: marathon build-codex")


def run_codex(runtime: Runtime, extra_args: list[str] | None = None) -> int:
    instance = getattr(getattr(runtime, "instance", None), "name", None)
    environment, codex_home, shared_profile = codex_environment(instance=instance)
    command = codex_command(runtime, extra_args, shared_profile=shared_profile)
    require_hardened_codex(command[0])
    environment["MARATHON_ROUTER_TOKEN"] = runtime.router_token
    environment["CODEX_CLI_NAME"] = _marathon_cli_name(instance)
    before = snapshot_sessions(codex_home)
    started = time.monotonic()
    runtime.record(
        "frontend.started",
        {
            "frontend": "codex",
            "binary": command[0],
            "cwd": str(Path.cwd()),
            "codex_home": str(codex_home),
        },
    )
    with runtime.frontend_signals():
        result = subprocess.run(
            command,
            cwd=Path.cwd(),
            env=environment,
            preexec_fn=_restore_sigint,
            check=False,
        )
    summaries = summarize_session_changes(
        before,
        cwd=Path.cwd(),
        provider="marathon-local",
        codex_home=codex_home,
    )
    for summary in summaries:
        runtime.record("codex.session.completed", summary)
    runtime.record(
        "frontend.completed",
        {
            "frontend": "codex",
            "returncode": result.returncode,
            "duration_ms": (time.monotonic() - started) * 1000.0,
            "sessions_changed": len(summaries),
        },
        level="info" if result.returncode in (0, 130) else "error",
    )
    return result.returncode


def hermes_command(runtime: Runtime, extra_args: list[str] | None = None) -> list[str]:
    """Build a Hermes invocation that keeps the user's normal agent setup."""

    command = [
        _hermes_binary(),
        "chat",
        "--model", runtime.model.alias,
        "--provider", "custom",
    ]
    command.extend(extra_args or [])
    return command


def run_hermes(runtime: Runtime, extra_args: list[str] | None = None) -> int:
    """Run Hermes in this terminal against Marathon's supervised backend."""

    command = hermes_command(runtime, extra_args)
    environment = os.environ.copy()
    for key in ("NO_PROXY", "no_proxy"):
        environment[key] = ",".join(filter(None, [environment.get(key), "127.0.0.1,localhost,::1"]))
    environment.update(
        {
            # Hermes deliberately treats its YAML as the normal source of truth,
            # but CUSTOM_BASE_URL is its supported per-process custom-provider
            # override.  This leaves ~/.hermes, memory, rules, and skills intact.
            "CUSTOM_BASE_URL": f"{runtime.router_url}/v1",
            "CUSTOM_API_KEY": runtime.router_token,
            "HERMES_INFERENCE_MODEL": runtime.model.alias,
            "HERMES_INFERENCE_PROVIDER": "custom",
        }
    )
    started = time.monotonic()
    runtime.record(
        "frontend.started",
        {
            "frontend": "hermes",
            "binary": command[0],
            "cwd": str(Path.cwd()),
            "endpoint": f"{runtime.router_url}/v1",
            "context": runtime.context_window,
        },
    )
    with runtime.frontend_signals():
        result = subprocess.run(
            command,
            cwd=Path.cwd(),
            env=environment,
            preexec_fn=_restore_sigint,
            check=False,
        )
    runtime.record(
        "frontend.completed",
        {
            "frontend": "hermes",
            "returncode": result.returncode,
            "duration_ms": (time.monotonic() - started) * 1000.0,
        },
        level="info" if result.returncode in (0, 130) else "error",
    )
    return result.returncode


def _stream_chat(
    runtime: Runtime,
    messages: list[dict[str, str]],
    on_text: Callable[[str], None] | None = None,
) -> str:
    started = time.monotonic()
    payload = json.dumps(
        {
            "model": runtime.model.alias,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": (
                runtime.profile.temperature
                if runtime.profile.temperature is not None
                else 0.7
            ),
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{runtime.router_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", **runtime.router_headers},
        method="POST",
    )
    parts: list[str] = []
    first_text_at: float | None = None
    usage: dict[str, object] | None = None
    runtime.record(
        "direct.turn.started",
        {
            "message_count": len(messages),
            "input_characters": sum(len(item.get("content", "")) for item in messages),
        },
    )
    with open_api_request(request, timeout=3600) as response:
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]
                delta = event.get("choices", [{}])[0].get("delta", {})
                text = delta.get("content")
            except (json.JSONDecodeError, IndexError, AttributeError):
                continue
            if isinstance(text, str):
                if first_text_at is None:
                    first_text_at = time.monotonic()
                parts.append(text)
                if on_text:
                    on_text(text)
    answer = "".join(parts)
    ended = time.monotonic()
    runtime.record(
        "direct.turn.completed",
        {
            "duration_ms": (ended - started) * 1000.0,
            "ttft_ms": (first_text_at - started) * 1000.0 if first_text_at else None,
            "output_characters": len(answer),
            "chunks": len(parts),
            "usage": usage,
        },
    )
    return answer


def direct_chat(runtime: Runtime, console: Console) -> None:
    messages: list[dict[str, str]] = []
    frontend_started = time.monotonic()
    runtime.record("frontend.started", {"frontend": "direct"})
    console.clear()
    console.print(
        Panel.fit(
            f"[bold cyan]Direct Chat[/bold cyan]\n{runtime.model.display_name} · "
            f"{runtime.profile.display_name} · {runtime.profile.context:,} tokens\n\n"
            "[dim]No tools, memory, skills, or coding-agent harness. "
            "Commands: /new, /back, /help[/dim]",
            border_style="cyan",
        )
    )
    while True:
        try:
            prompt = Prompt.ask("\n[bold green]You[/bold green]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            runtime.record(
                "frontend.completed",
                {"frontend": "direct", "duration_ms": (time.monotonic() - frontend_started) * 1000.0},
            )
            return
        if not prompt:
            continue
        if prompt in {"/back", "/exit", "/quit"}:
            runtime.record(
                "frontend.completed",
                {"frontend": "direct", "duration_ms": (time.monotonic() - frontend_started) * 1000.0},
            )
            return
        if prompt == "/new":
            messages.clear()
            console.print("[dim]Conversation cleared.[/dim]")
            continue
        if prompt == "/help":
            console.print("[dim]/new clears context · /back returns to Marathon[/dim]")
            continue
        messages.append({"role": "user", "content": prompt})
        console.print("\n[bold magenta]Model[/bold magenta]")
        try:
            answer = _stream_chat(
                runtime,
                messages,
                lambda text: console.print(text, end="", markup=False, highlight=False),
            )
            console.print()
        except Exception as error:
            runtime.record("direct.turn.error", {"error": str(error)}, level="error")
            console.print(f"[bold red]Request failed:[/bold red] {error}")
            messages.pop()
            continue
        if answer:
            messages.append({"role": "assistant", "content": answer})
        else:
            console.print("[yellow]The model returned no text.[/yellow]")
            messages.pop()
