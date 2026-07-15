"""Same-terminal Codex and direct-chat frontends."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import urllib.request
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from .runtime import Runtime


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


def codex_command(runtime: Runtime, extra_args: list[str] | None = None) -> list[str]:
    binary = _codex_binary()
    provider = (
        'model_providers.marathon_local={ name = "Marathon Local", '
        f'base_url = "{runtime.router_url}/v1", wire_api = "responses", '
        "requires_openai_auth = false, supports_websockets = true }"
    )
    command = [
        binary,
        "-c", provider,
        "-c", 'model_provider="marathon_local"',
        "-m", runtime.model.alias,
        "-c", f"model_context_window={runtime.context_window}",
        "-c", f"model_auto_compact_token_limit={runtime.auto_compact_token_limit}",
        "-c", f"model_catalog_json={json.dumps(str(runtime.catalog_file))}",
        "-c", 'web_search="cached"',
    ]
    command.extend(extra_args or [])
    return command


def run_codex(runtime: Runtime, extra_args: list[str] | None = None) -> int:
    command = codex_command(runtime, extra_args)
    environment = os.environ.copy()
    with runtime.frontend_signals():
        result = subprocess.run(
            command,
            cwd=Path.cwd(),
            env=environment,
            preexec_fn=_restore_sigint,
            check=False,
        )
    return result.returncode


def _stream_chat(
    runtime: Runtime,
    messages: list[dict[str, str]],
    on_text: Callable[[str], None] | None = None,
) -> str:
    payload = json.dumps(
        {
            "model": runtime.model.alias,
            "messages": messages,
            "stream": True,
            "temperature": 0.7,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{runtime.router_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    parts: list[str] = []
    with urllib.request.urlopen(request, timeout=3600) as response:
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
                delta = event.get("choices", [{}])[0].get("delta", {})
                text = delta.get("content")
            except (json.JSONDecodeError, IndexError, AttributeError):
                continue
            if isinstance(text, str):
                parts.append(text)
                if on_text:
                    on_text(text)
    return "".join(parts)


def direct_chat(runtime: Runtime, console: Console) -> None:
    messages: list[dict[str, str]] = []
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
            return
        if not prompt:
            continue
        if prompt in {"/back", "/exit", "/quit"}:
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
            console.print(f"[bold red]Request failed:[/bold red] {error}")
            messages.pop()
            continue
        if answer:
            messages.append({"role": "assistant", "content": answer})
        else:
            console.print("[yellow]The model returned no text.[/yellow]")
            messages.pop()
