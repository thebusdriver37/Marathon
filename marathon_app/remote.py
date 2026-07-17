"""SSH-backed Marathon client and foreground Linux runtime host protocol."""

from __future__ import annotations

import contextlib
import ipaddress
import json
import os
import queue
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterator

from . import __version__
from .catalog import (
    Family,
    Model,
    Profile,
    discover_models,
    find_model,
    find_profile,
    profiles_for_model,
    settings,
)
from .runtime import CONFIG_DIR, USER_STATE_DIR, Runtime


PROTOCOL_VERSION = 1
CATALOG_PREFIX = "MARATHON_REMOTE_CATALOG "
PROGRESS_PREFIX = "MARATHON_REMOTE_PROGRESS "
READY_PREFIX = "MARATHON_REMOTE_READY "
ERROR_PREFIX = "MARATHON_REMOTE_ERROR "
REMOTE_SELECTION_FILE = CONFIG_DIR / "remote-selections.json"


@dataclass(frozen=True)
class RemoteCatalog:
    host: str
    remote_version: str
    router_port: int
    models: list[Model]


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _require_loopback_bindings() -> None:
    configured = settings()
    exposed: list[str] = []
    if not _is_loopback(configured.llama_host):
        exposed.append(f"llama_host={configured.llama_host}")
    if not _is_loopback(configured.router_host):
        exposed.append(f"router_host={configured.router_host}")
    if exposed:
        raise RuntimeError(
            "remote mode requires loopback-only backend bindings; fix "
            + ", ".join(exposed)
        )


def _profile_payload(profile: Profile) -> dict[str, object]:
    payload = asdict(profile)
    payload["extra_args"] = list(profile.extra_args)
    payload["frontends"] = list(profile.frontends)
    return payload


def _profile_from_payload(payload: dict[str, object]) -> Profile:
    return Profile(
        id=str(payload["id"]),
        display_name=str(payload["display_name"]),
        description=str(payload.get("description") or ""),
        context=int(payload["context"]),
        batch=int(payload["batch"]),
        ubatch=int(payload["ubatch"]),
        parallel=int(payload["parallel"]),
        gpu_layers=str(payload["gpu_layers"]),
        split_mode=str(payload["split_mode"]),
        tensor_split=str(payload.get("tensor_split") or ""),
        main_gpu=int(payload.get("main_gpu") or 0),
        cache_k=str(payload["cache_k"]),
        cache_v=str(payload["cache_v"]),
        flash_attention=str(payload["flash_attention"]),
        extra_args=tuple(str(item) for item in payload.get("extra_args", [])),
        confidence=str(payload.get("confidence") or "baseline"),
        frontends=tuple(str(item) for item in payload.get("frontends", [])),
        tool_thinking_budget=(
            int(payload["tool_thinking_budget"])
            if payload.get("tool_thinking_budget") is not None
            else None
        ),
        parallel_tool_calls=bool(payload.get("parallel_tool_calls", False)),
    )


def remote_catalog_payload() -> dict[str, object]:
    """Describe models on the GPU host without starting a backend."""

    _require_loopback_bindings()
    models: list[dict[str, object]] = []
    for model in discover_models():
        models.append(
            {
                "id": model.id,
                "display_name": model.display_name,
                "path": str(model.path),
                "size_bytes": model.size_bytes,
                "quant": model.quant,
                "family": {
                    "id": model.family.id,
                    "display_name": model.family.display_name,
                    "backend": model.family.backend,
                    "default_profile": model.family.default_profile,
                },
                "profiles": [
                    _profile_payload(profile) for profile in profiles_for_model(model)
                ],
            }
        )
    return {
        "protocol": PROTOCOL_VERSION,
        "marathon_version": __version__,
        "router_port": settings().router_port,
        "models": models,
    }


def _models_from_payload(payload: dict[str, object]) -> list[Model]:
    result: list[Model] = []
    raw_models = payload.get("models")
    if not isinstance(raw_models, list):
        raise ValueError("remote catalog does not contain a model list")
    for raw in raw_models:
        if not isinstance(raw, dict) or not isinstance(raw.get("family"), dict):
            raise ValueError("remote catalog contains an invalid model")
        family_data = raw["family"]
        raw_profiles = raw.get("profiles")
        if not isinstance(raw_profiles, list):
            raise ValueError(f"remote model {raw.get('id')} has no profiles")
        profiles = tuple(
            _profile_from_payload(item)
            for item in raw_profiles
            if isinstance(item, dict)
        )
        family = Family(
            id=str(family_data["id"]),
            display_name=str(family_data["display_name"]),
            patterns=(),
            backend=str(family_data.get("backend") or "remote"),
            default_profile=str(family_data["default_profile"]),
            profiles=profiles,
        )
        result.append(
            Model(
                id=str(raw["id"]),
                display_name=str(raw["display_name"]),
                path=Path(str(raw.get("path") or f"/remote/{raw['id']}.gguf")),
                size_bytes=int(raw["size_bytes"]),
                family=family,
                quant=str(raw.get("quant") or "GGUF"),
            )
        )
    return result


def _ssh_binary() -> str:
    configured = os.environ.get("MARATHON_SSH_BIN")
    if configured:
        return configured
    binary = shutil.which("ssh")
    if not binary:
        raise RuntimeError("OpenSSH is required for Marathon remote mode")
    return binary


def _ssh_options() -> list[str]:
    return [
        "-T",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10",
        "-o", "ServerAliveInterval=15",
        "-o", "ServerAliveCountMax=3",
    ]


def _remote_shell_command(*arguments: str) -> str:
    configured = os.environ.get("MARATHON_REMOTE_BIN")
    quoted_arguments = shlex.join(list(arguments))
    if configured:
        binary = shlex.join(shlex.split(configured))
        return f"exec {binary} {quoted_arguments}"
    return (
        "marathon_bin=$(command -v marathon 2>/dev/null || true); "
        '[ -n "$marathon_bin" ] || marathon_bin="$HOME/.local/bin/marathon"; '
        f'exec "$marathon_bin" {quoted_arguments}'
    )


def _extract_protocol_payload(output: str, prefix: str) -> dict[str, object]:
    for line in reversed(output.splitlines()):
        if line.startswith(prefix):
            value = json.loads(line[len(prefix):])
            if isinstance(value, dict):
                return value
    raise ValueError(f"remote Marathon did not emit {prefix.strip()}")


def fetch_remote_catalog(host: str) -> RemoteCatalog:
    if not host or host.startswith("-") or any(character.isspace() for character in host):
        raise ValueError("SSH host must be a hostname, alias, or user@hostname")
    command = [
        _ssh_binary(),
        *_ssh_options(),
        host,
        _remote_shell_command("remote-host", "catalog"),
    ]
    try:
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=45,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"SSH catalog request to {host} timed out") from error
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"SSH connection to {host} failed: {detail or result.returncode}")
    try:
        payload = _extract_protocol_payload(result.stdout, CATALOG_PREFIX)
    except (ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{error}; update Marathon on {host}") from error
    if int(payload.get("protocol") or 0) != PROTOCOL_VERSION:
        raise RuntimeError(
            f"Marathon remote protocol mismatch: client {PROTOCOL_VERSION}, "
            f"host {payload.get('protocol')}"
        )
    return RemoteCatalog(
        host=host,
        remote_version=str(payload.get("marathon_version") or "unknown"),
        router_port=int(payload["router_port"]),
        models=_models_from_payload(payload),
    )


def load_remote_selection(host: str) -> dict[str, str]:
    try:
        payload = json.loads(REMOTE_SELECTION_FILE.read_text(encoding="utf-8"))
        value = payload.get("hosts", {}).get(host, {})
    except (OSError, AttributeError, json.JSONDecodeError):
        return {}
    return {
        key: str(value[key])
        for key in ("model", "profile", "frontend")
        if value.get(key)
    }


def save_remote_selection(host: str, model: Model, profile: Profile, frontend: str) -> None:
    try:
        payload = json.loads(REMOTE_SELECTION_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {"schema": 1, "hosts": {}}
    hosts = payload.setdefault("hosts", {})
    if not isinstance(hosts, dict):
        hosts = {}
        payload["hosts"] = hosts
    hosts[host] = {
        "model": model.id,
        "profile": profile.id,
        "frontend": frontend,
    }
    REMOTE_SELECTION_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = REMOTE_SELECTION_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(REMOTE_SELECTION_FILE)


def _available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


class RemoteRuntime:
    """Runtime-shaped adapter used by local Mac frontends over an SSH tunnel."""

    def __init__(
        self,
        host: str,
        router_port: int,
        model: Model,
        profile: Profile,
    ) -> None:
        self.host = host
        self.remote_router_port = router_port
        self.model = model
        self.profile = profile
        self.local_port = _available_port()
        self._context_window = profile.context
        self._auto_compact_token_limit: int | None = None
        self._truncation_limit: int | None = None
        self.process: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._output_tail: list[str] = []
        self._write_lock = threading.Lock()
        self._cleaned = False
        self.run_id: str | None = None
        self.run_log: str | None = None
        safe_host = re.sub(r"[^a-zA-Z0-9_.-]+", "-", host).strip("-") or "host"
        self.catalog_file = (
            USER_STATE_DIR / "remote" / safe_host / "codex-models.json"
        )

    @property
    def router_url(self) -> str:
        return f"http://127.0.0.1:{self.local_port}"

    @property
    def context_window(self) -> int:
        return self._context_window

    @property
    def context_reserve_tokens(self) -> int:
        configured = os.environ.get("MARATHON_CONTEXT_RESERVE_TOKENS")
        if configured:
            try:
                return min(self.context_window // 2, max(1, int(configured)))
            except ValueError:
                pass
        reserve = max(12_288, min(32_768, self.context_window // 8))
        return min(self.context_window // 2, reserve)

    @property
    def auto_compact_token_limit(self) -> int:
        return self._auto_compact_token_limit or max(
            1, self.context_window - self.context_reserve_tokens
        )

    @property
    def truncation_limit(self) -> int:
        if self._truncation_limit is not None:
            return self._truncation_limit
        guard = max(2_048, min(8_192, self.context_window // 20))
        return max(1, self.auto_compact_token_limit - guard)

    def _read_lines(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        try:
            for raw in self.process.stdout:
                line = raw.rstrip("\r\n")
                self._output_tail.append(line)
                del self._output_tail[:-80]
                self._lines.put(line)
        finally:
            self._lines.put(None)

    def _command(self) -> list[str]:
        forward = (
            f"127.0.0.1:{self.local_port}:"
            f"127.0.0.1:{self.remote_router_port}"
        )
        return [
            _ssh_binary(),
            *_ssh_options(),
            "-o", "ExitOnForwardFailure=yes",
            "-L", forward,
            self.host,
            _remote_shell_command(
                "remote-host", "run", self.model.id, self.profile.id
            ),
        ]

    def start(self, progress: Callable[[str], None] | None = None) -> None:
        self.process = subprocess.Popen(
            self._command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            start_new_session=True,
        )
        self._reader = threading.Thread(
            target=self._read_lines,
            name="marathon-remote-ssh-output",
            daemon=True,
        )
        self._reader.start()
        timeout = max(
            30, int(os.environ.get("MARATHON_REMOTE_START_TIMEOUT_SECONDS", "420"))
        )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                line = self._lines.get(timeout=1)
            except queue.Empty:
                if self.process.poll() is not None:
                    break
                continue
            if line is None:
                break
            try:
                if line.startswith(PROGRESS_PREFIX):
                    payload = json.loads(line[len(PROGRESS_PREFIX):])
                    if progress:
                        progress(str(payload.get("message") or "Preparing remote GPUs"))
                    continue
                if line.startswith(ERROR_PREFIX):
                    payload = json.loads(line[len(ERROR_PREFIX):])
                    raise RuntimeError(str(payload.get("error") or "remote runtime failed"))
                if line.startswith(READY_PREFIX):
                    payload = json.loads(line[len(READY_PREFIX):])
                    self._context_window = int(payload["context"])
                    self._auto_compact_token_limit = int(
                        payload["auto_compact_token_limit"]
                    )
                    self._truncation_limit = int(payload["truncation_limit"])
                    self.run_id = str(payload.get("run_id") or "") or None
                    self.run_log = str(payload.get("run_log") or "") or None
                    self._write_local_catalog()
                    return
            except json.JSONDecodeError:
                continue
        detail = "\n".join(self._output_tail[-12:]).strip()
        if self.process.poll() is None:
            raise TimeoutError(f"remote model did not become ready within {timeout}s")
        raise RuntimeError(
            f"remote Marathon exited before becoming ready"
            f"{': ' + detail if detail else ''}"
        )

    def _write_local_catalog(self) -> None:
        try:
            with urllib.request.urlopen(
                f"{self.router_url}/v1/models", timeout=10
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
            raise RuntimeError("SSH tunnel opened, but the Marathon router is unreachable") from error
        self.catalog_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.catalog_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.catalog_file)

    def _send(self, payload: dict[str, object]) -> None:
        if not self.process or not self.process.stdin or self.process.poll() is not None:
            return
        with self._write_lock:
            try:
                self.process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
                self.process.stdin.flush()
            except (BrokenPipeError, OSError, TypeError, ValueError):
                pass

    def record(
        self,
        event: str,
        data: dict[str, object] | None = None,
        *,
        level: str = "info",
    ) -> None:
        self._send(
            {
                "op": "event",
                "event": event,
                "data": data or {},
                "level": level,
            }
        )

    @contextlib.contextmanager
    def frontend_signals(self) -> Iterator[None]:
        old = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            yield
        finally:
            signal.signal(signal.SIGINT, old)

    def cleanup(self) -> None:
        if self._cleaned:
            return
        self._cleaned = True
        process = self.process
        if process is None:
            return
        self._send({"op": "stop"})
        if process.stdin:
            with contextlib.suppress(OSError):
                process.stdin.close()
        try:
            process.wait(timeout=35)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        if self._reader:
            self._reader.join(timeout=2)


def _protocol_print(prefix: str, payload: dict[str, object]) -> None:
    print(prefix + json.dumps(payload, separators=(",", ":")), flush=True)


def print_remote_catalog() -> int:
    _protocol_print(CATALOG_PREFIX, remote_catalog_payload())
    return 0


def run_remote_host(model_id: str, profile_id: str) -> int:
    runtime: Runtime | None = None
    try:
        _require_loopback_bindings()
        model = find_model(model_id, discover_models())
        profile = find_profile(model, profile_id)
        runtime = Runtime(model, profile)
        runtime.start(
            lambda message: _protocol_print(
                PROGRESS_PREFIX, {"message": message}
            )
        )
        _protocol_print(
            READY_PREFIX,
            {
                "protocol": PROTOCOL_VERSION,
                "model": model.id,
                "profile": profile.id,
                "context": runtime.context_window,
                "auto_compact_token_limit": runtime.auto_compact_token_limit,
                "truncation_limit": runtime.truncation_limit,
                "run_id": runtime.run_id,
                "run_log": str(runtime.run_log) if runtime.run_log else None,
            },
        )
        for raw in sys.stdin:
            try:
                command = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(command, dict):
                continue
            if command.get("op") == "stop":
                break
            if command.get("op") != "event":
                continue
            event = command.get("event")
            data = command.get("data")
            level = command.get("level")
            if isinstance(event, str) and isinstance(data, dict):
                runtime.record(
                    event,
                    {**data, "client": "ssh"},
                    level=level if level in {"info", "error"} else "info",
                )
        return 0
    except (OSError, RuntimeError, TimeoutError, ValueError) as error:
        _protocol_print(ERROR_PREFIX, {"error": str(error)})
        return 2
    finally:
        if runtime is not None:
            runtime.cleanup()


def run_remote_host_command(arguments: list[str]) -> int:
    if arguments == ["catalog"]:
        return print_remote_catalog()
    if len(arguments) == 3 and arguments[0] == "run":
        return run_remote_host(arguments[1], arguments[2])
    print("Usage: marathon remote-host catalog | run <model> <profile>", file=sys.stderr)
    return 2
