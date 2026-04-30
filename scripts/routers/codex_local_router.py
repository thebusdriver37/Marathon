#!/usr/bin/env python3
"""Local multi-model router for running Codex against llama.cpp backends.

This exposes a Codex-compatible `/v1/models` catalog with multiple local model
profiles and normalizes Codex Responses API requests before forwarding them to
the selected local backend. Only one heavyweight backend is kept warm at a time;
switching models starts the requested backend and stops the others.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


@dataclass(frozen=True)
class ModelProfile:
    slug: str
    alias: str
    display_name: str
    description: str
    launcher: str
    model_paths: tuple[str, ...]
    target: str
    context_window: int
    auto_compact_token_limit: int
    truncation_limit: int

    @property
    def port(self) -> int:
        return int(self.target.rsplit(":", 1)[-1])

    def resolved_model_path(self) -> str | None:
        for candidate in self.model_paths:
            if Path(candidate).exists():
                return candidate
        return None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _prompt_path() -> Path:
    configured = os.getenv("MARATHON_PROMPT_FILE") or os.getenv("CODEX_QWEN_PROMPT_FILE")
    if configured:
        return Path(configured).expanduser().resolve()
    return _repo_root() / "codex" / "codex-rs" / "models-manager" / "prompt.md"


def _base_instructions() -> str:
    prompt_path = _prompt_path()
    if not prompt_path.exists():
        raise FileNotFoundError(
            "Codex prompt file not found. Expected "
            f"{prompt_path}. Initialize the Codex submodule or set MARATHON_PROMPT_FILE."
        )
    return prompt_path.read_text(encoding="utf-8")


def _model_candidates(env_var: str, relative_paths: tuple[str, ...]) -> tuple[str, ...]:
    configured = os.getenv(env_var)
    if configured:
        return (configured,)

    models_dir = Path(os.getenv("MARATHON_MODELS_DIR") or os.getenv("MODELS_DIR") or Path.home() / "models")
    return tuple(str(models_dir / relative_path) for relative_path in relative_paths)


def _profiles() -> dict[str, ModelProfile]:
    root = _repo_root()
    return {
        "qwen3.6-27b-q4-128k": ModelProfile(
            slug="qwen3.6-27b-q4-128k",
            alias="qwen3.6-27b-q4-128k",
            display_name="Qwen3.6 27B Q4 128K",
            description="Long-context local Qwen3.6 27B Q4 profile",
            launcher=str(root / "scripts/launchers/server_27b_128k.sh"),
            model_paths=_model_candidates(
                "QWEN36_27B_GGUF",
                (
                    "Qwen3.6-27B-GGUF/qwen3.6-27b-q4_k_m.gguf",
                    "Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf",
                ),
            ),
            target="http://127.0.0.1:18091",
            context_window=131072,
            auto_compact_token_limit=115000,
            truncation_limit=110000,
        ),
        "qwen3.6-27b-q4": ModelProfile(
            slug="qwen3.6-27b-q4",
            alias="qwen3.6-27b-q4",
            display_name="Qwen3.6 27B Q4 32K",
            description="Fast local Qwen3.6 27B Q4 profile",
            launcher=str(root / "scripts/launchers/server_27b_fast.sh"),
            model_paths=_model_candidates(
                "QWEN36_27B_GGUF",
                (
                    "Qwen3.6-27B-GGUF/qwen3.6-27b-q4_k_m.gguf",
                    "Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf",
                ),
            ),
            target="http://127.0.0.1:18090",
            context_window=32768,
            auto_compact_token_limit=28000,
            truncation_limit=26000,
        ),
        "qwen3.6-35b-a3b": ModelProfile(
            slug="qwen3.6-35b-a3b",
            alias="qwen3.6-35b-a3b",
            display_name="Qwen3.6 35B A3B 32K",
            description="Single-GPU specialist local Qwen3.6 35B A3B profile",
            launcher=str(root / "scripts/launchers/server_35b_a3b.sh"),
            model_paths=_model_candidates(
                "QWEN36_35B_A3B_GGUF",
                (
                    "Qwen3.6-35B-A3B-GGUF/qwen3.6-35b-a3b-q4_k_m.gguf",
                    "Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf",
                ),
            ),
            target="http://127.0.0.1:18092",
            context_window=32768,
            auto_compact_token_limit=28000,
            truncation_limit=26000,
        ),
    }


def _available_profiles() -> dict[str, ModelProfile]:
    available: dict[str, ModelProfile] = {}
    for slug, profile in _profiles().items():
        if Path(profile.launcher).exists() and profile.resolved_model_path():
            available[slug] = profile
    return available


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict):
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
        elif isinstance(item, str):
            parts.append(item)
    return "\n".join(part for part in parts if part)


def normalize_responses_request(data: dict[str, Any]) -> dict[str, Any]:
    tools = data.get("tools")
    if isinstance(tools, list):
        data["tools"] = [tool for tool in tools if tool.get("type") == "function"]

    input_items = data.get("input")
    if not isinstance(input_items, list):
        return data

    lifted_messages: list[str] = []
    normalized_input: list[Any] = []
    for item in input_items:
        if not isinstance(item, dict):
            normalized_input.append(item)
            continue

        role = item.get("role")
        if item.get("type") == "message" and role in {"developer", "system"}:
            text = _content_text(item.get("content"))
            if text:
                lifted_messages.append(text)
            continue

        normalized_input.append(item)

    if lifted_messages:
        existing = data.get("instructions")
        instructions = existing if isinstance(existing, str) else ""
        data["instructions"] = "\n\n".join(part for part in [instructions, *lifted_messages] if part)
        data["input"] = normalized_input

    return data


def _json_model_matches(url: str, expected_model: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/v1/models", timeout=3) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return False

    for key in ("data", "models"):
        value = payload.get(key)
        if not isinstance(value, list):
            continue
        for item in value:
            if not isinstance(item, dict):
                continue
            if any(item.get(field) == expected_model for field in ("id", "slug", "model", "name")):
                return True
    return False


class RouterState:
    def __init__(self, default_model: str, state_dir: Path, log_dir: Path, debug: bool = False):
        self.lock = threading.Lock()
        self.debug = debug
        self.state_dir = state_dir
        self.log_dir = log_dir
        self.available_profiles = self._refresh_profiles()
        if not self.available_profiles:
            raise RuntimeError("no available local model profiles found")
        if default_model not in self.available_profiles:
            default_model = next(iter(self.available_profiles))
        self.default_model = default_model
        self.current_model: str | None = None
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _refresh_profiles(self) -> dict[str, ModelProfile]:
        profiles = _available_profiles()
        if profiles:
            self.available_profiles = profiles
        return self.available_profiles

    def model_catalog(self) -> dict[str, Any]:
        self._refresh_profiles()
        instructions = _base_instructions()
        models = []
        data = []
        for profile in self.available_profiles.values():
            models.append(
                {
                    "slug": profile.slug,
                    "display_name": profile.display_name,
                    "description": profile.description,
                    "default_reasoning_level": None,
                    "supported_reasoning_levels": [],
                    "shell_type": "shell_command",
                    "visibility": "list",
                    "supported_in_api": True,
                    "priority": 0,
                    "additional_speed_tiers": [],
                    "availability_nux": None,
                    "upgrade": None,
                    "base_instructions": instructions,
                    "model_messages": None,
                    "supports_reasoning_summaries": False,
                    "default_reasoning_summary": "auto",
                    "support_verbosity": False,
                    "default_verbosity": None,
                    "apply_patch_tool_type": "function",
                    "web_search_tool_type": "text",
                    "truncation_policy": {"mode": "tokens", "limit": profile.truncation_limit},
                    "supports_parallel_tool_calls": False,
                    "supports_image_detail_original": False,
                    "context_window": profile.context_window,
                    "max_context_window": profile.context_window,
                    "auto_compact_token_limit": profile.auto_compact_token_limit,
                    "effective_context_window_percent": 90,
                    "experimental_supported_tools": [],
                    "input_modalities": ["text"],
                    "supports_search_tool": False,
                }
            )
            data.append(
                {
                    "id": profile.slug,
                    "object": "model",
                    "owned_by": "local-codex-router",
                    "description": profile.description,
                }
            )
        return {"models": models, "object": "list", "data": data}

    def resolve_model(self, requested_model: str | None) -> ModelProfile:
        self._refresh_profiles()
        model_key = (requested_model or self.default_model).strip()
        if model_key in self.available_profiles:
            return self.available_profiles[model_key]
        raise ValueError(f"unknown or unavailable local model '{model_key}'")

    def ensure_model(self, requested_model: str | None) -> ModelProfile:
        profile = self.resolve_model(requested_model)
        with self.lock:
            if self.current_model != profile.slug:
                self._stop_other_backends(profile.slug)
            if not self._profile_ready(profile):
                self._stop_profile(profile)
                self._start_profile(profile)
                self._wait_for_profile(profile)
            self.current_model = profile.slug
        return profile

    def _profile_pid_file(self, profile: ModelProfile) -> Path:
        return self.state_dir / f"{profile.slug}.pid"

    def _profile_log_file(self, profile: ModelProfile) -> Path:
        return self.log_dir / f"{profile.slug}.log"

    def _port_owner_pid(self, port: int) -> int | None:
        proc = subprocess.run(
            ["bash", "-lc", f"ss -ltnp '( sport = :{port} )' 2>/dev/null | sed -n 's/.*pid=\\([0-9]\\+\\).*/\\1/p' | head -n1"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        text = proc.stdout.strip()
        return int(text) if text.isdigit() else None

    def _pid_cmdline(self, pid: int) -> str:
        proc = subprocess.run(
            ["ps", "-p", str(pid), "-o", "cmd="],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        return proc.stdout.strip()

    def _profile_ready(self, profile: ModelProfile) -> bool:
        return _json_model_matches(profile.target, profile.alias)

    def _stop_other_backends(self, keep_slug: str) -> None:
        for slug, profile in self.available_profiles.items():
            if slug == keep_slug:
                continue
            self._stop_profile(profile)

    def _stop_profile(self, profile: ModelProfile) -> None:
        pid = self._port_owner_pid(profile.port)
        if pid is None:
            self._profile_pid_file(profile).unlink(missing_ok=True)
            return
        cmd = self._pid_cmdline(pid)
        if "llama-server" not in cmd:
            raise RuntimeError(f"port {profile.port} is occupied by unexpected process: {cmd}")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        for _ in range(10):
            time.sleep(1)
            if self._port_owner_pid(profile.port) is None:
                self._profile_pid_file(profile).unlink(missing_ok=True)
                return
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self._profile_pid_file(profile).unlink(missing_ok=True)

    def _start_profile(self, profile: ModelProfile) -> None:
        env = os.environ.copy()
        env["HOST"] = "127.0.0.1"
        env["PORT"] = str(profile.port)
        env["MODEL_ALIAS"] = profile.alias
        model_path = profile.resolved_model_path()
        if model_path:
            env["MODEL_PATH"] = model_path
        log_file = self._profile_log_file(profile)
        with log_file.open("ab") as handle:
            proc = subprocess.Popen(
                [profile.launcher],
                cwd=str(_repo_root()),
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self._profile_pid_file(profile).write_text(f"{proc.pid}\n", encoding="utf-8")

    def _wait_for_profile(self, profile: ModelProfile) -> None:
        for _ in range(240):
            if self._profile_ready(profile):
                return
            time.sleep(1)
        raise RuntimeError(
            f"backend for {profile.slug} did not become ready; see {self._profile_log_file(profile)}"
        )


class RouterHandler(BaseHTTPRequestHandler):
    state: RouterState

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/v1/models":
            self._send_json(200, self.state.model_catalog())
            return
        if self.path.rstrip("/") == "/health":
            self.state._refresh_profiles()
            self._send_json(
                200,
                {
                    "ok": True,
                    "default_model": self.state.default_model,
                    "current_model": self.state.current_model,
                    "available_models": list(self.state.available_profiles),
                },
            )
            return
        self._send_json(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:
        raw_body: bytes | None = None
        length = int(self.headers.get("content-length", "0") or 0)
        if length:
            raw_body = self.rfile.read(length)

        body, profile = self._prepare_request(raw_body)
        if profile is None:
            return
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in HOP_BY_HOP_HEADERS
        }
        if body is not None:
            headers["Content-Type"] = headers.get("Content-Type", "application/json")

        request = urllib.request.Request(
            profile.target.rstrip("/") + self.path,
            data=body,
            headers=headers,
            method=self.command,
        )
        try:
            with urllib.request.urlopen(request, timeout=3600) as response:
                self._send_upstream_response(response.status, response.headers, response)
        except urllib.error.HTTPError as exc:
            self._send_upstream_response(exc.code, exc.headers, exc)
        except Exception as exc:  # pragma: no cover
            payload = json.dumps({"error": {"message": str(exc)}}).encode()
            try:
                self.send_response(502)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                self._note_client_disconnect()

    def _prepare_request(self, raw_body: bytes | None) -> tuple[bytes | None, ModelProfile | None]:
        if self.path.rstrip("/") not in {"/v1/responses", "/v1/chat/completions", "/v1/completions"}:
            self._send_json(404, {"error": {"message": "not found"}})
            return None, None

        data: dict[str, Any] | None = None
        if raw_body:
            try:
                parsed = json.loads(raw_body)
                if isinstance(parsed, dict):
                    data = parsed
            except json.JSONDecodeError:
                pass

        requested_model = None
        if isinstance(data, dict):
            requested_model = str(data.get("model") or "").strip() or None
            if self.state.debug and self.path.rstrip("/") == "/v1/responses":
                try:
                    debug_path = self.state.log_dir / "codex_local_router_request.json"
                    debug_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
                except Exception:
                    pass

        try:
            profile = self.state.ensure_model(requested_model)
        except Exception as exc:
            self._send_json(502, {"error": {"message": str(exc)}})
            return None, None

        if data is None:
            return raw_body, profile

        data["model"] = profile.alias
        if self.path.rstrip("/") == "/v1/responses":
            data = normalize_responses_request(data)
        return json.dumps(data, separators=(",", ":")).encode(), profile

    def _send_upstream_response(self, status: int, headers: Any, stream: Any) -> None:
        try:
            self.send_response(status)
            for key, value in headers.items():
                if key.lower() not in HOP_BY_HOP_HEADERS:
                    self.send_header(key, value)
            self.end_headers()
            while True:
                chunk = stream.read(8192)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            self._note_client_disconnect()

    def _note_client_disconnect(self) -> None:
        if self.state.debug:
            print(f"client disconnected mid-stream: {self.command} {self.path}", file=sys.stderr, flush=True)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        if self.state.debug:
            super().log_message(fmt, *args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18111)
    parser.add_argument(
        "--default-model",
        default=os.getenv("MARATHON_DEFAULT_MODEL")
        or os.getenv("CODEX_QWEN_DEFAULT_MODEL")
        or "qwen3.6-27b-q4-128k",
    )
    parser.add_argument("--state-dir", default=str(_repo_root() / ".marathon" / "state"))
    parser.add_argument("--log-dir", default=str(_repo_root() / "scripts" / "logs"))
    parser.add_argument("--debug", action="store_true", default=bool(os.getenv("CODEX_LLAMA_DEBUG")))
    args = parser.parse_args()

    _base_instructions()
    state = RouterState(
        default_model=args.default_model,
        state_dir=Path(args.state_dir).resolve(),
        log_dir=Path(args.log_dir).resolve(),
        debug=args.debug,
    )
    RouterHandler.state = state
    threading.Thread(target=state.ensure_model, args=(state.default_model,), daemon=True).start()
    server = ThreadingHTTPServer((args.host, args.port), RouterHandler)
    print(f"codex local router ready: http://{args.host}:{args.port}/v1", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
