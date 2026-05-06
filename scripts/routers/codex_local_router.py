#!/usr/bin/env python3
"""Local multi-model router for running Codex against llama.cpp backends.

This exposes a Codex-compatible `/v1/models` catalog with multiple local model
profiles and normalizes Codex Responses API requests before forwarding them to
the selected local backend. Only one heavyweight backend is kept warm at a time;
switching models starts the requested backend and stops the others.

It also implements a Responses websocket transport that preserves conversation
lineage with llama.cpp slot save/restore. That gives Marathon a local analogue
to the server-side `previous_response_id` behavior that upstream Codex expects.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aiohttp import ClientSession
from aiohttp import ClientTimeout
from aiohttp import WSMsgType
from aiohttp import web

from marathon_web_search import WebFetchExecutor
from marathon_web_search import WebFetchSettings
from marathon_web_search import WEB_BROWSE_TOOL_NAME
from marathon_web_search import WEB_FETCH_TOOL_NAME
from marathon_web_search import WEB_SEARCH_TOOL_NAME
from marathon_web_search import WebSearchExecutor
from marathon_web_search import WebSearchSettings
from marathon_web_search import collect_managed_calls
from marathon_web_search import externalize_for_codex
from marathon_web_search import format_results_for_model
from marathon_web_search import is_web_browse_function_call
from marathon_web_search import is_web_fetch_function_call
from marathon_web_search import is_web_search_function_call
from marathon_web_search import make_function_call_output
from marathon_web_search import parse_function_call_arguments
from marathon_web_search import request_has_web_search_tool
from marathon_web_search import synthesize_call_id
from marathon_web_search import web_browse_available
from marathon_web_search import web_browse_function_tool
from marathon_web_search import web_fetch_function_tool
from marathon_web_search import web_search_function_tool


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

DEFAULT_USAGE = {
    "input_tokens": 0,
    "input_tokens_details": None,
    "output_tokens": 0,
    "output_tokens_details": None,
    "total_tokens": 0,
}

MANAGED_WEB_TOOL_NAMES = {WEB_SEARCH_TOOL_NAME, WEB_FETCH_TOOL_NAME, WEB_BROWSE_TOOL_NAME}

BACKEND_UNSUPPORTED_CONTEXT_ITEM_TYPES = {
    # Codex-native UI/context markers. Official OpenAI accepts these in
    # replayed Responses input, but llama.cpp does not know their item types.
    "web_search_call",
    "tool_search_call",
    "tool_search_output",
    "reasoning",
    "compaction",
    "compaction_summary",
    "image_generation_call",
}

WS_KEEPALIVE_INTERVAL_SECONDS = float(os.getenv("MARATHON_WS_KEEPALIVE_INTERVAL_SECONDS", "15"))


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


@dataclass
class ResponseSnapshot:
    response_id: str
    profile_slug: str
    conversation_items: list[dict[str, Any]]
    snapshot_filename: str
    instructions_hash: str
    tools_hash: str
    created_at: float


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


def _target_override(env_var: str, default: str) -> str:
    return os.getenv(env_var) or default


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
            target=_target_override("MARATHON_QWEN36_27B_128K_TARGET", "http://127.0.0.1:18091"),
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
            target=_target_override("MARATHON_QWEN36_27B_FAST_TARGET", "http://127.0.0.1:18090"),
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
            target=_target_override("MARATHON_QWEN36_35B_A3B_TARGET", "http://127.0.0.1:18092"),
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


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _item_roles(items: list[Any]) -> list[str]:
    roles: list[str] = []
    for item in items:
        if isinstance(item, dict):
            role = item.get("role")
            roles.append(str(role) if role is not None else "?")
        else:
            roles.append("?")
    return roles


def _common_prefix_items(previous: list[Any], current: list[Any]) -> int:
    count = 0
    for prev_item, curr_item in zip(previous, current):
        if prev_item != curr_item:
            break
        count += 1
    return count


def _input_relation(previous: list[Any] | None, current: list[Any]) -> dict[str, Any]:
    if previous is None:
        return {
            "relation": "none",
            "common_prefix_items": 0,
            "previous_input_items": None,
            "current_input_items": len(current),
        }

    prefix = _common_prefix_items(previous, current)
    if current == previous:
        relation = "equal"
    elif prefix == len(previous) and len(current) > len(previous):
        relation = "extends_prev"
    elif prefix == len(current) and len(previous) > len(current):
        relation = "rewinds_prev"
    elif prefix > 0:
        relation = "branches_after_prefix"
    else:
        relation = "diverges"

    return {
        "relation": relation,
        "common_prefix_items": prefix,
        "previous_input_items": len(previous),
        "current_input_items": len(current),
    }


def _strip_managed_web_tools(tools: Any) -> list[Any]:
    if not isinstance(tools, list):
        return []
    return [
        tool
        for tool in tools
        if not (
            isinstance(tool, dict)
            and tool.get("type") == "function"
            and tool.get("name") in MANAGED_WEB_TOOL_NAMES
        )
    ]


def normalize_responses_request(data: dict[str, Any]) -> dict[str, Any]:
    tools = data.get("tools")
    web_search_requested = request_has_web_search_tool(tools) if isinstance(tools, list) else False
    if isinstance(tools, list):
        data["tools"] = [tool for tool in tools if tool.get("type") == "function"]
    if web_search_requested:
        existing = data.get("tools")
        normalized_tools = existing if isinstance(existing, list) else []
        names = {
            tool.get("name")
            for tool in normalized_tools
            if isinstance(tool, dict)
        }
        if WEB_SEARCH_TOOL_NAME not in names:
            normalized_tools.append(web_search_function_tool())
        if WEB_FETCH_TOOL_NAME not in names:
            normalized_tools.append(web_fetch_function_tool())
        if web_browse_available() and WEB_BROWSE_TOOL_NAME not in names:
            normalized_tools.append(web_browse_function_tool())
        data["tools"] = normalized_tools
    data["_marathon_web_search_enabled"] = web_search_requested

    input_items = data.get("input")
    if not isinstance(input_items, list):
        return data

    lifted_messages: list[str] = []
    normalized_input: list[Any] = []
    input_changed = False
    for item in input_items:
        if not isinstance(item, dict):
            normalized_input.append(item)
            continue

        if item.get("type") in BACKEND_UNSUPPORTED_CONTEXT_ITEM_TYPES:
            input_changed = True
            continue

        role = item.get("role")
        if item.get("type") == "message" and role in {"developer", "system"}:
            text = _content_text(item.get("content"))
            if text:
                lifted_messages.append(text)
            input_changed = True
            continue

        normalized_input.append(item)

    if lifted_messages:
        existing = data.get("instructions")
        instructions = existing if isinstance(existing, str) else ""
        data["instructions"] = "\n\n".join(part for part in [instructions, *lifted_messages] if part)
    if input_changed:
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
        self.backend_lock = asyncio.Lock()
        self.lineage_lock = asyncio.Lock()
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
        self.slot_id = int(os.getenv("MARATHON_ROUTER_SLOT_ID") or "0")
        self.experimental_delta_only = bool(os.getenv("MARATHON_WS_EXPERIMENTAL_DELTA_ONLY"))
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.trace_log_path = self.log_dir / "codex_local_router_trace.jsonl"
        self.request_log_path = self.log_dir / "codex_local_router_request.json"
        self._trace_seq = 0
        self._response_id_seq = 0
        self._last_trace_by_model: dict[str, dict[str, Any]] = {}
        self.lineage: dict[str, ResponseSnapshot] = {}
        self.last_response_by_model: dict[str, str] = {}
        self.http_client: ClientSession | None = None
        self.web_search_settings = WebSearchSettings.from_env()
        self.web_search: WebSearchExecutor | None = None
        self.web_fetch_settings = WebFetchSettings.from_env()
        self.web_fetch: WebFetchExecutor | None = None

    def _refresh_profiles(self) -> dict[str, ModelProfile]:
        profiles = _available_profiles()
        if profiles:
            self.available_profiles = profiles
        return self.available_profiles

    async def open(self) -> None:
        self.http_client = ClientSession(timeout=ClientTimeout(total=3600))
        self.web_search = WebSearchExecutor(self.web_search_settings)
        self.web_fetch = WebFetchExecutor(self.web_fetch_settings)

    async def close(self) -> None:
        if self.web_search is not None:
            await self.web_search.close()
            self.web_search = None
        if self.web_fetch is not None:
            await self.web_fetch.close()
            self.web_fetch = None
        if self.http_client is not None:
            await self.http_client.close()
            self.http_client = None

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
                    "supports_search_tool": True,
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

    def trace_request(
        self,
        *,
        requested_model: str | None,
        profile: ModelProfile,
        raw_request: dict[str, Any],
        normalized_request: dict[str, Any],
        path: str,
        method: str,
        lineage: dict[str, Any] | None = None,
    ) -> None:
        if not self.debug:
            return

        raw_input = raw_request.get("input")
        normalized_input = normalized_request.get("input")
        raw_input_items = raw_input if isinstance(raw_input, list) else []
        normalized_input_items = normalized_input if isinstance(normalized_input, list) else []
        raw_tools = raw_request.get("tools")
        normalized_tools = normalized_request.get("tools")
        raw_tools_list = raw_tools if isinstance(raw_tools, list) else []
        normalized_tools_list = normalized_tools if isinstance(normalized_tools, list) else []
        lifted_roles = sum(
            1
            for item in raw_input_items
            if isinstance(item, dict) and item.get("role") in {"developer", "system"}
        )

        instructions = normalized_request.get("instructions")
        instructions_text = instructions if isinstance(instructions, str) else ""
        tools_json = _stable_json(normalized_tools_list)
        input_json = _stable_json(normalized_input_items)
        normalized_json = _stable_json(normalized_request)

        with self.lock:
            self._trace_seq += 1
            previous = self._last_trace_by_model.get(profile.slug)
            instructions_hash = _sha256_text(instructions_text)
            tools_hash = _sha256_text(tools_json)
            input_hash = _sha256_text(input_json)
            normalized_hash = _sha256_text(normalized_json)
            input_relation = _input_relation(
                previous.get("input_items") if previous is not None else None,
                normalized_input_items,
            )
            entry = {
                "trace_id": self._trace_seq,
                "timestamp": time.time(),
                "method": method,
                "path": path,
                "requested_model": requested_model,
                "profile_slug": profile.slug,
                "profile_alias": profile.alias,
                "prompt_cache_key": normalized_request.get("prompt_cache_key"),
                "previous_response_id": normalized_request.get("previous_response_id"),
                "raw": {
                    "body_bytes": len(_stable_json(raw_request).encode("utf-8")),
                    "input_items": len(raw_input_items),
                    "input_roles": _item_roles(raw_input_items),
                    "tool_count": len(raw_tools_list),
                    "developer_or_system_items": lifted_roles,
                },
                "normalized": {
                    "body_bytes": len(normalized_json.encode("utf-8")),
                    "body_hash": normalized_hash,
                    "input_items": len(normalized_input_items),
                    "input_roles": _item_roles(normalized_input_items),
                    "input_bytes": len(input_json.encode("utf-8")),
                    "input_hash": input_hash,
                    "instructions_bytes": len(instructions_text.encode("utf-8")),
                    "instructions_hash": instructions_hash,
                    "tool_count": len(normalized_tools_list),
                    "tools_bytes": len(tools_json.encode("utf-8")),
                    "tools_hash": tools_hash,
                },
                "diff_from_previous": {
                    **input_relation,
                    "same_instructions": previous is not None
                    and previous.get("instructions_hash") == instructions_hash,
                    "same_tools": previous is not None and previous.get("tools_hash") == tools_hash,
                    "same_scaffold": previous is not None
                    and previous.get("instructions_hash") == instructions_hash
                    and previous.get("tools_hash") == tools_hash,
                    "same_normalized_body": previous is not None
                    and previous.get("body_hash") == normalized_hash,
                },
            }
            if lineage is not None:
                entry["lineage"] = lineage
            self._last_trace_by_model[profile.slug] = {
                "input_items": copy.deepcopy(normalized_input_items),
                "instructions_hash": instructions_hash,
                "tools_hash": tools_hash,
                "body_hash": normalized_hash,
            }
            try:
                with self.trace_log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(entry, sort_keys=True) + "\n")
            except Exception:
                pass

    def resolve_model(self, requested_model: str | None) -> ModelProfile:
        self._refresh_profiles()
        model_key = (requested_model or self.default_model).strip()
        if model_key in self.available_profiles:
            return self.available_profiles[model_key]
        raise ValueError(f"unknown or unavailable local model '{model_key}'")

    async def ensure_model_async(self, requested_model: str | None) -> ModelProfile:
        return await asyncio.to_thread(self.ensure_model, requested_model)

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
            ["bash", "-lc", f"ss -ltnp '( sport = :{port} )' 2>/dev/null | sed -n 's/.*pid=\\\\([0-9]\\\\+\\\\).*/\\\\1/p' | head -n1"],
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

    async def _request_json(self, profile: ModelProfile, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if self.http_client is None:
            raise RuntimeError("router HTTP client session is not open")
        url = f"{profile.target.rstrip('/')}{path}"
        async with self.http_client.request(method, url, json=payload) as response:
            text = await response.text()
            if response.status >= 400:
                raise RuntimeError(f"backend {method} {path} failed: {response.status} {text}")
            try:
                return json.loads(text) if text else {}
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"backend {method} {path} returned invalid JSON: {exc}") from exc

    async def _slot_action(self, profile: ModelProfile, action: str, filename: str | None = None) -> dict[str, Any]:
        payload = {"filename": filename} if filename is not None else None
        return await self._request_json(profile, "POST", f"/slots/{self.slot_id}?action={action}", payload)

    async def erase_slot(self, profile: ModelProfile) -> dict[str, Any]:
        return await self._slot_action(profile, "erase")

    async def save_slot(self, profile: ModelProfile, filename: str) -> dict[str, Any]:
        return await self._slot_action(profile, "save", filename)

    async def restore_slot(self, profile: ModelProfile, filename: str) -> dict[str, Any]:
        return await self._slot_action(profile, "restore", filename)

    async def backend_health(self, profile: ModelProfile | None = None) -> dict[str, Any]:
        target_profile = profile or self.resolve_model(self.current_model or self.default_model)
        return await self._request_json(target_profile, "GET", "/health")

    def mint_response_id(self, kind: str = "resp") -> str:
        """Generate a fresh, monotonic response id without touching the trace counter.

        Used so the WS handler can send ``response.created`` immediately on
        receiving ``response.create``, before the (potentially long) backend
        call begins.
        """

        with self.lock:
            self._response_id_seq += 1
            seq = self._response_id_seq
        return f"{kind}_{int(time.time() * 1000)}_{seq}"

    def sanitize_output_item(self, item: dict[str, Any]) -> dict[str, Any]:
        sanitized = copy.deepcopy(item)
        sanitized.pop("status", None)
        return sanitized

    def usage_payload(self, usage: Any) -> dict[str, Any]:
        if not isinstance(usage, dict):
            return copy.deepcopy(DEFAULT_USAGE)
        merged = copy.deepcopy(DEFAULT_USAGE)
        merged.update(usage)
        return merged

    def _log_tool_error(self, tool: str, key: str, exc: Exception) -> None:
        try:
            with self.log_dir.joinpath("codex_local_router.log").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(f"{tool} error for {key!r}: {exc}\n")
        except Exception:
            pass

    async def _execute_web_search_call(
        self, item: dict[str, Any], fallback_index: int
    ) -> dict[str, Any]:
        call_id = synthesize_call_id(item, fallback_index)
        args = parse_function_call_arguments(item.get("arguments"))
        query = str(args.get("query") or "").strip()
        max_results_raw = args.get("max_results")
        max_results: int | None
        if isinstance(max_results_raw, int):
            max_results = max_results_raw
        elif isinstance(max_results_raw, str) and max_results_raw.isdigit():
            max_results = int(max_results_raw)
        else:
            max_results = None

        if self.web_search is None:
            return make_function_call_output(
                call_id, "Web search is unavailable: router has no executor configured."
            )
        if not query:
            return make_function_call_output(
                call_id, "Web search error: missing 'query' argument."
            )

        try:
            results = await self.web_search.search(query, max_results=max_results)
        except Exception as exc:
            self._log_tool_error("web_search", query, exc)
            return make_function_call_output(
                call_id,
                f"Web search failed: {exc}\nThe SearXNG instance at "
                f"{self.web_search_settings.base_url} did not respond. Tell the user "
                "the search backend is unreachable and answer from your own knowledge "
                "if possible.",
            )

        return make_function_call_output(call_id, format_results_for_model(query, results))

    async def _execute_web_fetch_call(
        self, item: dict[str, Any], fallback_index: int
    ) -> dict[str, Any]:
        call_id = synthesize_call_id(item, fallback_index)
        args = parse_function_call_arguments(item.get("arguments"))
        url = str(args.get("url") or "").strip()
        max_chars_raw = args.get("max_chars")
        max_chars: int | None
        if isinstance(max_chars_raw, int):
            max_chars = max_chars_raw
        elif isinstance(max_chars_raw, str) and max_chars_raw.isdigit():
            max_chars = int(max_chars_raw)
        else:
            max_chars = None

        if self.web_fetch is None:
            return make_function_call_output(
                call_id, "Web fetch is unavailable: router has no executor configured."
            )
        if not url:
            return make_function_call_output(
                call_id, "Web fetch error: missing 'url' argument."
            )

        try:
            content = await self.web_fetch.fetch(url, max_chars=max_chars)
        except Exception as exc:
            self._log_tool_error("web_fetch", url, exc)
            return make_function_call_output(
                call_id,
                f"web_fetch failed for {url}: {exc}\nIf this URL depends on "
                "browser-rendered JavaScript, tell the user static fetch could "
                "not extract it. "
                "Otherwise prefer a different result from the previous search.",
            )

        return make_function_call_output(call_id, content)

    async def _execute_web_browse_call(
        self, item: dict[str, Any], fallback_index: int
    ) -> dict[str, Any]:
        call_id = synthesize_call_id(item, fallback_index)
        args = parse_function_call_arguments(item.get("arguments"))
        url = str(args.get("url") or "").strip()
        max_chars_raw = args.get("max_chars")
        max_chars: int | None
        if isinstance(max_chars_raw, int):
            max_chars = max_chars_raw
        elif isinstance(max_chars_raw, str) and max_chars_raw.isdigit():
            max_chars = int(max_chars_raw)
        else:
            max_chars = None

        if self.web_fetch is None:
            return make_function_call_output(
                call_id, "Web browse is unavailable: router has no executor configured."
            )
        if not url:
            return make_function_call_output(
                call_id, "Web browse error: missing 'url' argument."
            )

        try:
            content = await self.web_fetch.browse(url, max_chars=max_chars)
        except Exception as exc:
            self._log_tool_error("web_browse", url, exc)
            return make_function_call_output(
                call_id,
                f"web_browse failed for {url}: {exc}\nUse web_fetch for static "
                "pages, or tell the user the browser-rendered extractor is unavailable.",
            )

        return make_function_call_output(call_id, content)

    async def _execute_managed_call(
        self, item: dict[str, Any], fallback_index: int
    ) -> dict[str, Any]:
        if is_web_browse_function_call(item):
            return await self._execute_web_browse_call(item, fallback_index)
        if is_web_fetch_function_call(item):
            return await self._execute_web_fetch_call(item, fallback_index)
        return await self._execute_web_search_call(item, fallback_index)

    async def _run_responses_loop(
        self,
        *,
        profile: ModelProfile,
        forward_request: dict[str, Any],
        web_search_enabled: bool,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
        """Run the llama.cpp /v1/responses tool-call loop until the model stops calling web_search.

        Returns the final backend response, the cumulative output items (with the
        real ``function_call`` and ``function_call_output`` items kept intact for
        lineage), and the number of web_search iterations executed.
        """

        request = forward_request
        max_iters = max(1, self.web_search_settings.max_iterations)
        cumulative_items: list[dict[str, Any]] = []
        last_response: dict[str, Any] = {}
        usage_acc = copy.deepcopy(DEFAULT_USAGE)
        iterations = 0

        for iteration in range(max_iters + 1):
            response = await self._request_json(profile, "POST", "/v1/responses", request)
            last_response = response
            iter_items: list[dict[str, Any]] = []
            for item in response.get("output", []):
                if isinstance(item, dict):
                    iter_items.append(self.sanitize_output_item(item))

            iter_usage = response.get("usage")
            if isinstance(iter_usage, dict):
                for key, value in iter_usage.items():
                    if isinstance(value, (int, float)) and isinstance(usage_acc.get(key), (int, float)):
                        usage_acc[key] = (usage_acc.get(key) or 0) + value
                    elif key not in usage_acc or usage_acc.get(key) is None:
                        usage_acc[key] = value

            cumulative_items.extend(iter_items)

            if not web_search_enabled:
                break

            pending_calls = collect_managed_calls(iter_items)
            if not pending_calls:
                break

            if iteration >= max_iters:
                # Safety cap: drop the managed tools so the model must finalize.
                tool_outputs = [
                    make_function_call_output(
                        synthesize_call_id(call, idx),
                        "Tool-call iteration cap reached; please answer from "
                        "the results you already have.",
                    )
                    for idx, call in enumerate(pending_calls)
                ]
                cumulative_items.extend(tool_outputs)
                request = copy.deepcopy(request)
                request["input"] = list(request.get("input") or []) + copy.deepcopy(iter_items) + tool_outputs
                request["tools"] = _strip_managed_web_tools(request.get("tools"))
                final = await self._request_json(profile, "POST", "/v1/responses", request)
                last_response = final
                final_items: list[dict[str, Any]] = []
                for item in final.get("output", []):
                    if isinstance(item, dict):
                        final_items.append(self.sanitize_output_item(item))
                final_usage = final.get("usage")
                if isinstance(final_usage, dict):
                    for key, value in final_usage.items():
                        if isinstance(value, (int, float)) and isinstance(usage_acc.get(key), (int, float)):
                            usage_acc[key] = (usage_acc.get(key) or 0) + value
                        elif key not in usage_acc or usage_acc.get(key) is None:
                            usage_acc[key] = value
                cumulative_items.extend(final_items)
                iterations = max_iters
                break

            tool_outputs = []
            for idx, call in enumerate(pending_calls):
                tool_outputs.append(await self._execute_managed_call(call, idx))
            cumulative_items.extend(tool_outputs)

            request = copy.deepcopy(request)
            request["input"] = list(request.get("input") or []) + copy.deepcopy(iter_items) + tool_outputs
            iterations += 1

        last_response = copy.deepcopy(last_response)
        last_response["usage"] = usage_acc
        return last_response, cumulative_items, iterations

    async def process_websocket_create(
        self,
        payload: dict[str, Any],
        *,
        preset_response_id: str | None = None,
    ) -> dict[str, Any]:
        raw_snapshot = copy.deepcopy(payload)
        request = copy.deepcopy(payload)
        request.pop("type", None)

        requested_model = str(request.get("model") or "").strip() or None
        generate = request.get("generate")
        previous_response_id = request.get("previous_response_id")
        if previous_response_id is not None and not isinstance(previous_response_id, str):
            raise RuntimeError("previous_response_id must be a string when provided")

        parent_snapshot: ResponseSnapshot | None = None
        if previous_response_id:
            async with self.lineage_lock:
                parent_snapshot = self.lineage.get(previous_response_id)
            if parent_snapshot is None:
                raise RuntimeError(f"unknown previous_response_id: {previous_response_id}")
            if requested_model and requested_model != parent_snapshot.profile_slug:
                raise RuntimeError(
                    f"previous_response_id {previous_response_id} belongs to model "
                    f"{parent_snapshot.profile_slug}, not {requested_model}"
                )
            profile = await self.ensure_model_async(parent_snapshot.profile_slug)
        else:
            profile = await self.ensure_model_async(requested_model)

        request["model"] = profile.alias
        request = normalize_responses_request(request)
        request["model"] = profile.alias

        delta_input = request.get("input")
        if not isinstance(delta_input, list):
            raise RuntimeError("response.create requires list input")

        tools = request.get("tools")
        if not isinstance(tools, list):
            tools = []
            request["tools"] = tools
        instructions = request.get("instructions")
        instructions_text = instructions if isinstance(instructions, str) else ""
        instructions_hash = _sha256_text(instructions_text)
        tools_hash = _sha256_text(_stable_json(tools))

        relation = "root"
        full_input: list[dict[str, Any]]
        if parent_snapshot is None:
            full_input = copy.deepcopy(delta_input)
        else:
            full_input = copy.deepcopy(parent_snapshot.conversation_items) + copy.deepcopy(delta_input)
            relation = "continue" if self.last_response_by_model.get(profile.slug) == previous_response_id else "branch"
        scaffold_matches = (
            parent_snapshot is not None
            and parent_snapshot.instructions_hash == instructions_hash
            and parent_snapshot.tools_hash == tools_hash
        )

        delta_only_restore = (
            self.experimental_delta_only
            and parent_snapshot is not None
            and scaffold_matches
            and generate is not False
        )

        forward_request = copy.deepcopy(request)
        forward_request.pop("previous_response_id", None)
        forward_request["input"] = copy.deepcopy(delta_input if delta_only_restore else full_input)
        if delta_only_restore:
            forward_request.pop("instructions", None)
            forward_request["tools"] = []
        forward_request["id_slot"] = self.slot_id
        forward_request["cache_prompt"] = True
        forward_request["stream"] = False

        self.trace_request(
            requested_model=requested_model,
            profile=profile,
            raw_request=raw_snapshot,
            normalized_request=request,
            path="/v1/responses",
            method="WS",
            lineage={
                "mode": relation,
                "delta_input_items": len(delta_input),
                "full_input_items": len(full_input),
                "known_previous_response_id": previous_response_id,
                "resolved_profile_slug": profile.slug,
                "scaffold_matches": scaffold_matches,
                "delta_only_restore": delta_only_restore,
            },
        )

        if generate is False:
            response_id = preset_response_id or self.mint_response_id("warm")
            async with self.lineage_lock:
                self.lineage[response_id] = ResponseSnapshot(
                    response_id=response_id,
                    profile_slug=profile.slug,
                    conversation_items=copy.deepcopy(full_input),
                    snapshot_filename="",
                    instructions_hash=instructions_hash,
                    tools_hash=tools_hash,
                    created_at=time.time(),
                )
                self.last_response_by_model[profile.slug] = response_id

            if self.debug:
                with self.lock:
                    self._trace_seq += 1
                    trace_entry = {
                        "trace_id": self._trace_seq,
                        "timestamp": time.time(),
                        "method": "WS",
                        "path": "/v1/responses",
                        "profile_slug": profile.slug,
                        "profile_alias": profile.alias,
                        "requested_model": requested_model,
                        "previous_response_id": previous_response_id,
                        "response_id": response_id,
                        "relation": "warmup",
                        "backend_ms": 0.0,
                        "backend_timings": None,
                        "restore_result": None,
                        "lineage": {
                            "mode": "warmup",
                            "delta_input_items": len(delta_input),
                            "full_input_items": len(full_input),
                            "scaffold_matches": scaffold_matches,
                            "delta_only_restore": delta_only_restore,
                        },
                        "slot": {
                            "slot_id": self.slot_id,
                            "prepare_ms": 0.0,
                            "save_ms": 0.0,
                            "erase_result": None,
                            "restore_error": None,
                            "restore_result": None,
                            "save_error": None,
                            "save_result": None,
                            "snapshot_filename": "",
                        },
                    }
                    try:
                        with self.trace_log_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(trace_entry, sort_keys=True) + "\n")
                    except Exception:
                        pass

            return {
                "response_id": response_id,
                "usage": copy.deepcopy(DEFAULT_USAGE),
                "output_items": [],
                "backend_response": {},
            }

        web_search_enabled = bool(
            forward_request.pop("_marathon_web_search_enabled", False)
        ) and self.web_search is not None

        async with self.backend_lock:
            restore_result: dict[str, Any] | None = None
            erase_result: dict[str, Any] | None = None
            restore_error: str | None = None
            slot_prepare_mode = "erase-root"
            slot_prepare_start = time.perf_counter()
            if parent_snapshot is None:
                erase_result = await self.erase_slot(profile)
            elif not scaffold_matches:
                slot_prepare_mode = "erase-scaffold-mismatch"
                erase_result = await self.erase_slot(profile)
            elif not parent_snapshot.snapshot_filename:
                slot_prepare_mode = "erase-parent-no-snapshot"
                erase_result = await self.erase_slot(profile)
            else:
                slot_prepare_mode = "restore-parent"
                try:
                    restore_result = await self.restore_slot(profile, parent_snapshot.snapshot_filename)
                except Exception as exc:
                    restore_error = str(exc)
                    slot_prepare_mode = "erase-restore-error"
                    erase_result = await self.erase_slot(profile)
            slot_prepare_ms = (time.perf_counter() - slot_prepare_start) * 1000.0

            backend_start = time.perf_counter()
            backend_response, all_output_items, web_search_iterations = await self._run_responses_loop(
                profile=profile,
                forward_request=forward_request,
                web_search_enabled=web_search_enabled,
            )
            backend_ms = (time.perf_counter() - backend_start) * 1000.0

            response_id = preset_response_id or str(
                backend_response.get("id") or self.mint_response_id("resp")
            )
            snapshot_filename = f"{profile.slug}__{response_id}.bin"

            save_start = time.perf_counter()
            save_result: dict[str, Any] | None = None
            save_error: str | None = None
            try:
                save_result = await self.save_slot(profile, snapshot_filename)
            except Exception as exc:
                save_error = str(exc)
            slot_save_ms = (time.perf_counter() - save_start) * 1000.0

        output_items = all_output_items

        usage_payload = self.usage_payload(backend_response.get("usage"))

        conversation_items = full_input + copy.deepcopy(output_items)
        async with self.lineage_lock:
            self.lineage[response_id] = ResponseSnapshot(
                response_id=response_id,
                profile_slug=profile.slug,
                conversation_items=conversation_items,
                snapshot_filename=snapshot_filename,
                instructions_hash=instructions_hash,
                tools_hash=tools_hash,
                created_at=time.time(),
            )
            self.last_response_by_model[profile.slug] = response_id

        if self.debug:
            with self.lock:
                self._trace_seq += 1
                trace_entry = {
                    "trace_id": self._trace_seq,
                    "timestamp": time.time(),
                    "method": "WS",
                    "path": "/v1/responses",
                    "profile_slug": profile.slug,
                    "profile_alias": profile.alias,
                    "requested_model": requested_model,
                    "previous_response_id": previous_response_id,
                    "response_id": response_id,
                    "relation": relation,
                    "backend_ms": backend_ms,
                    "backend_timings": backend_response.get("timings"),
                    "restore_result": restore_result,
                    "lineage": {
                        "mode": relation,
                        "delta_input_items": len(delta_input),
                        "full_input_items": len(full_input),
                        "scaffold_matches": scaffold_matches,
                        "delta_only_restore": delta_only_restore,
                    },
                    "slot": {
                        "slot_id": self.slot_id,
                        "prepare_mode": slot_prepare_mode,
                        "prepare_ms": slot_prepare_ms,
                        "save_ms": slot_save_ms,
                        "erase_result": erase_result,
                        "restore_result": restore_result,
                        "restore_error": restore_error,
                        "save_result": save_result,
                        "save_error": save_error,
                        "snapshot_filename": snapshot_filename,
                    },
                    "backend": {
                        "usage": usage_payload,
                        "timings": backend_response.get("timings"),
                        "latency_ms": backend_ms,
                    },
                }
                try:
                    with self.trace_log_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(trace_entry, sort_keys=True) + "\n")
                except Exception:
                    pass

        codex_output_items = (
            externalize_for_codex(output_items) if web_search_enabled else output_items
        )

        return {
            "response_id": response_id,
            "usage": usage_payload,
            "output_items": codex_output_items,
            "backend_response": backend_response,
        }


async def handle_models(request: web.Request) -> web.Response:
    state: RouterState = request.app["state"]
    return web.json_response(state.model_catalog())


async def handle_health(request: web.Request) -> web.Response:
    state: RouterState = request.app["state"]
    state._refresh_profiles()
    backend_status: dict[str, Any] | None = None
    if state.current_model:
        try:
            backend_status = await state.backend_health(state.resolve_model(state.current_model))
        except Exception as exc:
            backend_status = {"status": "error", "message": str(exc)}
    return web.json_response(
        {
            "ok": True,
            "default_model": state.default_model,
            "current_model": state.current_model,
            "available_models": list(state.available_profiles),
            "slot_id": state.slot_id,
            "known_lineage": len(state.lineage),
            "backend_health": backend_status,
        }
    )


async def handle_http_proxy(request: web.Request) -> web.StreamResponse:
    state: RouterState = request.app["state"]
    raw_body = await request.read()
    path = request.path.rstrip("/")
    if path not in {"/v1/responses", "/v1/chat/completions", "/v1/completions"}:
        return web.json_response({"error": {"message": "not found"}}, status=404)

    data: dict[str, Any] | None = None
    if raw_body:
        try:
            parsed = json.loads(raw_body)
            if isinstance(parsed, dict):
                data = parsed
        except json.JSONDecodeError:
            data = None

    requested_model = None
    if isinstance(data, dict):
        requested_model = str(data.get("model") or "").strip() or None
        if state.debug and path == "/v1/responses":
            try:
                state.request_log_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            except Exception:
                pass

    try:
        profile = await state.ensure_model_async(requested_model)
    except Exception as exc:
        return web.json_response({"error": {"message": str(exc)}}, status=502)

    body = raw_body
    if data is not None:
        raw_snapshot = copy.deepcopy(data)
        data["model"] = profile.alias
        if path == "/v1/responses":
            data = normalize_responses_request(data)
            if data.pop("_marathon_web_search_enabled", False):
                # The managed web-tool loop is only implemented on the
                # WebSocket Responses path. Keep HTTP/SSE fallback from
                # leaking router-private fields or unmanaged web functions to
                # llama.cpp.
                data["tools"] = _strip_managed_web_tools(data.get("tools"))
            state.trace_request(
                requested_model=requested_model,
                profile=profile,
                raw_request=raw_snapshot,
                normalized_request=data,
                path=path,
                method=request.method,
            )
        body = json.dumps(data, separators=(",", ":")).encode()

    if state.http_client is None:
        return web.json_response({"error": {"message": "router HTTP client session is not open"}}, status=500)

    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS
    }
    if body:
        headers["Content-Type"] = headers.get("Content-Type", "application/json")

    upstream_url = profile.target.rstrip("/") + path
    try:
        async with state.http_client.request(
            request.method,
            upstream_url,
            data=body if body else None,
            headers=headers,
            allow_redirects=False,
        ) as upstream:
            response = web.StreamResponse(status=upstream.status)
            for key, value in upstream.headers.items():
                if key.lower() not in HOP_BY_HOP_HEADERS:
                    response.headers[key] = value
            await response.prepare(request)
            async for chunk in upstream.content.iter_chunked(8192):
                await response.write(chunk)
            await response.write_eof()
            return response
    except Exception as exc:
        return web.json_response({"error": {"message": str(exc)}}, status=502)


async def handle_ws_responses(request: web.Request) -> web.StreamResponse:
    state: RouterState = request.app["state"]
    # Do not use aiohttp's server heartbeat here. During long local prefills the
    # handler is awaiting llama.cpp and not reading client pong frames, so aiohttp
    # can falsely treat a healthy client as dead after heartbeat + pong timeout.
    ws = web.WebSocketResponse(heartbeat=None, max_msg_size=32 * 1024 * 1024)
    await ws.prepare(request)

    async def safe_send(message: dict[str, Any]) -> bool:
        if ws.closed:
            return False
        try:
            await ws.send_json(message)
            return True
        except (ConnectionResetError, ConnectionError):
            return False

    async for msg in ws:
        if msg.type == WSMsgType.TEXT:
            try:
                payload = json.loads(msg.data)
            except json.JSONDecodeError:
                await safe_send({"type": "error", "error": {"message": "invalid JSON"}})
                continue

            if payload.get("type") != "response.create":
                await safe_send(
                    {
                        "type": "error",
                        "error": {"message": f"unsupported websocket message type: {payload.get('type')}"},
                    }
                )
                continue

            # Pre-mint the response id and acknowledge IMMEDIATELY so Codex's
            # WS client doesn't time out waiting through a long generation.
            # The same id flows into process_websocket_create so backend
            # lineage and the events we send Codex stay consistent.
            preset_id = state.mint_response_id(
                "warm" if payload.get("generate") is False else "resp"
            )
            if not await safe_send(
                {"type": "response.created", "response": {"id": preset_id}}
            ):
                # Client dropped before we could ack; skip the heavy work.
                continue

            result_task = asyncio.create_task(
                state.process_websocket_create(payload, preset_response_id=preset_id)
            )
            try:
                client_connected = True
                while True:
                    done, _ = await asyncio.wait(
                        {result_task},
                        timeout=WS_KEEPALIVE_INTERVAL_SECONDS
                        if WS_KEEPALIVE_INTERVAL_SECONDS > 0
                        else None,
                    )
                    if done:
                        result = result_task.result()
                        break
                    if not await safe_send(
                        {
                            "type": "response.in_progress",
                            "response": {"id": preset_id, "status": "in_progress"},
                        }
                    ):
                        result_task.cancel()
                        try:
                            await result_task
                        except asyncio.CancelledError:
                            pass
                        client_connected = False
                        break
                if not client_connected:
                    continue
            except Exception as exc:
                await safe_send(
                    {
                        "type": "response.failed",
                        "response": {
                            "id": preset_id,
                            "status": "failed",
                            "error": {"message": str(exc)},
                        },
                    }
                )
                continue

            for item in result["output_items"]:
                if not await safe_send(
                    {"type": "response.output_item.done", "item": item}
                ):
                    break
            await safe_send(
                {
                    "type": "response.completed",
                    "response": {
                        "id": result["response_id"],
                        "usage": result["usage"],
                    },
                }
            )
        elif msg.type == WSMsgType.ERROR:
            break

    return ws


async def on_startup(app: web.Application) -> None:
    state: RouterState = app["state"]
    await state.open()
    threading.Thread(target=state.ensure_model, args=(state.default_model,), daemon=True).start()


async def on_cleanup(app: web.Application) -> None:
    await app["state"].close()


def build_app(state: RouterState) -> web.Application:
    app = web.Application()
    app["state"] = state
    app.router.add_get("/health", handle_health)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_post("/v1/responses", handle_http_proxy)
    app.router.add_post("/v1/chat/completions", handle_http_proxy)
    app.router.add_post("/v1/completions", handle_http_proxy)
    app.router.add_get("/v1/responses", handle_ws_responses)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


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
    app = build_app(state)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
