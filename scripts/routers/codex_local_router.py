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
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from collections import Counter
from collections import OrderedDict
from collections.abc import Awaitable
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aiohttp import ClientSession
from aiohttp import ClientTimeout
from aiohttp import WSMsgType
from aiohttp import web

from marathon_app.telemetry import EventWriter

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
APPLY_PATCH_TOOL_NAME = "apply_patch"

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
WS_SEND_TIMEOUT_SECONDS = float(os.getenv("MARATHON_WS_SEND_TIMEOUT_SECONDS", "5"))
DEFAULT_SLOT_SNAPSHOT_MAX_COUNT = 16
DEFAULT_SLOT_SNAPSHOT_MAX_BYTES = 32 * 1024 * 1024 * 1024
DEFAULT_SLOT_SNAPSHOT_CLEAN_STARTUP = True
DEFAULT_SLOT_SNAPSHOTS_ENABLED = False
DEFAULT_STARTER_CACHE_ENABLED = True
DEFAULT_STARTER_CACHE_MAX_COUNT = 8
DEFAULT_STARTER_CACHE_MAX_BYTES = 16 * 1024 * 1024 * 1024
DEFAULT_TOOL_OUTPUT_MAX_CHARS = 16_384
DEFAULT_WEB_TOOL_CACHE_MAX_ENTRIES = 256
DEFAULT_WEB_TURN_PROGRESS_MAX_ENTRIES = 64
DEFAULT_WEB_TURN_PROGRESS_TTL_SECONDS = 3_600
DEFAULT_MAX_OUTPUT_TOKENS = 8_192
DEFAULT_STALLED_RESPONSE_RECOVERIES = 1
DEFAULT_TOOL_PROTOCOL_RECOVERIES = 1
DEFAULT_TOOL_ARGUMENT_MAX_CHARS = 24_576
_BACKEND_ARGUMENTS_KEY = "_marathon_backend_arguments"
_WEB_REPLAYED_COMPLETION_KEY = "_marathon_web_replayed_completion"

StreamEventSink = Callable[[dict[str, Any]], Awaitable[bool]]

CODEX_STREAM_EVENT_TYPES = {
    "response.output_item.added",
    "response.output_item.done",
    "response.output_text.delta",
    "response.custom_tool_call_input.delta",
    "response.reasoning_summary_part.added",
    "response.reasoning_summary_text.delta",
    "response.reasoning_text.delta",
}

FOLLOWUP_WORK_ITEM_TYPES = {
    "function_call",
    "custom_tool_call",
    "local_shell_call",
    "tool_search_call",
    "web_search_call",
    "image_generation_call",
}


@dataclass(frozen=True)
class SseEvent:
    name: str | None
    data: str


class ToolProtocolError(RuntimeError):
    """The backend stopped making valid, bounded tool-call progress."""


def _backend_tool_protocol_error_reason(status: int, body: str) -> str | None:
    """Classify backend HTTP errors caused by malformed generated tool calls."""

    if status < 400:
        return None
    normalized = body.casefold()
    markers = (
        "failed to parse tool call arguments as json",
        "failed to parse function arguments as json",
        "invalid tool call arguments",
    )
    if any(marker in normalized for marker in markers):
        return "the backend received malformed or truncated tool-call arguments"
    return None


def _is_client_disconnect(error: Exception) -> bool:
    """Return whether a downstream client closed an otherwise valid response."""

    return isinstance(error, (ConnectionResetError, BrokenPipeError)) or str(error) in {
        "Cannot write to closing transport",
        "Cannot write to closing transport.",
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
    supports_parallel_tool_calls: bool = False
    supports_slots: bool = True
    supervised: bool = False
    temperature: float | None = None
    default_reasoning_level: str | None = None
    supported_reasoning_levels: tuple[tuple[str, str], ...] = ()
    input_modalities: tuple[str, ...] = ("text",)

    @property
    def port(self) -> int:
        return int(self.target.rsplit(":", 1)[-1])

    def resolved_model_path(self) -> str | None:
        for candidate in self.model_paths:
            path = Path(candidate).expanduser()
            if path.exists():
                return str(path.resolve())
        return None


@dataclass
class ResponseSnapshot:
    response_id: str
    profile_slug: str
    conversation_items: list[dict[str, Any]]
    snapshot_filename: str
    instructions_text: str
    base_instructions_hash: str
    instructions_hash: str
    tools_hash: str
    prompt_cache_key: str
    created_at: float


@dataclass
class ManagedWebTurnProgress:
    """Model-visible managed-web state that survives transport reconnects."""

    request_suffix: list[dict[str, Any]]
    cumulative_items: list[dict[str, Any]]
    iterations: int
    seen_signatures: set[str]
    finalizing: bool
    completed_response: dict[str, Any] | None
    updated_at: float


def _effective_instructions_for_request(
    parent: ResponseSnapshot | None,
    current: str,
    base_instructions_hash: str,
    lifted_instruction_count: int,
) -> str:
    if (
        parent is not None
        and lifted_instruction_count == 0
        and base_instructions_hash == parent.base_instructions_hash
    ):
        return parent.instructions_text
    return current


def _can_reuse_reconnect_root(
    profile_slug: str,
    prompt_cache_key: str,
    live_slots: dict[str, str],
    live_cache_keys: dict[str, str],
) -> bool:
    return (
        bool(prompt_cache_key)
        and live_cache_keys.get(profile_slug) == prompt_cache_key
        and profile_slug in live_slots
    )


def _root_prompt_cache_mode(
    profile_slug: str,
    prompt_cache_key: str,
    live_slots: dict[str, str],
    live_cache_keys: dict[str, str],
) -> str:
    if _can_reuse_reconnect_root(
        profile_slug,
        prompt_cache_key,
        live_slots,
        live_cache_keys,
    ):
        return "reuse-live-reconnect-root"
    if profile_slug in live_slots:
        return "reuse-live-cross-conversation-root"
    return "reuse-backend-root-prefix"


def _is_warmup_root(snapshot: ResponseSnapshot | None) -> bool:
    """Return whether a non-generating Codex warmup precedes the first turn."""

    return bool(
        snapshot is not None
        and snapshot.response_id.startswith("warm_")
        and not snapshot.conversation_items
        and not snapshot.snapshot_filename
    )


def _starter_scaffold_chat_body(request: dict[str, Any]) -> dict[str, Any]:
    """Build the token-exact chat-template input before conversation messages."""

    body: dict[str, Any] = {"add_generation_prompt": False}
    if "instructions" in request:
        instructions = request.get("instructions")
        body["messages"] = [
            {
                "role": "system",
                "content": instructions if isinstance(instructions, str) else "",
            }
        ]
    else:
        body["messages"] = []

    chat_tools: list[dict[str, Any]] = []
    tools = request.get("tools")
    if isinstance(tools, list):
        for tool in tools:
            if not isinstance(tool, dict) or tool.get("type") != "function":
                continue
            function = copy.deepcopy(tool)
            function.pop("type", None)
            function.setdefault("strict", True)
            chat_tools.append({"type": "function", "function": function})
    if chat_tools:
        body["tools"] = chat_tools

    for key in (
        "chat_template_kwargs",
        "enable_thinking",
        "parallel_tool_calls",
        "reasoning_format",
        "tool_choice",
    ):
        if key in request:
            body[key] = copy.deepcopy(request[key])
    return body


def _starter_cache_fingerprint(
    profile: ModelProfile,
    backend_cache_id: str,
    scaffold_body: dict[str, Any],
) -> str:
    payload = {
        "schema": 1,
        "profile_slug": profile.slug,
        "profile_alias": profile.alias,
        "backend_cache_id": backend_cache_id,
        "scaffold": scaffold_body,
    }
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _prompt_path() -> Path:
    configured = os.getenv("MARATHON_PROMPT_FILE")
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

    ai_root = Path(os.getenv("MARATHON_AI_ROOT") or Path.home() / "AI").expanduser()
    models_dir = Path(
        os.getenv("MARATHON_MODELS_DIR")
        or os.getenv("MODELS_DIR")
        or ai_root / "models" / "gguf"
    ).expanduser()
    return tuple(str(models_dir / relative_path) for relative_path in relative_paths)


def _target_override(env_var: str, default: str) -> str:
    return os.getenv(env_var) or default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _bounded_text(value: str, limit: int) -> tuple[str, bool]:
    """Keep useful head and tail context while bounding one tool result."""

    if limit <= 0 or len(value) <= limit:
        return value, False
    marker = f"\n… [Marathon truncated {len(value) - limit:,} chars; run a narrower command] …\n"
    available = max(0, limit - len(marker))
    head = available * 3 // 4
    tail = available - head
    suffix = value[-tail:] if tail else ""
    return value[:head] + marker + suffix, True


def _bound_tool_output_item(item: dict[str, Any], limit: int) -> tuple[dict[str, Any], bool]:
    if item.get("type") not in {
        "function_call_output",
        "custom_tool_call_output",
        "local_shell_call_output",
    }:
        return item, False
    output = item.get("output")
    if not isinstance(output, str):
        return item, False
    bounded, changed = _bounded_text(output, limit)
    if not changed:
        return item, False
    result = copy.deepcopy(item)
    result["output"] = bounded
    return result, True


def _backend_image_tool_output_items(
    item: dict[str, Any],
) -> list[dict[str, Any]] | None:
    """Adapt Codex image tool output to llama.cpp's Responses input shape."""

    if item.get("type") != "function_call_output":
        return None
    output = item.get("output")
    if not isinstance(output, list):
        return None
    images = [
        copy.deepcopy(part)
        for part in output
        if isinstance(part, dict) and part.get("type") == "input_image"
    ]
    if not images:
        return None
    text_parts = [
        copy.deepcopy(part)
        for part in output
        if isinstance(part, dict) and part.get("type") == "input_text"
    ]
    tool_output = copy.deepcopy(item)
    tool_output["output"] = (
        text_parts
        if text_parts
        else "Image attached in the following user message."
    )
    return [
        tool_output,
        {
            "type": "message",
            "role": "user",
            "content": images,
        },
    ]


def _function_call_arguments_are_valid(arguments: Any) -> bool:
    """Return whether replayed function arguments form one JSON object."""

    if isinstance(arguments, dict):
        return True
    if not isinstance(arguments, str):
        return False
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, dict)


def _malformed_function_call_keys(items: list[Any]) -> set[str]:
    """Identify malformed replayed calls so their complete pair can be omitted."""

    keys: set[str] = set()
    for item in items:
        if not isinstance(item, dict) or item.get("type") != "function_call":
            continue
        if _function_call_arguments_are_valid(item.get("arguments")):
            continue
        for key in (item.get("call_id"), item.get("id")):
            if isinstance(key, str) and key:
                keys.add(key)
    return keys


def _is_malformed_function_call(
    item: dict[str, Any], malformed_call_keys: set[str]
) -> bool:
    if item.get("type") != "function_call":
        return False
    if not _function_call_arguments_are_valid(item.get("arguments")):
        return True
    return any(
        isinstance(key, str) and key in malformed_call_keys
        for key in (item.get("call_id"), item.get("id"))
    )


_UNIFIED_RANGE_HEADER = re.compile(
    r"^(?:@@\s*)?-\d+(?:,\d+)?\s+\+\d+(?:,\d+)?\s*@@\s*$"
)


def _normalize_apply_patch_dialect(value: str) -> str:
    """Translate unified-diff numeric hunks into Codex apply_patch hunks."""

    if not value:
        return value
    lines = value.splitlines()
    normalized = ["@@" if _UNIFIED_RANGE_HEADER.fullmatch(line.strip()) else line for line in lines]
    suffix = "\n" if value.endswith("\n") else ""
    return "\n".join(normalized) + suffix


def _managed_call_name(item: dict[str, Any]) -> str:
    name = item.get("name")
    return str(name) if isinstance(name, str) else "unknown"


def _managed_call_signature(item: dict[str, Any]) -> str:
    name = _managed_call_name(item)
    args = parse_function_call_arguments(item.get("arguments"))
    normalized = copy.deepcopy(args)
    if name == WEB_SEARCH_TOOL_NAME and isinstance(normalized.get("query"), str):
        normalized["query"] = " ".join(normalized["query"].split()).casefold()
    if name in {WEB_FETCH_TOOL_NAME, WEB_BROWSE_TOOL_NAME} and isinstance(
        normalized.get("url"), str
    ):
        normalized["url"] = normalized["url"].strip()
    return _sha256_text(f"{name}\n{_stable_json(normalized)}")


def _web_turn_scope(profile: "ModelProfile", request: dict[str, Any]) -> str:
    """Identify one user turn across Responses websocket reconnects."""

    input_items = request.get("input")
    items = input_items if isinstance(input_items, list) else []
    # Codex can replay partial assistant text after a broken stream. Excluding
    # only those messages keeps a reconnect stable while preserving local tool
    # calls/outputs, which must create a distinct generation scope.
    stable_items = [
        item
        for item in items
        if not (
            isinstance(item, dict)
            and item.get("type") == "message"
            and item.get("role") == "assistant"
        )
    ]
    seed = {
        "model": profile.slug,
        "prompt_cache_key": request.get("prompt_cache_key"),
        "input": stable_items,
        "instructions": _sha256_text(str(request.get("instructions") or "")),
        "tools": _sha256_text(_stable_json(request.get("tools") or [])),
    }
    return _sha256_text(_stable_json(seed))


def _active_ws_request_scope(request: dict[str, Any]) -> str | None:
    """Return the one in-flight generation scope owned by a Codex session."""

    if request.get("generate") is False:
        return None
    prompt_cache_key = request.get("prompt_cache_key")
    if not isinstance(prompt_cache_key, str) or not prompt_cache_key.strip():
        return None
    model = str(request.get("model") or "").strip()
    return f"{model}\0{prompt_cache_key.strip()}"


def _max_output_tokens(profile: "ModelProfile") -> int:
    dynamic_default = max(
        2_048,
        min(DEFAULT_MAX_OUTPUT_TOKENS, profile.context_window // 8),
    )
    return max(256, _env_int("MARATHON_MAX_OUTPUT_TOKENS", dynamic_default))


def _tool_thinking_budget_for_turn(
    request: dict[str, Any],
    delta_input: list[dict[str, Any]],
) -> int | None:
    """Return a native backend thinking cap only for post-tool continuations."""

    configured = os.environ.get("MARATHON_MODEL_TOOL_THINKING_BUDGET_TOKENS")
    if configured is None or not _env_bool("MARATHON_ADAPTIVE_THINKING_BUDGET", True):
        return None
    tools = request.get("tools")
    has_tools = isinstance(tools, list) and bool(tools)
    follows_tool = any(
        isinstance(item, dict)
        and item.get("type") in {
            "function_call_output",
            "custom_tool_call_output",
            "local_shell_call_output",
        }
        for item in delta_input
    )
    if not has_tools or not follows_tool:
        return None
    try:
        return max(0, int(configured))
    except ValueError:
        return None


def _response_stalled_at_output_limit(
    response: dict[str, Any],
    items: list[dict[str, Any]],
    output_limit: int,
) -> bool:
    usage = response.get("usage")
    output_tokens = usage.get("output_tokens") if isinstance(usage, dict) else None
    if not isinstance(output_tokens, int) or output_tokens < output_limit:
        return False
    actionable_types = {"function_call", "custom_tool_call", "local_shell_call"}
    for item in items:
        if item.get("type") in actionable_types:
            return False
        if _is_assistant_message_item(item) and not _is_ellipsis_filler_text(
            _assistant_message_text(item)
        ):
            return False
    return True


def _stalled_recovery_message() -> dict[str, Any]:
    return {
        "type": "message",
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": (
                    "Your previous response reached the generation budget without "
                    "producing a message or tool call. Do not continue internal "
                    "analysis. Use one available tool now to make concrete progress; "
                    "split large edits into smaller tool calls."
                ),
            }
        ],
    }


def _tool_protocol_recovery_message(reason: str) -> dict[str, Any]:
    return {
        "type": "message",
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": (
                    f"Marathon aborted the previous tool call because {reason}. "
                    "Do not repeat or continue that output. Call one available tool "
                    "now with valid, concise arguments. Split large file edits into "
                    "several smaller apply_patch calls."
                ),
            }
        ],
    }


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip()


def _input_modalities_from_env() -> tuple[str, ...]:
    values = [
        value.strip().lower()
        for value in _env_str("MARATHON_MODEL_INPUT_MODALITIES", "text").split(",")
        if value.strip()
    ]
    supported = tuple(
        dict.fromkeys(value for value in values if value in {"text", "image"})
    )
    return supported or ("text",)


def _reasoning_config_from_env() -> tuple[str | None, tuple[tuple[str, str], ...]]:
    raw_levels = os.getenv("MARATHON_MODEL_REASONING_LEVELS", "[]")
    try:
        parsed_levels = json.loads(raw_levels)
    except json.JSONDecodeError as exc:
        raise ValueError("MARATHON_MODEL_REASONING_LEVELS must be valid JSON") from exc
    if not isinstance(parsed_levels, list):
        raise ValueError("MARATHON_MODEL_REASONING_LEVELS must be a JSON list")

    levels: list[tuple[str, str]] = []
    for item in parsed_levels:
        if not isinstance(item, dict):
            raise ValueError("each reasoning level must be a JSON object")
        effort = str(item.get("effort") or "").strip()
        if not effort:
            raise ValueError("reasoning effort names must not be empty")
        levels.append((effort, str(item.get("description") or "").strip()))

    efforts = [effort for effort, _description in levels]
    if len(efforts) != len(set(efforts)):
        raise ValueError("reasoning effort names must be unique")
    default = os.getenv("MARATHON_MODEL_DEFAULT_REASONING_LEVEL")
    default = default.strip() if default and default.strip() else None
    if default is not None and default not in efforts:
        raise ValueError(f"default reasoning effort {default!r} is not supported")
    if levels and default is None:
        raise ValueError("reasoning levels require a default reasoning effort")
    return default, tuple(levels)


def _safe_model_slug(raw: str) -> str:
    value = raw.strip()
    if re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        return value
    return "custom"


def _custom_model_profile(root: Path) -> ModelProfile | None:
    model_path = os.getenv("MARATHON_MODEL_PATH") or os.getenv("MODEL_PATH")
    if not model_path or not model_path.strip():
        return None

    slug = _safe_model_slug(_env_str("MARATHON_MODEL_SLUG", "custom"))
    context_window = _env_int(
        "MARATHON_MODEL_CONTEXT",
        _env_int("MARATHON_CONTEXT", _env_int("CTX_SIZE", 32768)),
    )
    auto_compact_limit = _env_int(
        "MARATHON_MODEL_AUTO_COMPACT_TOKEN_LIMIT",
        max(1, context_window * 9 // 10),
    )
    truncation_limit = _env_int("MARATHON_MODEL_TRUNCATION_LIMIT", auto_compact_limit)
    port = _env_int("MARATHON_MODEL_PORT", 18095)
    temperature_raw = os.getenv("MARATHON_MODEL_TEMPERATURE")
    temperature = (
        _env_float("MARATHON_MODEL_TEMPERATURE", 0.0)
        if temperature_raw is not None and temperature_raw.strip()
        else None
    )
    default_reasoning_level, supported_reasoning_levels = (
        _reasoning_config_from_env()
    )

    return ModelProfile(
        slug=slug,
        alias=_safe_model_slug(_env_str("MARATHON_BACKEND_MODEL_ID", slug)),
        display_name=_env_str("MARATHON_MODEL_DISPLAY_NAME", slug),
        description=_env_str("MARATHON_MODEL_DESCRIPTION", "Custom GGUF model served by llama.cpp."),
        launcher=str(root / "scripts/launchers/server_custom.sh"),
        model_paths=(str(Path(model_path).expanduser()),),
        target=_target_override("MARATHON_MODEL_TARGET", f"http://127.0.0.1:{port}"),
        context_window=context_window,
        auto_compact_token_limit=auto_compact_limit,
        truncation_limit=truncation_limit,
        supports_parallel_tool_calls=_env_bool(
            "MARATHON_MODEL_PARALLEL_TOOL_CALLS", False
        ),
        supports_slots=_env_bool("MARATHON_BACKEND_SLOT_API", True),
        supervised=_env_bool("MARATHON_MODEL_SUPERVISED", False),
        temperature=temperature,
        default_reasoning_level=default_reasoning_level,
        supported_reasoning_levels=supported_reasoning_levels,
        input_modalities=_input_modalities_from_env(),
    )


def _profiles() -> dict[str, ModelProfile]:
    root = _repo_root()
    profiles = {
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
        "qwen3.6-27b-q4-128k-single": ModelProfile(
            slug="qwen3.6-27b-q4-128k-single",
            alias="qwen3.6-27b-q4-128k-single",
            display_name="Qwen3.6 27B Q4 128K Single GPU",
            description="Single-GPU long-context local Qwen3.6 27B Q4 profile",
            launcher=str(root / "scripts/launchers/server_27b_128k_single_gpu.sh"),
            model_paths=_model_candidates(
                "QWEN36_27B_GGUF",
                (
                    "Qwen3.6-27B-GGUF/qwen3.6-27b-q4_k_m.gguf",
                    "Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf",
                ),
            ),
            target=_target_override(
                "MARATHON_QWEN36_27B_128K_SINGLE_TARGET",
                "http://127.0.0.1:18094",
            ),
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
            display_name="Qwen3.6 35B A3B 128K",
            description="Long-context single-GPU specialist local Qwen3.6 35B A3B profile",
            launcher=str(root / "scripts/launchers/server_35b_a3b.sh"),
            model_paths=_model_candidates(
                "QWEN36_35B_A3B_GGUF",
                (
                    "Qwen3.6-35B-A3B-GGUF/qwen3.6-35b-a3b-q4_k_m.gguf",
                    "Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf",
                ),
            ),
            target=_target_override("MARATHON_QWEN36_35B_A3B_TARGET", "http://127.0.0.1:18092"),
            context_window=131072,
            auto_compact_token_limit=115000,
            truncation_limit=110000,
        ),
        "qwopus3.6-35b-a3b-v1": ModelProfile(
            slug="qwopus3.6-35b-a3b-v1",
            alias="qwopus3.6-35b-a3b-v1",
            display_name="Qwopus3.6 35B A3B v1",
            description="Qwopus3.6 35B A3B v1 GGUF profile served by llama.cpp.",
            launcher=str(root / "scripts/launchers/server_35b_a3b.sh"),
            model_paths=_model_candidates(
                "QWOPUS36_35B_A3B_GGUF",
                (
                    "Qwopus3.6-35B-A3B-v1-GGUF/Qwopus3.6-35B-A3B-v1-Q4_K_M.gguf",
                    "Qwopus3.6-35B-A3B-v1-GGUF/Qwopus3.6-35B-A3B-v1-IQ4_XS.gguf",
                ),
            ),
            target=_target_override("MARATHON_QWOPUS36_35B_A3B_TARGET", "http://127.0.0.1:18096"),
            context_window=131072,
            auto_compact_token_limit=115000,
            truncation_limit=110000,
        ),
        "gemma4-26b-a4b-it-128k": ModelProfile(
            slug="gemma4-26b-a4b-it-128k",
            alias="gemma4-26b-a4b-it-128k",
            display_name="Gemma 4 26B A4B IT 128K",
            description="Single-GPU local Gemma 4 26B A4B instruction profile served by llama.cpp.",
            launcher=str(root / "scripts/launchers/server_gemma4_26b_a4b.sh"),
            model_paths=_model_candidates(
                "GEMMA4_26B_A4B_GGUF",
                (
                    "gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-Q4_K_M.gguf",
                    "gemma-4-26b-a4b-it-GGUF/gemma-4-26B-A4B-it-Q4_K_M.gguf",
                ),
            ),
            target=_target_override("MARATHON_GEMMA4_26B_A4B_TARGET", "http://127.0.0.1:18097"),
            context_window=131072,
            auto_compact_token_limit=115000,
            truncation_limit=110000,
        ),
    }
    custom_profile = _custom_model_profile(root)
    if custom_profile is not None:
        profiles[custom_profile.slug] = custom_profile
    return profiles


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


def _is_assistant_message_item(item: Any) -> bool:
    return (
        isinstance(item, dict)
        and item.get("type") == "message"
        and item.get("role") == "assistant"
    )


def _starts_followup_work(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    return str(item.get("type") or "") in FOLLOWUP_WORK_ITEM_TYPES


def _completed_message_phase(items: list[dict[str, Any]]) -> str:
    """Keep the working indicator alive when a response also requested work."""

    return "commentary" if any(_starts_followup_work(item) for item in items) else "final_answer"


def _set_assistant_message_phase(item: dict[str, Any], phase: str) -> None:
    if _is_assistant_message_item(item) and not item.get("phase"):
        item["phase"] = phase


def _assistant_message_text(item: dict[str, Any]) -> str:
    if not _is_assistant_message_item(item):
        return ""
    content = item.get("content")
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") not in {"output_text", "text"}:
            continue
        text = part.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def _is_ellipsis_filler_text(text: str) -> bool:
    compact = "".join(text.strip().split())
    if not compact:
        return True
    return len(compact) <= 6 and all(ch in {".", "…"} for ch in compact)


def _is_droppable_commentary_message(item: dict[str, Any]) -> bool:
    return (
        _is_assistant_message_item(item)
        and item.get("phase") == "commentary"
        and _is_ellipsis_filler_text(_assistant_message_text(item))
    )


def _annotate_message_phases(items: list[dict[str, Any]], final_response: bool) -> None:
    contains_followup_work = any(_starts_followup_work(item) for item in items)
    phase = "commentary" if contains_followup_work or not final_response else "final_answer"
    for item in items:
        _set_assistant_message_phase(item, phase)


def _pop_sse_frame(buffer: bytes) -> tuple[bytes | None, bytes]:
    candidates = [
        (idx, len(marker))
        for marker in (b"\n\n", b"\r\n\r\n")
        if (idx := buffer.find(marker)) >= 0
    ]
    if not candidates:
        return None, buffer
    idx, marker_len = min(candidates, key=lambda item: item[0])
    return buffer[:idx], buffer[idx + marker_len :]


def _parse_sse_frame(frame: bytes) -> SseEvent | None:
    text = frame.decode("utf-8", errors="replace")
    event_name: str | None = None
    data_lines: list[str] = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if not raw_line or raw_line.startswith(":"):
            continue
        if raw_line.startswith("event:"):
            event_name = raw_line[6:].lstrip(" ")
            continue
        if raw_line.startswith("data:"):
            data_lines.append(raw_line[5:].lstrip(" "))
    if not data_lines:
        return None
    return SseEvent(name=event_name, data="\n".join(data_lines))


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


def _apply_patch_function_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "name": APPLY_PATCH_TOOL_NAME,
        "description": (
            "Edit files with structured operations. For replace, old_text must "
            "exactly match existing file text and new_text is its replacement. "
            "Use several small replace operations for unrelated edits. Marathon "
            "will compile these operations into Codex's native patch format; do "
            "not write a raw diff or patch envelope."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "operations": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "oneOf": [
                            {
                                "type": "object",
                                "properties": {
                                    "action": {"type": "string", "enum": ["add"]},
                                    "path": {"type": "string"},
                                    "content": {"type": "string"},
                                },
                                "required": ["action", "path", "content"],
                                "additionalProperties": False,
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "action": {"type": "string", "enum": ["replace"]},
                                    "path": {"type": "string"},
                                    "old_text": {"type": "string"},
                                    "new_text": {"type": "string"},
                                },
                                "required": ["action", "path", "old_text", "new_text"],
                                "additionalProperties": False,
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "action": {"type": "string", "enum": ["delete"]},
                                    "path": {"type": "string"},
                                },
                                "required": ["action", "path"],
                                "additionalProperties": False,
                            },
                        ],
                    },
                }
            },
            "required": ["operations"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def _backend_tool_for_llama(tool: Any) -> dict[str, Any] | None:
    if not isinstance(tool, dict):
        return None
    if tool.get("type") == "function":
        return tool
    if tool.get("type") == "custom" and tool.get("name") == APPLY_PATCH_TOOL_NAME:
        return _apply_patch_function_tool()
    return None


def _patch_lines(value: str, prefix: str) -> list[str]:
    lines = value.splitlines()
    if not lines and value == "":
        return []
    return [prefix + line for line in lines]


def _contains_patch_envelope(value: str) -> bool:
    return any(
        line.strip() in {"*** Begin Patch", "*** End Patch"}
        for line in value.splitlines()
    )


def _tool_argument_max_chars() -> int:
    return max(
        1_024,
        _env_int("MARATHON_TOOL_ARGUMENT_MAX_CHARS", DEFAULT_TOOL_ARGUMENT_MAX_CHARS),
    )


def _has_runaway_repetition(value: str) -> bool:
    """Detect a long exact suffix loop without judging normal repeated syntax."""

    if len(value) < 2_048:
        return False
    tail = value[-8_192:]
    for width in range(1, 129):
        repeats = max(32, (2_048 + width - 1) // width)
        span = width * repeats
        if span > len(tail):
            continue
        unit = tail[-width:]
        if tail[-span:] == unit * repeats:
            return True
    return False


def _partial_tool_argument_error(arguments: str, limit: int) -> str | None:
    if len(arguments) > limit:
        return f"tool arguments exceeded {limit:,} characters"
    if _has_runaway_repetition(arguments):
        return "tool arguments entered an exact repetition loop"
    return None


def _apply_patch_protocol_error(item: dict[str, Any], limit: int) -> str | None:
    arguments = item.get("arguments")
    if isinstance(arguments, str):
        error = _partial_tool_argument_error(arguments, limit)
        if error:
            return error
    elif not isinstance(arguments, dict):
        return "apply_patch arguments were missing"
    if not _apply_patch_input_from_arguments(arguments):
        return "apply_patch arguments were not valid structured JSON"
    return None


def _response_tool_protocol_error(
    response: dict[str, Any],
    limit: int,
) -> str | None:
    output = response.get("output")
    if not isinstance(output, list):
        return None
    for item in output:
        if isinstance(item, dict) and _is_apply_patch_function_call(item):
            error = _apply_patch_protocol_error(item, limit)
            if error:
                return error
    return None


def _structured_patch_to_input(operations: Any) -> str:
    if not isinstance(operations, list) or not operations:
        return ""
    result = ["*** Begin Patch"]
    for operation in operations:
        if not isinstance(operation, dict):
            return ""
        action = operation.get("action")
        path = operation.get("path")
        if (
            not isinstance(path, str)
            or not path.strip()
            or "\n" in path
            or "\r" in path
        ):
            return ""
        path = path.strip()
        if action == "add":
            content = operation.get("content")
            if not isinstance(content, str) or _contains_patch_envelope(content):
                return ""
            result.append(f"*** Add File: {path}")
            result.extend(_patch_lines(content, "+"))
        elif action == "delete":
            result.append(f"*** Delete File: {path}")
        elif action == "replace":
            old_text = operation.get("old_text")
            new_text = operation.get("new_text")
            if (
                not isinstance(old_text, str)
                or not old_text
                or not isinstance(new_text, str)
                or _contains_patch_envelope(old_text)
                or _contains_patch_envelope(new_text)
            ):
                return ""
            result.extend([f"*** Update File: {path}", "@@"])
            result.extend(_patch_lines(old_text, "-"))
            result.extend(_patch_lines(new_text, "+"))
        else:
            return ""
    result.append("*** End Patch")
    return "\n".join(result)


def _apply_patch_input_from_arguments(arguments: Any) -> str:
    if isinstance(arguments, dict):
        structured = _structured_patch_to_input(arguments.get("operations"))
        if structured:
            return structured
        value = arguments.get("input")
        return _normalize_apply_patch_dialect(value) if isinstance(value, str) else ""
    if not isinstance(arguments, str):
        return ""
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return (
            _normalize_apply_patch_dialect(arguments)
            if arguments.startswith("*** Begin Patch")
            else ""
        )
    if isinstance(parsed, dict) and isinstance(parsed.get("input"), str):
        return _normalize_apply_patch_dialect(parsed["input"])
    if isinstance(parsed, dict):
        return _structured_patch_to_input(parsed.get("operations"))
    return ""


def _apply_patch_arguments_from_input(patch_input: Any) -> str:
    return json.dumps(
        {"input": patch_input if isinstance(patch_input, str) else ""},
        separators=(",", ":"),
    )


def _is_apply_patch_function_call(item: Any) -> bool:
    return (
        isinstance(item, dict)
        and item.get("type") == "function_call"
        and item.get("name") == APPLY_PATCH_TOOL_NAME
    )


def _is_apply_patch_custom_call(item: Any) -> bool:
    return (
        isinstance(item, dict)
        and item.get("type") == "custom_tool_call"
        and item.get("name") == APPLY_PATCH_TOOL_NAME
    )


def _stream_keys_for_item(item_id: str, item: dict[str, Any]) -> list[str]:
    keys = []
    for value in (item_id, item.get("id"), item.get("call_id")):
        if isinstance(value, str) and value and value not in keys:
            keys.append(value)
    return keys


def _apply_patch_function_to_custom_call(item: dict[str, Any]) -> dict[str, Any]:
    converted = copy.deepcopy(item)
    converted["type"] = "custom_tool_call"
    converted["input"] = _apply_patch_input_from_arguments(converted.pop("arguments", ""))
    converted.pop("namespace", None)
    return converted


def _apply_patch_custom_to_function_call(item: dict[str, Any]) -> dict[str, Any]:
    converted = copy.deepcopy(item)
    converted["type"] = "function_call"
    converted["arguments"] = _apply_patch_arguments_from_input(converted.pop("input", ""))
    converted.pop("status", None)
    return converted


def _apply_patch_custom_output_to_function_output(item: dict[str, Any]) -> dict[str, Any]:
    converted = copy.deepcopy(item)
    converted["type"] = "function_call_output"
    converted.pop("name", None)
    return converted


def _backend_lineage_item(item: dict[str, Any]) -> dict[str, Any]:
    """Store llama.cpp-compatible items even when Codex saw a custom patch item."""

    if _is_apply_patch_custom_call(item):
        converted = _apply_patch_custom_to_function_call(item)
        backend_arguments = item.get(_BACKEND_ARGUMENTS_KEY)
        original_input = item.get("input")
        if (
            isinstance(backend_arguments, str)
            and backend_arguments
            and isinstance(original_input, str)
            and original_input
            and _apply_patch_input_from_arguments(backend_arguments) == original_input
        ):
            converted["arguments"] = backend_arguments
        converted.pop(_BACKEND_ARGUMENTS_KEY, None)
        return converted
    if item.get("type") == "custom_tool_call_output":
        return _apply_patch_custom_output_to_function_output(item)
    converted = copy.deepcopy(item)
    converted.pop(_BACKEND_ARGUMENTS_KEY, None)
    return converted


def _apply_reasoning_effort(
    data: dict[str, Any], profile: ModelProfile | None
) -> None:
    if profile is None or not profile.supported_reasoning_levels:
        return
    supported = {
        effort for effort, _description in profile.supported_reasoning_levels
    }
    reasoning = data.get("reasoning")
    if reasoning is not None and not isinstance(reasoning, dict):
        raise ValueError("reasoning must be an object")
    effort = reasoning.get("effort") if isinstance(reasoning, dict) else None
    if effort is None:
        effort = profile.default_reasoning_level
    if not isinstance(effort, str) or effort not in supported:
        choices = ", ".join(effort_name for effort_name, _ in profile.supported_reasoning_levels)
        raise ValueError(
            f"reasoning effort {effort!r} is not supported by {profile.slug}; "
            f"choose one of: {choices}"
        )

    template_kwargs = data.get("chat_template_kwargs")
    if template_kwargs is not None and not isinstance(template_kwargs, dict):
        raise ValueError("chat_template_kwargs must be an object")
    normalized_template_kwargs = dict(template_kwargs or {})
    if effort == "none":
        normalized_template_kwargs.pop("reasoning_effort", None)
        normalized_template_kwargs["enable_thinking"] = False
    else:
        normalized_template_kwargs.update(
            enable_thinking=True,
            reasoning_effort=effort,
        )
    data["chat_template_kwargs"] = normalized_template_kwargs


def normalize_responses_request(
    data: dict[str, Any], profile: ModelProfile | None = None
) -> dict[str, Any]:
    original_instructions = data.get("instructions")
    instruction_base = original_instructions if isinstance(original_instructions, str) else ""
    data["_marathon_instruction_base_hash"] = _sha256_text(instruction_base)
    _apply_reasoning_effort(data, profile)
    tools = data.get("tools")
    web_search_requested = request_has_web_search_tool(tools) if isinstance(tools, list) else False
    if isinstance(tools, list):
        data["tools"] = [
            converted
            for tool in tools
            if (converted := _backend_tool_for_llama(tool)) is not None
        ]
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
        data["_marathon_lifted_instruction_count"] = 0
        return data

    lifted_messages: list[str] = []
    normalized_input: list[Any] = []
    input_changed = False
    tool_output_truncations = 0
    tool_output_limit = max(
        1,
        _env_int("MARATHON_TOOL_OUTPUT_MAX_CHARS", DEFAULT_TOOL_OUTPUT_MAX_CHARS),
    )
    malformed_call_keys = _malformed_function_call_keys(input_items)
    malformed_tool_replay_drops = 0
    for item in input_items:
        if not isinstance(item, dict):
            normalized_input.append(item)
            continue

        if _is_malformed_function_call(item, malformed_call_keys):
            input_changed = True
            malformed_tool_replay_drops += 1
            continue

        if (
            item.get("type") == "function_call_output"
            and item.get("call_id") in malformed_call_keys
        ):
            input_changed = True
            malformed_tool_replay_drops += 1
            continue

        bounded_item, bounded = _bound_tool_output_item(item, tool_output_limit)
        if bounded:
            item = bounded_item
            input_changed = True
            tool_output_truncations += 1

        image_items = _backend_image_tool_output_items(item)
        if image_items is not None:
            normalized_input.extend(image_items)
            input_changed = True
            continue

        if _is_apply_patch_custom_call(item):
            normalized_input.append(_apply_patch_custom_to_function_call(item))
            input_changed = True
            continue

        if item.get("type") == "custom_tool_call_output":
            normalized_input.append(_apply_patch_custom_output_to_function_output(item))
            input_changed = True
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
    data["_marathon_lifted_instruction_count"] = len(lifted_messages)
    data["_marathon_tool_output_truncations"] = tool_output_truncations
    data["_marathon_malformed_tool_replay_drops"] = malformed_tool_replay_drops

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
        self.telemetry = EventWriter.from_env("router")
        self.available_profiles: dict[str, ModelProfile] = {}
        self._refresh_profiles()
        if not self.available_profiles:
            raise RuntimeError("no available local model profiles found")
        if default_model not in self.available_profiles:
            default_model = next(iter(self.available_profiles))
        self.default_model = default_model
        self.current_model: str | None = None
        self.slot_id = int(os.getenv("MARATHON_ROUTER_SLOT_ID") or "0")
        self.experimental_delta_only = bool(os.getenv("MARATHON_WS_EXPERIMENTAL_DELTA_ONLY"))
        configured_slot_save_root = os.getenv("MARATHON_SLOT_SAVE_ROOT")
        self.slot_save_root = (
            Path(configured_slot_save_root).expanduser()
            if configured_slot_save_root
            else _repo_root() / ".marathon" / "llama-slots"
        )
        self.slot_snapshot_max_count = max(
            0,
            _env_int("MARATHON_SLOT_SNAPSHOT_MAX_COUNT", DEFAULT_SLOT_SNAPSHOT_MAX_COUNT),
        )
        self.slot_snapshot_max_bytes = max(
            0,
            _env_int("MARATHON_SLOT_SNAPSHOT_MAX_BYTES", DEFAULT_SLOT_SNAPSHOT_MAX_BYTES),
        )
        self.slot_snapshot_clean_startup = _env_bool(
            "MARATHON_SLOT_SNAPSHOT_CLEAN_STARTUP",
            DEFAULT_SLOT_SNAPSHOT_CLEAN_STARTUP,
        )
        self.slot_snapshots_enabled = _env_bool(
            "MARATHON_SLOT_SNAPSHOTS_ENABLED",
            DEFAULT_SLOT_SNAPSHOTS_ENABLED,
        )
        self.starter_cache_enabled = _env_bool(
            "MARATHON_STARTER_CACHE_ENABLED",
            DEFAULT_STARTER_CACHE_ENABLED,
        )
        self.starter_cache_max_count = max(
            1,
            _env_int(
                "MARATHON_STARTER_CACHE_MAX_COUNT",
                DEFAULT_STARTER_CACHE_MAX_COUNT,
            ),
        )
        self.starter_cache_max_bytes = max(
            0,
            _env_int(
                "MARATHON_STARTER_CACHE_MAX_BYTES",
                DEFAULT_STARTER_CACHE_MAX_BYTES,
            ),
        )
        self.backend_cache_id = os.getenv("MARATHON_BACKEND_CACHE_ID", "")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.trace_log_path = self.log_dir / "codex_local_router_trace.jsonl"
        self.request_log_path = self.log_dir / "codex_local_router_request.json"
        self._trace_seq = 0
        self._response_id_seq = 0
        self._last_trace_by_model: dict[str, dict[str, Any]] = {}
        self.lineage: dict[str, ResponseSnapshot] = {}
        self.last_response_by_model: dict[str, str] = {}
        self.live_slot_by_model: dict[str, str] = {}
        self.live_prompt_cache_key_by_model: dict[str, str] = {}
        self.active_ws_tasks: dict[str, asyncio.Task[Any]] = {}
        self.web_tool_cache_max_entries = max(
            1,
            _env_int(
                "MARATHON_WEB_TOOL_CACHE_MAX_ENTRIES",
                DEFAULT_WEB_TOOL_CACHE_MAX_ENTRIES,
            ),
        )
        self.web_tool_cache: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()
        self.web_turn_progress_max_entries = max(
            1,
            _env_int(
                "MARATHON_WEB_TURN_PROGRESS_MAX_ENTRIES",
                DEFAULT_WEB_TURN_PROGRESS_MAX_ENTRIES,
            ),
        )
        self.web_turn_progress_ttl_seconds = max(
            1,
            _env_int(
                "MARATHON_WEB_TURN_PROGRESS_TTL_SECONDS",
                DEFAULT_WEB_TURN_PROGRESS_TTL_SECONDS,
            ),
        )
        self.web_turn_progress: OrderedDict[str, ManagedWebTurnProgress] = OrderedDict()
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
        active_tasks = list(self.active_ws_tasks.values())
        self.active_ws_tasks.clear()
        for task in active_tasks:
            if not task.done():
                task.cancel()
        if active_tasks:
            await asyncio.gather(*active_tasks, return_exceptions=True)
        if self.web_search is not None:
            await self.web_search.close()
            self.web_search = None
        if self.web_fetch is not None:
            await self.web_fetch.close()
            self.web_fetch = None
        if self.http_client is not None:
            await self.http_client.close()
            self.http_client = None

    def replace_active_ws_task(
        self,
        scope: str | None,
        task: asyncio.Task[Any],
    ) -> asyncio.Task[Any] | None:
        if scope is None:
            return None
        previous = self.active_ws_tasks.get(scope)
        self.active_ws_tasks[scope] = task

        def remove_when_done(done: asyncio.Task[Any]) -> None:
            if self.active_ws_tasks.get(scope) is done:
                self.active_ws_tasks.pop(scope, None)

        task.add_done_callback(remove_when_done)
        if previous is task or previous is None or previous.done():
            return None
        return previous

    def _prune_web_turn_progress(self) -> None:
        cache = getattr(self, "web_turn_progress", None)
        if cache is None:
            cache = OrderedDict()
            self.web_turn_progress = cache
        ttl = max(
            1,
            int(
                getattr(
                    self,
                    "web_turn_progress_ttl_seconds",
                    DEFAULT_WEB_TURN_PROGRESS_TTL_SECONDS,
                )
            ),
        )
        cutoff = time.time() - ttl
        for scope in list(cache):
            if cache[scope].updated_at < cutoff:
                cache.pop(scope, None)
        max_entries = max(
            1,
            int(
                getattr(
                    self,
                    "web_turn_progress_max_entries",
                    DEFAULT_WEB_TURN_PROGRESS_MAX_ENTRIES,
                )
            ),
        )
        while len(cache) > max_entries:
            cache.popitem(last=False)

    def load_web_turn_progress(self, scope: str) -> ManagedWebTurnProgress | None:
        self._prune_web_turn_progress()
        progress = self.web_turn_progress.get(scope)
        if progress is None:
            return None
        self.web_turn_progress.move_to_end(scope)
        return copy.deepcopy(progress)

    def save_web_turn_progress(
        self,
        scope: str,
        progress: ManagedWebTurnProgress,
    ) -> None:
        stored = copy.deepcopy(progress)
        stored.updated_at = time.time()
        self.web_turn_progress[scope] = stored
        self.web_turn_progress.move_to_end(scope)
        self._prune_web_turn_progress()

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
                    "default_reasoning_level": profile.default_reasoning_level,
                    "supported_reasoning_levels": [
                        {"effort": effort, "description": description}
                        for effort, description in profile.supported_reasoning_levels
                    ],
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
                    "apply_patch_tool_type": "freeform",
                    "web_search_tool_type": "text",
                    "truncation_policy": {"mode": "tokens", "limit": profile.truncation_limit},
                    "supports_parallel_tool_calls": profile.supports_parallel_tool_calls,
                    "supports_image_detail_original": False,
                    "context_window": profile.context_window,
                    "max_context_window": profile.context_window,
                    "auto_compact_token_limit": profile.auto_compact_token_limit,
                    "effective_context_window_percent": 100,
                    "experimental_supported_tools": [],
                    "input_modalities": list(profile.input_modalities),
                    "supports_search_tool": True,
                }
            )
            data.append(
                {
                    "id": profile.slug,
                    "object": "model",
                    "owned_by": "local-codex-router",
                    "description": profile.description,
                    # OpenAI does not require these fields, but local clients
                    # such as Hermes use them to discover the context actually
                    # allocated by the active Marathon profile.
                    "context_length": profile.context_window,
                    "max_model_len": profile.context_window,
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
        telemetry_started = time.perf_counter()
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
            entry["telemetry_prepare_ms"] = (time.perf_counter() - telemetry_started) * 1000.0
            self._last_trace_by_model[profile.slug] = {
                "input_items": copy.deepcopy(normalized_input_items),
                "instructions_hash": instructions_hash,
                "tools_hash": tools_hash,
                "body_hash": normalized_hash,
            }
            self.telemetry.emit("router.request.normalized", entry)
            if self.debug:
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
                if profile.supervised:
                    raise RuntimeError(
                        f"Marathon's supervised backend for {profile.slug} is unavailable"
                    )
                self._stop_profile(profile)
                self.live_slot_by_model.pop(profile.slug, None)
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

    def _pid_args(self, pid: int) -> tuple[str, ...]:
        try:
            raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        except OSError:
            return ()
        return tuple(
            item.decode("utf-8", errors="replace")
            for item in raw.split(b"\0")
            if item
        )

    def _process_start_ticks(self, pid: int) -> str | None:
        try:
            fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        except OSError:
            return None
        return fields[21] if len(fields) > 21 else None

    def _read_profile_pid(self, profile: ModelProfile) -> tuple[int, str | None] | None:
        try:
            fields = self._profile_pid_file(profile).read_text(encoding="utf-8").split()
            pid = int(fields[0])
        except (OSError, ValueError, IndexError):
            return None
        return pid, fields[1] if len(fields) > 1 else None

    def _owns_profile_pid(self, profile: ModelProfile, pid: int, started: str | None) -> bool:
        if started is not None and self._process_start_ticks(pid) != started:
            return False
        args = self._pid_args(pid)
        if not args or "llama-server" not in Path(args[0]).name:
            return False
        return any(
            args[index] == "--port" and args[index + 1] == str(profile.port)
            for index in range(len(args) - 1)
        )

    def _profile_ready(self, profile: ModelProfile) -> bool:
        return _json_model_matches(profile.target, profile.alias)

    def _stop_other_backends(self, keep_slug: str) -> None:
        for slug, profile in self.available_profiles.items():
            if slug == keep_slug:
                continue
            self._stop_profile(profile)

    def _stop_profile(self, profile: ModelProfile) -> None:
        if profile.supervised:
            return
        record = self._read_profile_pid(profile)
        port_pid = self._port_owner_pid(profile.port)
        if record is None:
            self.live_slot_by_model.pop(profile.slug, None)
            self._profile_pid_file(profile).unlink(missing_ok=True)
            if port_pid is not None:
                cmd = self._pid_cmdline(port_pid)
                raise RuntimeError(
                    f"port {profile.port} is occupied by a process Marathon does not own: {cmd}"
                )
            return
        pid, started = record
        if not self._owns_profile_pid(profile, pid, started):
            self._profile_pid_file(profile).unlink(missing_ok=True)
            if port_pid is not None:
                foreign = self._pid_cmdline(port_pid)
                raise RuntimeError(
                    f"port {profile.port} is occupied by a process Marathon does not own: {foreign}"
                )
            return
        if port_pid is not None and port_pid != pid:
            foreign = self._pid_cmdline(port_pid)
            raise RuntimeError(
                f"port {profile.port} is occupied by a process Marathon does not own: {foreign}"
            )
        owned_started = started or self._process_start_ticks(pid)
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        for _ in range(10):
            time.sleep(1)
            if self._process_start_ticks(pid) != owned_started:
                self.live_slot_by_model.pop(profile.slug, None)
                self._profile_pid_file(profile).unlink(missing_ok=True)
                return
        if self._process_start_ticks(pid) == owned_started:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.live_slot_by_model.pop(profile.slug, None)
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
        started = self._process_start_ticks(proc.pid) or ""
        self._profile_pid_file(profile).write_text(
            f"{proc.pid} {started}\n", encoding="utf-8"
        )

    def _wait_for_profile(self, profile: ModelProfile) -> None:
        for _ in range(240):
            if self._profile_ready(profile):
                return
            time.sleep(1)
        raise RuntimeError(
            f"backend for {profile.slug} did not become ready; see {self._profile_log_file(profile)}"
        )

    async def _request_json(
        self,
        profile: ModelProfile,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        retry_connection_error: bool = False,
    ) -> dict[str, Any]:
        if self.http_client is None:
            raise RuntimeError("router HTTP client session is not open")
        url = f"{profile.target.rstrip('/')}{path}"
        attempts = 2 if retry_connection_error else 1
        last_connection_error: Exception | None = None
        for attempt in range(attempts):
            try:
                async with self.http_client.request(method, url, json=payload) as response:
                    text = await response.text()
                    if response.status >= 400:
                        protocol_error = _backend_tool_protocol_error_reason(
                            response.status,
                            text,
                        )
                        if protocol_error:
                            raise ToolProtocolError(protocol_error)
                        raise RuntimeError(f"backend {method} {path} failed: {response.status} {text}")
                    try:
                        return json.loads(text) if text else {}
                    except json.JSONDecodeError as exc:
                        raise RuntimeError(f"backend {method} {path} returned invalid JSON: {exc}") from exc
            except RuntimeError:
                raise
            except Exception as exc:
                last_connection_error = exc
                if attempt + 1 >= attempts:
                    break
                await asyncio.sleep(0.2)
        raise RuntimeError(f"backend {method} {path} connection failed: {last_connection_error}")

    async def _iter_sse_json(self, response: Any) -> Any:
        buffer = b""
        async for chunk in response.content.iter_chunked(8192):
            buffer += chunk
            while True:
                frame, buffer = _pop_sse_frame(buffer)
                if frame is None:
                    break
                event = _parse_sse_frame(frame)
                if event is None:
                    continue
                if event.data == "[DONE]":
                    continue
                try:
                    payload = json.loads(event.data)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    yield payload

        if buffer.strip():
            event = _parse_sse_frame(buffer)
            if event is not None and event.data != "[DONE]":
                try:
                    payload = json.loads(event.data)
                except json.JSONDecodeError:
                    return
                if isinstance(payload, dict):
                    yield payload

    async def _request_responses_stream(
        self,
        profile: ModelProfile,
        payload: dict[str, Any],
        *,
        event_sink: StreamEventSink | None = None,
    ) -> dict[str, Any]:
        if self.http_client is None:
            raise RuntimeError("router HTTP client session is not open")

        request = copy.deepcopy(payload)
        request["stream"] = True
        url = f"{profile.target.rstrip('/')}/v1/responses"
        output_items: list[dict[str, Any]] = []
        completed_response: dict[str, Any] | None = None
        suppressed_item_ids: set[str] = set()
        pending_message_done: dict[str, Any] | None = None
        pending_message_added: dict[str, dict[str, Any]] = {}
        pending_message_deltas: dict[str, list[dict[str, Any]]] = {}
        pending_message_text: dict[str, str] = {}
        forwarded_message_ids: set[str] = set()
        apply_patch_function_items: dict[str, str] = {}
        apply_patch_argument_buffers: dict[str, str] = {}
        pending_apply_patch_added: dict[str, dict[str, Any]] = {}
        tool_argument_limit = _tool_argument_max_chars()

        async def send_event(event: dict[str, Any]) -> None:
            if event_sink is None:
                return
            if not await event_sink(event):
                raise ConnectionError("websocket client disconnected")

        async def flush_pending_message_stream(item_id: str) -> None:
            if not item_id or item_id in forwarded_message_ids:
                return
            added = pending_message_added.pop(item_id, None)
            if added is not None:
                await send_event(added)
            for delta_event in pending_message_deltas.pop(item_id, []):
                await send_event(delta_event)
            pending_message_text.pop(item_id, None)
            forwarded_message_ids.add(item_id)

        def drop_pending_message_stream(item_id: str) -> None:
            if not item_id:
                return
            pending_message_added.pop(item_id, None)
            pending_message_deltas.pop(item_id, None)
            pending_message_text.pop(item_id, None)
            forwarded_message_ids.discard(item_id)

        async def flush_pending_message(phase: str) -> None:
            nonlocal pending_message_done
            if pending_message_done is None:
                return
            item = pending_message_done.get("item")
            if isinstance(item, dict):
                _set_assistant_message_phase(item, phase)
                item_id = str(item.get("id") or "")
                if _is_droppable_commentary_message(item):
                    drop_pending_message_stream(item_id)
                    pending_message_done = None
                    return
                await flush_pending_message_stream(item_id)
                output_items.append(copy.deepcopy(item))
            await send_event(pending_message_done)
            pending_message_done = None

        async with self.http_client.post(url, json=request) as response:
            if response.status >= 400:
                text = await response.text()
                protocol_error = _backend_tool_protocol_error_reason(
                    response.status,
                    text,
                )
                if protocol_error:
                    raise ToolProtocolError(protocol_error)
                raise RuntimeError(f"backend POST /v1/responses failed: {response.status} {text}")

            # llama.cpp can remain wire-silent while it builds one structured
            # tool call. DeepSeek regularly needs more than 90 seconds here at
            # long context even though the supervised process is healthy and
            # decoding. Do not cancel that request: the websocket handler sends
            # Codex independent response.in_progress keepalives, and an actual
            # backend exit closes this stream immediately.
            async for event in self._iter_sse_json(response):
                event_type = event.get("type")
                if not isinstance(event_type, str):
                    continue

                if event_type in {"response.created", "response.in_progress"}:
                    continue

                item = event.get("item")
                item_id = ""
                if isinstance(item, dict):
                    item_id = str(item.get("id") or "")
                event_item_id = event.get("item_id")
                if isinstance(event_item_id, str) and event_item_id:
                    item_id = event_item_id

                if event_type == "response.output_item.added":
                    if isinstance(item, dict) and _is_assistant_message_item(item) and item_id:
                        pending_message_added[item_id] = copy.deepcopy(event)
                        pending_message_text[item_id] = _assistant_message_text(item)
                        if not _is_ellipsis_filler_text(pending_message_text[item_id]):
                            await flush_pending_message_stream(item_id)
                        continue

                    if isinstance(item, dict):
                        await flush_pending_message("commentary")

                    if isinstance(item, dict) and _is_apply_patch_function_call(item):
                        converted_item = _apply_patch_function_to_custom_call(item)
                        call_id = str(converted_item.get("call_id") or "")
                        for key in _stream_keys_for_item(item_id, converted_item):
                            apply_patch_function_items[key] = call_id
                        converted_event = copy.deepcopy(event)
                        converted_event["item"] = converted_item
                        pending_apply_patch_added[call_id] = converted_event
                        apply_patch_argument_buffers[call_id] = ""
                        continue

                    if isinstance(item, dict) and (
                        is_web_search_function_call(item)
                        or is_web_fetch_function_call(item)
                        or is_web_browse_function_call(item)
                    ):
                        if item_id:
                            suppressed_item_ids.add(item_id)
                        continue
                    if event_type in CODEX_STREAM_EVENT_TYPES:
                        await send_event(event)
                    continue

                if event_type == "response.output_text.delta" and item_id in pending_message_added:
                    pending_message_deltas.setdefault(item_id, []).append(copy.deepcopy(event))
                    delta = event.get("delta")
                    if isinstance(delta, str):
                        pending_message_text[item_id] = pending_message_text.get(item_id, "") + delta
                    if not _is_ellipsis_filler_text(pending_message_text.get(item_id, "")):
                        await flush_pending_message_stream(item_id)
                    continue

                if event_type == "response.function_call_arguments.delta":
                    event_call_id = event.get("call_id")
                    call_id = apply_patch_function_items.get(item_id, "")
                    if not call_id and isinstance(event_call_id, str):
                        call_id = apply_patch_function_items.get(event_call_id, "")
                    if call_id:
                        delta = event.get("delta")
                        if isinstance(delta, str):
                            buffered = apply_patch_argument_buffers.get(call_id, "") + delta
                            apply_patch_argument_buffers[call_id] = buffered
                            error = _partial_tool_argument_error(
                                buffered,
                                tool_argument_limit,
                            )
                            if error:
                                raise ToolProtocolError(error)
                        continue

                if event_type == "response.output_item.done":
                    if isinstance(item, dict):
                        if _is_apply_patch_function_call(item):
                            protocol_error = _apply_patch_protocol_error(
                                item,
                                tool_argument_limit,
                            )
                            if protocol_error:
                                raise ToolProtocolError(protocol_error)
                            converted_item = self.sanitize_output_item(
                                _apply_patch_function_to_custom_call(item)
                            )
                            call_id = str(converted_item.get("call_id") or "")
                            added_event = pending_apply_patch_added.pop(call_id, None)
                            if added_event is not None:
                                await send_event(added_event)
                            patch_input = converted_item.get("input")
                            if isinstance(patch_input, str) and patch_input:
                                await send_event(
                                    {
                                        "type": "response.custom_tool_call_input.delta",
                                        "item_id": item_id or call_id,
                                        "call_id": call_id,
                                        "delta": patch_input,
                                    }
                                )
                            stored_item = copy.deepcopy(converted_item)
                            backend_arguments = item.get("arguments")
                            if isinstance(backend_arguments, str) and backend_arguments:
                                stored_item[_BACKEND_ARGUMENTS_KEY] = backend_arguments
                            output_items.append(stored_item)
                            converted_event = copy.deepcopy(event)
                            converted_event["item"] = converted_item
                            await send_event(converted_event)
                            for key in _stream_keys_for_item(item_id, converted_item):
                                apply_patch_function_items.pop(key, None)
                            apply_patch_argument_buffers.pop(call_id, None)
                            continue

                        sanitized = self.sanitize_output_item(item)
                        if _is_assistant_message_item(sanitized):
                            pending_message_done = copy.deepcopy(event)
                            pending_message_done["item"] = sanitized
                            continue
                        output_items.append(sanitized)
                        if collect_managed_calls([sanitized]):
                            await flush_pending_message("commentary")
                            if item_id:
                                suppressed_item_ids.add(item_id)
                            continue
                    if event_type in CODEX_STREAM_EVENT_TYPES:
                        if isinstance(item, dict) and _starts_followup_work(item):
                            await flush_pending_message("commentary")
                        await send_event(event)
                    continue

                if item_id and item_id in suppressed_item_ids:
                    continue

                if event_type == "response.completed":
                    await flush_pending_message(_completed_message_phase(output_items))
                    response_payload = event.get("response")
                    if isinstance(response_payload, dict):
                        completed_response = copy.deepcopy(response_payload)
                    continue

                if event_type == "response.failed":
                    response_payload = event.get("response")
                    message = "response.failed event received"
                    if isinstance(response_payload, dict):
                        error = response_payload.get("error")
                        if isinstance(error, dict) and isinstance(error.get("message"), str):
                            message = error["message"]
                    raise RuntimeError(message)

                if event_type in CODEX_STREAM_EVENT_TYPES:
                    await send_event(event)

        backend_response = completed_response or {}
        if output_items:
            backend_response["output"] = copy.deepcopy(output_items)
        elif not isinstance(backend_response.get("output"), list):
            backend_response["output"] = copy.deepcopy(output_items)
        if "usage" not in backend_response:
            backend_response["usage"] = copy.deepcopy(DEFAULT_USAGE)
        return backend_response

    async def _slot_action(self, profile: ModelProfile, action: str, filename: str | None = None) -> dict[str, Any]:
        payload = {"filename": filename} if filename is not None else None
        return await self._request_json(
            profile,
            "POST",
            f"/slots/{self.slot_id}?action={action}",
            payload,
            retry_connection_error=True,
        )

    async def erase_slot(self, profile: ModelProfile) -> dict[str, Any]:
        return await self._slot_action(profile, "erase")

    async def save_slot(self, profile: ModelProfile, filename: str) -> dict[str, Any]:
        return await self._slot_action(profile, "save", filename)

    async def restore_slot(self, profile: ModelProfile, filename: str) -> dict[str, Any]:
        return await self._slot_action(profile, "restore", filename)

    def _slot_save_dir(self, profile: ModelProfile) -> Path:
        return self.slot_save_root / profile.alias

    @staticmethod
    def _snapshot_ready(path: Path) -> bool:
        try:
            return path.is_file() and path.stat().st_size > 0
        except OSError:
            return False

    def _prune_starter_cache_sync(
        self,
        profile: ModelProfile,
        protected_filename: str,
    ) -> dict[str, Any]:
        slot_dir = self._slot_save_dir(profile)
        prefix = f"starter__{profile.slug}__"
        snapshots: list[tuple[Path, int, float]] = []
        for path in slot_dir.glob(f"{prefix}*.bin"):
            try:
                stat = path.stat()
            except OSError:
                continue
            if path.is_file():
                snapshots.append((path, stat.st_size, stat.st_mtime))

        snapshots.sort(key=lambda item: (item[2], item[0].name), reverse=True)
        kept_count = 0
        kept_bytes = 0
        deleted: list[str] = []
        for path, size, _mtime in snapshots:
            if size <= 0:
                try:
                    path.unlink()
                except OSError:
                    continue
                deleted.append(path.name)
                continue

            protected = path.name == protected_filename
            within_count = kept_count < self.starter_cache_max_count
            within_bytes = (
                self.starter_cache_max_bytes <= 0
                or kept_bytes + size <= self.starter_cache_max_bytes
            )
            if protected or (within_count and within_bytes):
                kept_count += 1
                kept_bytes += size
                continue

            try:
                path.unlink()
            except OSError:
                continue
            deleted.append(path.name)

        return {
            "deleted": deleted,
            "deleted_count": len(deleted),
            "kept_count": kept_count,
            "kept_bytes": kept_bytes,
        }

    async def prepare_starter_cache(
        self,
        profile: ModelProfile,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        """Restore or create the persistent system-and-tools prompt prefix."""

        if not self.starter_cache_enabled or not profile.supports_slots:
            return {
                "mode": "starter-cache-disabled",
                "status": "skipped",
            }

        scaffold_body = _starter_scaffold_chat_body(request)
        fingerprint = _starter_cache_fingerprint(
            profile,
            self.backend_cache_id,
            scaffold_body,
        )
        filename = f"starter__{profile.slug}__{fingerprint}.bin"
        slot_dir = self._slot_save_dir(profile)
        slot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = slot_dir / filename

        restore_error: str | None = None
        if self._snapshot_ready(snapshot_path):
            try:
                restored = await self.restore_slot(profile, filename)
                snapshot_path.touch()
                return {
                    "mode": "restore-starter-cache",
                    "status": "restored",
                    "fingerprint": fingerprint,
                    "snapshot_filename": filename,
                    "restore_result": restored,
                }
            except Exception as exc:
                restore_error = str(exc)
                try:
                    snapshot_path.unlink()
                except OSError:
                    pass

        try:
            erased = await self.erase_slot(profile)
            rendered = await self._request_json(
                profile,
                "POST",
                "/apply-template",
                scaffold_body,
                retry_connection_error=True,
            )
            prompt = rendered.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                raise RuntimeError("llama.cpp returned an empty starter prompt")
            prefilled = await self._request_json(
                profile,
                "POST",
                "/completion",
                {
                    "prompt": prompt,
                    "n_predict": 0,
                    "id_slot": self.slot_id,
                    "cache_prompt": True,
                },
                retry_connection_error=True,
            )
            saved = await self.save_slot(profile, filename)
            if not self._snapshot_ready(snapshot_path):
                raise RuntimeError("llama.cpp produced an empty starter snapshot")
            pruned = await asyncio.to_thread(
                self._prune_starter_cache_sync,
                profile,
                filename,
            )
            return {
                "mode": "build-starter-cache",
                "status": "built",
                "fingerprint": fingerprint,
                "snapshot_filename": filename,
                "restore_error": restore_error,
                "erase_result": erased,
                "prefill_timings": prefilled.get("timings"),
                "save_result": saved,
                "prune_result": pruned,
            }
        except Exception as exc:
            try:
                erased = await self.erase_slot(profile)
            except Exception:
                erased = None
            return {
                "mode": "starter-cache-fallback",
                "status": "error",
                "fingerprint": fingerprint,
                "snapshot_filename": filename,
                "restore_error": restore_error,
                "error": str(exc),
                "erase_result": erased,
            }

    def _delete_slot_snapshots_sync(self, profile: ModelProfile) -> dict[str, Any]:
        slot_dir = self._slot_save_dir(profile)
        deleted: list[str] = []
        deleted_bytes = 0
        if not slot_dir.is_dir():
            return {
                "enabled": True,
                "deleted": deleted,
                "deleted_count": 0,
                "deleted_bytes": 0,
                "slot_dir": str(slot_dir),
            }

        prefix = f"{profile.slug}__"
        for path in slot_dir.glob(f"{prefix}*.bin"):
            try:
                if not path.is_file():
                    continue
                size = path.stat().st_size
                path.unlink()
            except OSError:
                continue
            deleted.append(path.name)
            deleted_bytes += size

        return {
            "enabled": True,
            "deleted": deleted,
            "deleted_count": len(deleted),
            "deleted_bytes": deleted_bytes,
            "slot_dir": str(slot_dir),
        }

    async def clean_startup_slot_snapshots(self) -> list[dict[str, Any]]:
        if not self.slot_snapshot_clean_startup:
            return []
        results: list[dict[str, Any]] = []
        for profile in self.available_profiles.values():
            if not profile.supports_slots:
                continue
            result = await asyncio.to_thread(self._delete_slot_snapshots_sync, profile)
            if result.get("deleted_count"):
                results.append({"profile_slug": profile.slug, **result})
        return results

    def _prune_slot_snapshots_sync(
        self,
        profile: ModelProfile,
        protected_filename: str | None,
    ) -> dict[str, Any]:
        max_count = self.slot_snapshot_max_count
        max_bytes = self.slot_snapshot_max_bytes
        if max_count <= 0 and max_bytes <= 0:
            return {
                "enabled": False,
                "deleted": [],
                "deleted_bytes": 0,
                "kept_count": 0,
                "kept_bytes": 0,
            }

        slot_dir = self._slot_save_dir(profile)
        if not slot_dir.is_dir():
            return {
                "enabled": True,
                "deleted": [],
                "deleted_bytes": 0,
                "kept_count": 0,
                "kept_bytes": 0,
            }

        prefix = f"{profile.slug}__"
        snapshots: list[dict[str, Any]] = []
        for path in slot_dir.glob(f"{prefix}*.bin"):
            try:
                stat = path.stat()
            except OSError:
                continue
            if not path.is_file():
                continue
            snapshots.append(
                {
                    "path": path,
                    "name": path.name,
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                }
            )

        snapshots.sort(key=lambda item: (item["mtime"], item["name"]), reverse=True)
        kept_count = 0
        kept_bytes = 0
        deleted: list[str] = []
        deleted_bytes = 0

        for snapshot in snapshots:
            name = str(snapshot["name"])
            size = int(snapshot["size"])
            path = snapshot["path"]
            if size <= 0:
                try:
                    path.unlink()
                except OSError:
                    continue
                deleted.append(name)
                continue

            if protected_filename and name == protected_filename:
                kept_count += 1
                kept_bytes += size
                continue

            within_count = max_count <= 0 or kept_count < max_count
            within_bytes = max_bytes <= 0 or kept_bytes + size <= max_bytes
            if within_count and within_bytes:
                kept_count += 1
                kept_bytes += size
                continue

            try:
                path.unlink()
            except OSError:
                continue
            deleted.append(name)
            deleted_bytes += size

        return {
            "enabled": True,
            "max_count": max_count,
            "max_bytes": max_bytes,
            "deleted": deleted,
            "deleted_count": len(deleted),
            "deleted_bytes": deleted_bytes,
            "kept_count": kept_count,
            "kept_bytes": kept_bytes,
            "slot_dir": str(slot_dir),
        }

    async def prune_slot_snapshots(
        self,
        profile: ModelProfile,
        protected_filename: str | None = None,
    ) -> dict[str, Any]:
        result = await asyncio.to_thread(
            self._prune_slot_snapshots_sync,
            profile,
            protected_filename,
        )
        deleted = result.get("deleted")
        if isinstance(deleted, list) and deleted:
            deleted_names = {str(name) for name in deleted}
            async with self.lineage_lock:
                for snapshot in self.lineage.values():
                    if (
                        snapshot.profile_slug == profile.slug
                        and snapshot.snapshot_filename in deleted_names
                    ):
                        snapshot.snapshot_filename = ""
        return result

    async def backend_health(self, profile: ModelProfile | None = None) -> dict[str, Any]:
        target_profile = profile or self.resolve_model(self.current_model or self.default_model)
        if target_profile.supervised:
            return {
                "status": "ok" if self._profile_ready(target_profile) else "error",
                "supervised": True,
            }
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

    def _log_slot_cleanup(self, results: list[dict[str, Any]]) -> None:
        if not results:
            return
        deleted_count = sum(int(result.get("deleted_count") or 0) for result in results)
        deleted_bytes = sum(int(result.get("deleted_bytes") or 0) for result in results)
        try:
            with self.log_dir.joinpath("codex_local_router.log").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(
                    "slot snapshot startup cleanup deleted "
                    f"{deleted_count} files / {deleted_bytes} bytes\n"
                )
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

    async def _execute_managed_call_cached(
        self,
        item: dict[str, Any],
        fallback_index: int,
        scope: str,
    ) -> dict[str, Any]:
        """Execute each exact web action once per user turn, even after reconnect."""

        signature = _managed_call_signature(item)
        key = (scope, signature)
        cache = getattr(self, "web_tool_cache", None)
        if cache is None:
            cache = OrderedDict()
            self.web_tool_cache = cache
        cached = cache.get(key)
        call_id = synthesize_call_id(item, fallback_index)
        if cached is not None:
            cache.move_to_end(key)
            result = copy.deepcopy(cached)
            result["call_id"] = call_id
            telemetry = getattr(self, "telemetry", None)
            if telemetry is not None:
                telemetry.emit(
                    "router.web_tool.cache_hit",
                    {
                        "tool": _managed_call_name(item),
                        "scope": scope[:16],
                        "signature": signature[:16],
                    },
                )
            return result

        result = await self._execute_managed_call(item, fallback_index)
        stored = copy.deepcopy(result)
        stored.pop("call_id", None)
        cache[key] = stored
        cache.move_to_end(key)
        max_entries = max(
            1,
            int(getattr(self, "web_tool_cache_max_entries", DEFAULT_WEB_TOOL_CACHE_MAX_ENTRIES)),
        )
        while len(cache) > max_entries:
            cache.popitem(last=False)
        telemetry = getattr(self, "telemetry", None)
        if telemetry is not None:
            telemetry.emit(
                "router.web_tool.executed",
                {
                    "tool": _managed_call_name(item),
                    "scope": scope[:16],
                    "signature": signature[:16],
                },
            )
        result["call_id"] = call_id
        return result

    async def _run_responses_loop(
        self,
        *,
        profile: ModelProfile,
        forward_request: dict[str, Any],
        web_search_enabled: bool,
        event_sink: StreamEventSink | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
        """Run the llama.cpp /v1/responses tool-call loop until the model stops calling web_search.

        Returns the final backend response, the cumulative output items (with the
        real ``function_call`` and ``function_call_output`` items kept intact for
        lineage), and the number of web_search iterations executed.
        """

        request = copy.deepcopy(forward_request)
        web_scope = _web_turn_scope(profile, forward_request)
        max_iters = max(1, self.web_search_settings.max_iterations)
        cumulative_items: list[dict[str, Any]] = []
        request_suffix: list[dict[str, Any]] = []
        seen_signatures: set[str] = set()
        finalizing = False
        last_response: dict[str, Any] = {}
        iterations = 0
        stalled_recoveries = 0
        tool_protocol_recoveries = 0
        max_stalled_recoveries = max(
            0,
            _env_int(
                "MARATHON_STALLED_RESPONSE_RECOVERIES",
                DEFAULT_STALLED_RESPONSE_RECOVERIES,
            ),
        )
        max_tool_protocol_recoveries = max(
            0,
            _env_int(
                "MARATHON_TOOL_PROTOCOL_RECOVERIES",
                DEFAULT_TOOL_PROTOCOL_RECOVERIES,
            ),
        )

        def persist_progress(
            *,
            completed_response: dict[str, Any] | None = None,
        ) -> None:
            if not web_search_enabled:
                return
            self.save_web_turn_progress(
                web_scope,
                ManagedWebTurnProgress(
                    request_suffix=copy.deepcopy(request_suffix),
                    cumulative_items=copy.deepcopy(cumulative_items),
                    iterations=iterations,
                    seen_signatures=set(seen_signatures),
                    finalizing=finalizing,
                    completed_response=copy.deepcopy(completed_response),
                    updated_at=time.time(),
                ),
            )

        if web_search_enabled:
            progress = self.load_web_turn_progress(web_scope)
            if progress is not None:
                request_suffix = copy.deepcopy(progress.request_suffix)
                cumulative_items = copy.deepcopy(progress.cumulative_items)
                iterations = progress.iterations
                seen_signatures = set(progress.seen_signatures)
                finalizing = progress.finalizing
                request["input"] = list(request.get("input") or []) + copy.deepcopy(
                    request_suffix
                )
                if finalizing:
                    request["tools"] = _strip_managed_web_tools(request.get("tools"))
                self.telemetry.emit(
                    "router.web_turn.resumed",
                    {
                        "scope": web_scope[:16],
                        "iterations": iterations,
                        "restored_items": len(request_suffix),
                        "finalizing": finalizing,
                        "completed": progress.completed_response is not None,
                    },
                )
                if progress.completed_response is not None:
                    replayed = copy.deepcopy(progress.completed_response)
                    replayed[_WEB_REPLAYED_COMPLETION_KEY] = True
                    self.telemetry.emit(
                        "router.web_turn.completion_replayed",
                        {
                            "scope": web_scope[:16],
                            "iterations": iterations,
                            "output_items": len(cumulative_items),
                        },
                    )
                    return replayed, cumulative_items, iterations

        for _attempt in range(
            max_iters + max_stalled_recoveries + max_tool_protocol_recoveries + 2
        ):
            attempt_output_limit = int(
                request.get("max_output_tokens") or _max_output_tokens(profile)
            )
            try:
                if event_sink is None:
                    request = copy.deepcopy(request)
                    request["stream"] = False
                    response = await self._request_json(
                        profile, "POST", "/v1/responses", request
                    )
                else:
                    response = await self._request_responses_stream(
                        profile,
                        request,
                        event_sink=event_sink,
                    )
                protocol_error = _response_tool_protocol_error(
                    response,
                    _tool_argument_max_chars(),
                )
                if protocol_error:
                    raise ToolProtocolError(protocol_error)
            except ToolProtocolError as error:
                if (
                    tool_protocol_recoveries >= max_tool_protocol_recoveries
                    or not request.get("tools")
                ):
                    raise
                tool_protocol_recoveries += 1
                reason = str(error)
                self.telemetry.emit(
                    "router.response.tool_protocol_recovery",
                    {
                        "attempt": tool_protocol_recoveries,
                        "reason": reason,
                        "available_tools": len(request.get("tools") or []),
                    },
                    level="warning",
                )
                request = copy.deepcopy(request)
                recovery_items = [_tool_protocol_recovery_message(reason)]
                request["input"] = list(request.get("input") or []) + recovery_items
                request_suffix.extend(copy.deepcopy(recovery_items))
                request["tool_choice"] = "required"
                request["max_output_tokens"] = min(attempt_output_limit, 4_096)
                persist_progress()
                continue
            last_response = response
            iter_items: list[dict[str, Any]] = []
            for item in response.get("output", []):
                if isinstance(item, dict):
                    iter_items.append(self.sanitize_output_item(item))
            pending_calls = collect_managed_calls(iter_items)
            _annotate_message_phases(iter_items, final_response=not pending_calls)
            iter_items = [
                item for item in iter_items if not _is_droppable_commentary_message(item)
            ]
            pending_calls = collect_managed_calls(iter_items)

            cumulative_items.extend(iter_items)

            if (
                _response_stalled_at_output_limit(
                    response,
                    iter_items,
                    attempt_output_limit,
                )
                and stalled_recoveries < max_stalled_recoveries
                and bool(request.get("tools"))
            ):
                stalled_recoveries += 1
                self.telemetry.emit(
                    "router.response.stalled_recovery",
                    {
                        "attempt": stalled_recoveries,
                        "output_tokens": attempt_output_limit,
                        "available_tools": len(request.get("tools") or []),
                    },
                    level="warning",
                )
                request = copy.deepcopy(request)
                recovery_items = [_stalled_recovery_message()]
                request["input"] = list(request.get("input") or []) + recovery_items
                request_suffix.extend(copy.deepcopy(recovery_items))
                request["tool_choice"] = "required"
                persist_progress()
                continue

            if not web_search_enabled:
                break

            if not pending_calls:
                break

            pending_signatures = {
                _managed_call_signature(call) for call in pending_calls
            }
            repeated_only = bool(pending_signatures) and pending_signatures.issubset(
                seen_signatures
            )
            if iterations >= max_iters or repeated_only:
                # A reconnect must never restart an already-completed web
                # action indefinitely. Feed a valid output for the current
                # call id, then remove managed tools and force a final answer.
                if repeated_only:
                    tool_outputs = []
                    for idx, call in enumerate(pending_calls):
                        output = await self._execute_managed_call_cached(
                            call, idx, web_scope
                        )
                        text = str(output.get("output") or "")
                        output["output"] = (
                            text
                            + "\n\n[Marathon: this exact web action already completed "
                            "during this user turn. Use the existing result and finish "
                            "without requesting it again.]"
                        )
                        tool_outputs.append(output)
                    iterations += 1
                    self.telemetry.emit(
                        "router.web_tool.repeat_guard",
                        {
                            "scope": web_scope[:16],
                            "iterations": iterations,
                            "repeated_calls": len(pending_calls),
                        },
                        level="warning",
                    )
                else:
                    tool_outputs = [
                        make_function_call_output(
                            synthesize_call_id(call, idx),
                            "Tool-call iteration cap reached; please answer from "
                            "the results you already have.",
                        )
                        for idx, call in enumerate(pending_calls)
                    ]
                    iterations = max_iters
                seen_signatures.update(pending_signatures)
                cumulative_items.extend(tool_outputs)
                request = copy.deepcopy(request)
                appended_items = copy.deepcopy(iter_items) + copy.deepcopy(tool_outputs)
                request["input"] = list(request.get("input") or []) + appended_items
                request_suffix.extend(copy.deepcopy(appended_items))
                request["tools"] = _strip_managed_web_tools(request.get("tools"))
                finalizing = True
                persist_progress()
                if event_sink is None:
                    request["stream"] = False
                    final = await self._request_json(profile, "POST", "/v1/responses", request)
                else:
                    final = await self._request_responses_stream(
                        profile,
                        request,
                        event_sink=event_sink,
                    )
                last_response = final
                final_items: list[dict[str, Any]] = []
                for item in final.get("output", []):
                    if isinstance(item, dict):
                        final_items.append(self.sanitize_output_item(item))
                _annotate_message_phases(final_items, final_response=True)
                final_items = [
                    item for item in final_items if not _is_droppable_commentary_message(item)
                ]
                cumulative_items.extend(final_items)
                break

            tool_outputs = []
            for idx, call in enumerate(pending_calls):
                tool_outputs.append(
                    await self._execute_managed_call_cached(call, idx, web_scope)
                )
            cumulative_items.extend(tool_outputs)
            seen_signatures.update(pending_signatures)
            request = copy.deepcopy(request)
            appended_items = copy.deepcopy(iter_items) + copy.deepcopy(tool_outputs)
            request["input"] = list(request.get("input") or []) + appended_items
            request_suffix.extend(copy.deepcopy(appended_items))
            thinking_budget = _tool_thinking_budget_for_turn(request, tool_outputs)
            if thinking_budget is not None:
                request["thinking_budget_tokens"] = thinking_budget
            iterations += 1
            # Persist before touching the client stream. If that socket has
            # already gone away, the reconnect resumes after this tool result
            # instead of reconstructing the turn from its original prompt.
            persist_progress()
            if event_sink is not None:
                for item in externalize_for_codex(copy.deepcopy(pending_calls)):
                    if not await event_sink({"type": "response.output_item.done", "item": item}):
                        raise ConnectionError("websocket client disconnected")

        last_response = copy.deepcopy(last_response)
        last_response["usage"] = self.usage_payload(last_response.get("usage"))
        persist_progress(completed_response=last_response)
        return last_response, cumulative_items, iterations

    async def process_websocket_create(
        self,
        payload: dict[str, Any],
        *,
        preset_response_id: str | None = None,
        event_sink: StreamEventSink | None = None,
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
        if profile.temperature is not None:
            request["temperature"] = profile.temperature
        request = normalize_responses_request(request, profile)
        request["model"] = profile.alias

        base_instructions_hash = str(
            request.pop("_marathon_instruction_base_hash", "") or ""
        )
        lifted_instruction_count = int(
            request.pop("_marathon_lifted_instruction_count", 0) or 0
        )
        tool_output_truncations = int(
            request.pop("_marathon_tool_output_truncations", 0) or 0
        )
        malformed_tool_replay_drops = int(
            request.pop("_marathon_malformed_tool_replay_drops", 0) or 0
        )
        current_instructions = request.get("instructions")
        current_instructions_text = (
            current_instructions if isinstance(current_instructions, str) else ""
        )
        # Responses instructions are turn-scoped upstream. Locally, retain
        # developer/system messages lifted on an earlier delta so the effective
        # scaffold does not disappear on the next continuation.
        request["instructions"] = _effective_instructions_for_request(
            parent_snapshot,
            current_instructions_text,
            base_instructions_hash,
            lifted_instruction_count,
        )
        if tool_output_truncations:
            self.telemetry.emit(
                "router.tool_output.truncated",
                {
                    "count": tool_output_truncations,
                    "limit_chars": max(
                        1,
                        _env_int(
                            "MARATHON_TOOL_OUTPUT_MAX_CHARS",
                            DEFAULT_TOOL_OUTPUT_MAX_CHARS,
                        ),
                    ),
                },
            )
        if malformed_tool_replay_drops:
            self.telemetry.emit(
                "router.tool_history.sanitized",
                {
                    "dropped_items": malformed_tool_replay_drops,
                    "transport": "websocket",
                },
                level="warning",
            )

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
        prompt_cache_key = str(request.get("prompt_cache_key") or "")

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
            and profile.supports_slots
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
        if profile.supports_slots:
            forward_request["id_slot"] = self.slot_id
            forward_request["cache_prompt"] = True
        else:
            forward_request.pop("id_slot", None)
            forward_request.pop("cache_prompt", None)
        forward_request["stream"] = False
        output_limit = _max_output_tokens(profile)
        requested_output_limit = forward_request.get("max_output_tokens")
        if isinstance(requested_output_limit, int) and requested_output_limit > 0:
            output_limit = min(output_limit, requested_output_limit)
        forward_request["max_output_tokens"] = output_limit
        thinking_budget = _tool_thinking_budget_for_turn(forward_request, delta_input)
        if thinking_budget is not None:
            forward_request["thinking_budget_tokens"] = thinking_budget

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
                "tool_thinking_budget_tokens": thinking_budget,
            },
        )
        self.telemetry.emit(
            "router.thinking_budget.selected",
            {
                "mode": "native-cap" if thinking_budget is not None else "unrestricted",
                "thinking_budget_tokens": thinking_budget,
                "max_output_tokens": output_limit,
                "delta_input_items": len(delta_input),
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
                    instructions_text=instructions_text,
                    base_instructions_hash=base_instructions_hash,
                    instructions_hash=instructions_hash,
                    tools_hash=tools_hash,
                    prompt_cache_key=prompt_cache_key,
                    created_at=time.time(),
                )
                self.last_response_by_model[profile.slug] = response_id

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
                self.telemetry.emit("router.response.completed", trace_entry)
                if self.debug:
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
                "streamed": False,
            }

        web_search_enabled = bool(
            forward_request.pop("_marathon_web_search_enabled", False)
        ) and self.web_search is not None

        async with self.backend_lock:
            restore_result: dict[str, Any] | None = None
            erase_result: dict[str, Any] | None = None
            restore_error: str | None = None
            starter_cache_result: dict[str, Any] | None = None
            slot_prepare_mode = "erase-root"
            slot_prepare_start = time.perf_counter()
            live_parent = (
                parent_snapshot is not None
                and previous_response_id is not None
                and self.live_slot_by_model.get(profile.slug) == previous_response_id
            )
            starter_root = parent_snapshot is None or _is_warmup_root(parent_snapshot)
            live_reconnect_root = (
                profile.supports_slots
                and parent_snapshot is None
                and _can_reuse_reconnect_root(
                    profile.slug,
                    prompt_cache_key,
                    self.live_slot_by_model,
                    self.live_prompt_cache_key_by_model,
                )
            )
            if not profile.supports_slots:
                slot_prepare_mode = "backend-prefix-cache"
                restore_result = {
                    "status": "skipped",
                    "reason": "backend replays full lineage and manages its own prefix cache",
                }
                self.live_slot_by_model.pop(profile.slug, None)
                self.live_prompt_cache_key_by_model.pop(profile.slug, None)
            elif live_reconnect_root:
                slot_prepare_mode = "reuse-live-reconnect-root"
                restore_result = {
                    "status": "skipped",
                    "reason": "same prompt cache key; llama.cpp will prefix-match full prompt",
                }
            elif starter_root:
                if profile.slug in self.live_slot_by_model:
                    slot_prepare_mode = _root_prompt_cache_mode(
                        profile.slug,
                        prompt_cache_key,
                        self.live_slot_by_model,
                        self.live_prompt_cache_key_by_model,
                    )
                    # llama.cpp compares the complete token stream before
                    # reusing live KV state, so changed scaffolds naturally
                    # invalidate only the mismatched suffix.
                    restore_result = {
                        "status": "skipped",
                        "reason": (
                            "new conversation; llama.cpp will reuse only the "
                            "token-exact prompt prefix"
                        ),
                    }
                else:
                    starter_cache_result = await self.prepare_starter_cache(
                        profile,
                        forward_request,
                    )
                    slot_prepare_mode = str(starter_cache_result["mode"])
                    action_result = starter_cache_result.get("restore_result")
                    restore_result = (
                        action_result
                        if isinstance(action_result, dict)
                        else starter_cache_result
                    )
            elif not scaffold_matches:
                slot_prepare_mode = "erase-scaffold-mismatch"
                erase_result = await self.erase_slot(profile)
                self.live_slot_by_model.pop(profile.slug, None)
            elif live_parent:
                slot_prepare_mode = "reuse-live-parent"
                restore_result = {"status": "skipped", "reason": "live slot already matches parent"}
            elif not parent_snapshot.snapshot_filename:
                slot_prepare_mode = "erase-parent-no-snapshot"
                erase_result = await self.erase_slot(profile)
                self.live_slot_by_model.pop(profile.slug, None)
            else:
                slot_prepare_mode = "restore-parent"
                try:
                    restore_result = await self.restore_slot(profile, parent_snapshot.snapshot_filename)
                    self.live_slot_by_model[profile.slug] = previous_response_id
                except Exception as exc:
                    restore_error = str(exc)
                    slot_prepare_mode = "erase-restore-error"
                    erase_result = await self.erase_slot(profile)
                    self.live_slot_by_model.pop(profile.slug, None)
            slot_prepare_ms = (time.perf_counter() - slot_prepare_start) * 1000.0

            backend_start = time.perf_counter()
            backend_response, all_output_items, web_search_iterations = await self._run_responses_loop(
                profile=profile,
                forward_request=forward_request,
                web_search_enabled=web_search_enabled,
                event_sink=event_sink,
            )
            replayed_web_completion = bool(
                backend_response.pop(_WEB_REPLAYED_COMPLETION_KEY, False)
            )
            backend_ms = (time.perf_counter() - backend_start) * 1000.0

            response_id = preset_response_id or str(
                backend_response.get("id") or self.mint_response_id("resp")
            )
            snapshot_filename = ""
            snapshot_path: Path | None = None
            pre_prune_result: dict[str, Any] | None = None
            save_result: dict[str, Any] | None = None
            save_error: str | None = None
            snapshot_saved = False
            save_start = time.perf_counter()
            if self.slot_snapshots_enabled and profile.supports_slots:
                snapshot_filename = f"{profile.slug}__{response_id}.bin"
                snapshot_path = self._slot_save_dir(profile) / snapshot_filename
                pre_prune_result = await self.prune_slot_snapshots(profile)
                try:
                    save_result = await self.save_slot(profile, snapshot_filename)
                except Exception as exc:
                    save_error = str(exc)
                if save_error is None:
                    try:
                        snapshot_saved = (
                            snapshot_path.is_file() and snapshot_path.stat().st_size > 0
                        )
                    except OSError:
                        snapshot_saved = False
                    if not snapshot_saved:
                        save_error = "slot save produced an empty or missing snapshot"
            else:
                reason = (
                    "snapshots disabled"
                    if profile.supports_slots
                    else "backend has no llama.cpp slot API"
                )
                save_result = {"status": "skipped", "reason": reason}
            slot_save_ms = (time.perf_counter() - save_start) * 1000.0
            post_prune_result: dict[str, Any] | None = None
            if self.slot_snapshots_enabled and profile.supports_slots:
                post_prune_result = await self.prune_slot_snapshots(
                    profile,
                    protected_filename=snapshot_filename if snapshot_saved else None,
                )
            if profile.supports_slots:
                self.live_slot_by_model[profile.slug] = response_id
            if profile.supports_slots and prompt_cache_key:
                self.live_prompt_cache_key_by_model[profile.slug] = prompt_cache_key

        output_items = all_output_items

        usage_payload = self.usage_payload(backend_response.get("usage"))

        conversation_items = full_input + [
            _backend_lineage_item(item)
            for item in output_items
            if isinstance(item, dict)
        ]
        async with self.lineage_lock:
            self.lineage[response_id] = ResponseSnapshot(
                response_id=response_id,
                profile_slug=profile.slug,
                conversation_items=conversation_items,
                snapshot_filename=snapshot_filename if snapshot_saved else "",
                instructions_text=instructions_text,
                base_instructions_hash=base_instructions_hash,
                instructions_hash=instructions_hash,
                tools_hash=tools_hash,
                prompt_cache_key=prompt_cache_key,
                created_at=time.time(),
            )
            self.last_response_by_model[profile.slug] = response_id

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
                    "starter_cache": starter_cache_result,
                    "save_ms": slot_save_ms,
                    "erase_result": erase_result,
                    "restore_result": restore_result,
                    "restore_error": restore_error,
                    "save_result": save_result,
                    "save_error": save_error,
                    "pre_prune_result": pre_prune_result,
                    "post_prune_result": post_prune_result,
                    "snapshot_filename": snapshot_filename,
                    "snapshot_saved": snapshot_saved,
                },
                "backend": {
                    "usage": usage_payload,
                    "timings": backend_response.get("timings"),
                    "latency_ms": backend_ms,
                },
                "output": {
                    "item_types": dict(Counter(
                        str(item.get("type") or "unknown")
                        for item in output_items if isinstance(item, dict)
                    )),
                    "tool_calls": dict(Counter(
                        str(item.get("name") or item.get("type") or "unknown")
                        for item in output_items
                        if isinstance(item, dict) and item.get("type") in {
                            "function_call", "custom_tool_call", "local_shell_call",
                            "web_search_call", "tool_search_call",
                        }
                    )),
                    "web_search_iterations": web_search_iterations,
                },
            }
            self.telemetry.emit("router.response.completed", trace_entry)
            if self.debug:
                try:
                    with self.trace_log_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(trace_entry, sort_keys=True) + "\n")
                except Exception:
                    pass

        codex_output_items = (
            externalize_for_codex(copy.deepcopy(output_items))
            if web_search_enabled
            else copy.deepcopy(output_items)
        )
        for item in codex_output_items:
            if isinstance(item, dict):
                item.pop(_BACKEND_ARGUMENTS_KEY, None)

        return {
            "response_id": response_id,
            "usage": usage_payload,
            "output_items": codex_output_items,
            "backend_response": backend_response,
            # A completed turn replayed after reconnect emitted no fresh
            # backend SSE events, so the WS handler must send its stored items.
            "streamed": event_sink is not None and not replayed_web_completion,
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
            "slot_snapshot_retention": {
                "max_count": state.slot_snapshot_max_count,
                "max_bytes": state.slot_snapshot_max_bytes,
                "clean_startup": state.slot_snapshot_clean_startup,
            },
            "starter_cache": {
                "enabled": state.starter_cache_enabled,
                "max_count": state.starter_cache_max_count,
                "max_bytes": state.starter_cache_max_bytes,
            },
            "known_lineage": len(state.lineage),
            "live_slots": dict(state.live_slot_by_model),
            "backend_health": backend_status,
        }
    )


async def handle_http_proxy(request: web.Request) -> web.StreamResponse:
    state: RouterState = request.app["state"]
    request_started = time.perf_counter()
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

    state.telemetry.emit(
        "router.http.started",
        {
            "method": request.method,
            "path": path,
            "body_bytes": len(raw_body),
            "requested_model": requested_model,
            "stream": data.get("stream") if isinstance(data, dict) else None,
            "message_count": len(data.get("messages", []))
            if isinstance(data, dict) and isinstance(data.get("messages"), list)
            else None,
        },
    )

    try:
        profile = await state.ensure_model_async(requested_model)
    except Exception as exc:
        state.telemetry.emit(
            "router.http.error",
            {"path": path, "phase": "model", "error": str(exc)},
            level="error",
        )
        return web.json_response({"error": {"message": str(exc)}}, status=502)

    body = raw_body
    if data is not None:
        raw_snapshot = copy.deepcopy(data)
        data["model"] = profile.alias
        if profile.temperature is not None:
            data["temperature"] = profile.temperature
        if path == "/v1/responses":
            try:
                data = normalize_responses_request(data, profile)
            except ValueError as exc:
                state.telemetry.emit(
                    "router.http.rejected",
                    {"path": path, "phase": "request", "error": str(exc)},
                    level="warning",
                )
                return web.json_response(
                    {
                        "error": {
                            "message": str(exc),
                            "type": "invalid_request_error",
                        }
                    },
                    status=400,
                )
            if data.pop("_marathon_web_search_enabled", False):
                # The managed web-tool loop is only implemented on the
                # WebSocket Responses path. Keep HTTP/SSE fallback from
                # leaking router-private fields or unmanaged web functions to
                # llama.cpp.
                data["tools"] = _strip_managed_web_tools(data.get("tools"))
            data.pop("_marathon_instruction_base_hash", None)
            data.pop("_marathon_lifted_instruction_count", None)
            tool_output_truncations = int(
                data.pop("_marathon_tool_output_truncations", 0) or 0
            )
            malformed_tool_replay_drops = int(
                data.pop("_marathon_malformed_tool_replay_drops", 0) or 0
            )
            if tool_output_truncations:
                state.telemetry.emit(
                    "router.tool_output.truncated",
                    {
                        "count": tool_output_truncations,
                        "limit_chars": max(
                            1,
                            _env_int(
                                "MARATHON_TOOL_OUTPUT_MAX_CHARS",
                                DEFAULT_TOOL_OUTPUT_MAX_CHARS,
                            ),
                        ),
                        "transport": "http",
                    },
                )
            if malformed_tool_replay_drops:
                state.telemetry.emit(
                    "router.tool_history.sanitized",
                    {
                        "dropped_items": malformed_tool_replay_drops,
                        "transport": "http",
                    },
                    level="warning",
                )
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
    response: web.StreamResponse | None = None
    try:
        async with state.http_client.request(
            request.method,
            upstream_url,
            data=body if body else None,
            headers=headers,
            allow_redirects=False,
        ) as upstream:
            first_chunk_ms: float | None = None
            response_bytes = 0
            response = web.StreamResponse(status=upstream.status)
            for key, value in upstream.headers.items():
                if key.lower() not in HOP_BY_HOP_HEADERS:
                    response.headers[key] = value
            await response.prepare(request)
            async for chunk in upstream.content.iter_chunked(8192):
                if first_chunk_ms is None:
                    first_chunk_ms = (time.perf_counter() - request_started) * 1000.0
                response_bytes += len(chunk)
                await response.write(chunk)
            await response.write_eof()
            state.telemetry.emit(
                "router.http.completed",
                {
                    "path": path,
                    "status": upstream.status,
                    "duration_ms": (time.perf_counter() - request_started) * 1000.0,
                    "first_chunk_ms": first_chunk_ms,
                    "response_bytes": response_bytes,
                },
                level="info" if upstream.status < 400 else "error",
            )
            return response
    except Exception as exc:
        if _is_client_disconnect(exc):
            state.telemetry.emit(
                "router.http.client_disconnected",
                {
                    "path": path,
                    "duration_ms": (time.perf_counter() - request_started) * 1000.0,
                    "response_bytes": response_bytes if response is not None else 0,
                    "message": str(exc),
                },
            )
            # Headers may already be on the wire, so attempting to replace the
            # response with a JSON error would only cause a second write error.
            return response if response is not None else web.Response(status=499)
        state.telemetry.emit(
            "router.http.error",
            {
                "path": path,
                "phase": "upstream",
                "duration_ms": (time.perf_counter() - request_started) * 1000.0,
                "error": str(exc),
            },
            level="error",
        )
        return web.json_response({"error": {"message": str(exc)}}, status=502)


async def handle_ws_responses(request: web.Request) -> web.StreamResponse:
    state: RouterState = request.app["state"]
    # Do not use aiohttp's server heartbeat here. During long local prefills the
    # handler is awaiting llama.cpp and not reading client pong frames, so aiohttp
    # can falsely treat a healthy client as dead after heartbeat + pong timeout.
    ws = web.WebSocketResponse(heartbeat=None, max_msg_size=32 * 1024 * 1024)
    await ws.prepare(request)
    send_lock = asyncio.Lock()

    async def safe_send(message: dict[str, Any]) -> bool:
        if ws.closed:
            return False
        try:
            async with send_lock:
                if ws.closed:
                    return False
                send = ws.send_json(message)
                if WS_SEND_TIMEOUT_SECONDS > 0:
                    await asyncio.wait_for(send, timeout=WS_SEND_TIMEOUT_SECONDS)
                else:
                    await send
            return True
        except (asyncio.TimeoutError, ConnectionResetError, ConnectionError):
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
            request_scope = _active_ws_request_scope(payload)
            disconnect_reported = False

            async def response_send(message: dict[str, Any]) -> bool:
                nonlocal disconnect_reported
                sent = await safe_send(message)
                if not sent and not disconnect_reported:
                    disconnect_reported = True
                    state.telemetry.emit(
                        "router.ws.client_disconnected",
                        {
                            "scope": _sha256_text(request_scope or "")[:16],
                            "response_id": preset_id,
                            "last_event_type": str(message.get("type") or "unknown"),
                        },
                        level="warning",
                    )
                return sent

            if not await response_send(
                {"type": "response.created", "response": {"id": preset_id}}
            ):
                # Client dropped before we could ack; skip the heavy work.
                continue

            result_task = asyncio.create_task(
                state.process_websocket_create(
                    payload,
                    preset_response_id=preset_id,
                    event_sink=response_send,
                )
            )
            previous_task = state.replace_active_ws_task(request_scope, result_task)
            if previous_task is not None:
                state.telemetry.emit(
                    "router.ws.request.superseded",
                    {
                        "scope": _sha256_text(request_scope or "")[:16],
                        "response_id": preset_id,
                    },
                    level="warning",
                )
                previous_task.cancel()
                await asyncio.gather(previous_task, return_exceptions=True)
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
                    if not await response_send(
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
                await response_send(
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

            if not result.get("streamed"):
                for item in result["output_items"]:
                    if not await response_send(
                        {"type": "response.output_item.done", "item": item}
                    ):
                        break
            await response_send(
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
    state._log_slot_cleanup(await state.clean_startup_slot_snapshots())
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
        or "qwen3.6-27b-q4-128k-single",
    )
    parser.add_argument("--state-dir", default=str(_repo_root() / ".marathon" / "state"))
    parser.add_argument("--log-dir", default=str(_repo_root() / "logs"))
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
