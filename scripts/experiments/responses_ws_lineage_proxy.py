#!/usr/bin/env python3
"""Experimental Responses websocket proxy with lineage-aware llama slot restore.

This sits between Codex-style incremental websocket requests and a local
llama.cpp `/v1/responses` backend. It implements the missing server-side piece
that OpenAI provides for Codex today:

- accept `previous_response_id`
- reconstruct the full conversation for the referenced lineage
- restore the exact llama slot snapshot for that ancestor response
- run the next turn on that restored slot
- save a fresh snapshot keyed by the new response id

The goal is to test whether server-managed lineage removes the expensive full
prompt replay seen after resume/fork-style history changes.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aiohttp import ClientSession
from aiohttp import WSMsgType
from aiohttp import web

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "scripts" / "routers"))

from codex_local_router import _base_instructions  # type: ignore  # noqa: E402
from codex_local_router import _sha256_text  # type: ignore  # noqa: E402
from codex_local_router import _stable_json  # type: ignore  # noqa: E402
from codex_local_router import normalize_responses_request  # type: ignore  # noqa: E402


DEFAULT_USAGE = {
    "input_tokens": 0,
    "input_tokens_details": None,
    "output_tokens": 0,
    "output_tokens_details": None,
    "total_tokens": 0,
}


@dataclass
class ResponseState:
    response_id: str
    conversation_items: list[dict[str, Any]]
    snapshot_filename: str
    instructions_hash: str
    tools_hash: str
    created_at: float


class ProxyState:
    def __init__(
        self,
        *,
        backend_base_url: str,
        model: str,
        display_name: str,
        description: str,
        context_window: int,
        auto_compact_token_limit: int,
        truncation_limit: int,
        slot_id: int,
        slot_save_dir: Path,
        log_dir: Path,
        debug: bool,
    ):
        self.backend_base_url = backend_base_url.rstrip("/")
        self.model = model
        self.display_name = display_name
        self.description = description
        self.context_window = context_window
        self.auto_compact_token_limit = auto_compact_token_limit
        self.truncation_limit = truncation_limit
        self.slot_id = slot_id
        self.slot_save_dir = slot_save_dir
        self.log_dir = log_dir
        self.debug = debug
        self.slot_save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.trace_log_path = self.log_dir / "responses_ws_lineage_proxy_trace.jsonl"
        self.lineage: dict[str, ResponseState] = {}
        self.last_response_id: str | None = None
        self._trace_seq = 0
        self._lock = asyncio.Lock()
        self._client: ClientSession | None = None

    async def open(self) -> None:
        self._client = ClientSession()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    def model_catalog(self) -> dict[str, Any]:
        instructions = _base_instructions()
        model_info = {
            "slug": self.model,
            "display_name": self.display_name,
            "description": self.description,
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
            "truncation_policy": {"mode": "tokens", "limit": self.truncation_limit},
            "supports_parallel_tool_calls": False,
            "supports_image_detail_original": False,
            "context_window": self.context_window,
            "max_context_window": self.context_window,
            "auto_compact_token_limit": self.auto_compact_token_limit,
            "effective_context_window_percent": 90,
            "experimental_supported_tools": [],
            "input_modalities": ["text"],
            "supports_search_tool": False,
        }
        return {
            "models": [model_info],
            "object": "list",
            "data": [
                {
                    "id": self.model,
                    "object": "model",
                    "owned_by": "marathon-ws-lineage-proxy",
                    "description": self.description,
                }
            ],
        }

    def sanitize_output_item(self, item: dict[str, Any]) -> dict[str, Any]:
        sanitized = copy.deepcopy(item)
        sanitized.pop("status", None)
        return sanitized

    def _usage(self, usage: Any) -> dict[str, Any]:
        if not isinstance(usage, dict):
            return copy.deepcopy(DEFAULT_USAGE)
        merged = copy.deepcopy(DEFAULT_USAGE)
        merged.update(usage)
        return merged

    async def _request_json(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._client is None:
            raise RuntimeError("proxy client session is not open")
        url = f"{self.backend_base_url}{path}"
        async with self._client.request(method, url, json=payload) as response:
            text = await response.text()
            if response.status >= 400:
                raise RuntimeError(f"backend {method} {path} failed: {response.status} {text}")
            try:
                return json.loads(text) if text else {}
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"backend {method} {path} returned invalid JSON: {exc}") from exc

    async def erase_slot(self) -> dict[str, Any]:
        return await self._request_json("POST", f"/slots/{self.slot_id}?action=erase")

    async def save_slot(self, filename: str) -> dict[str, Any]:
        return await self._request_json(
            "POST",
            f"/slots/{self.slot_id}?action=save",
            {"filename": filename},
        )

    async def restore_slot(self, filename: str) -> dict[str, Any]:
        return await self._request_json(
            "POST",
            f"/slots/{self.slot_id}?action=restore",
            {"filename": filename},
        )

    async def backend_health(self) -> dict[str, Any]:
        return await self._request_json("GET", "/health")

    async def backend_models(self) -> dict[str, Any]:
        return await self._request_json("GET", "/v1/models")

    async def forward_http_responses(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = copy.deepcopy(payload)
        request["model"] = self.model
        request = normalize_responses_request(request)
        request["model"] = self.model
        request["stream"] = False
        return await self._request_json("POST", "/v1/responses", request)

    async def process_create(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = copy.deepcopy(payload)
        request.pop("type", None)
        request["model"] = self.model
        request = normalize_responses_request(request)
        request["model"] = self.model
        request["stream"] = False

        delta_input = request.get("input")
        if not isinstance(delta_input, list):
            raise RuntimeError("response.create requires list input")

        previous_response_id = request.get("previous_response_id")
        if previous_response_id is not None and not isinstance(previous_response_id, str):
            raise RuntimeError("previous_response_id must be a string when provided")

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
        parent_state: ResponseState | None = None
        if previous_response_id:
            parent_state = self.lineage.get(previous_response_id)
            if parent_state is None:
                raise RuntimeError(f"unknown previous_response_id: {previous_response_id}")
            full_input = copy.deepcopy(parent_state.conversation_items) + copy.deepcopy(delta_input)
            relation = "continue" if previous_response_id == self.last_response_id else "branch"
        else:
            full_input = copy.deepcopy(delta_input)

        forward_request = copy.deepcopy(request)
        forward_request.pop("previous_response_id", None)
        forward_request["input"] = full_input
        forward_request["id_slot"] = self.slot_id
        forward_request["cache_prompt"] = True
        forward_request["stream"] = False

        async with self._lock:
            slot_op_start = time.perf_counter()
            restore_result: dict[str, Any] | None = None
            erase_result: dict[str, Any] | None = None
            if parent_state is None:
                erase_result = await self.erase_slot()
            else:
                restore_result = await self.restore_slot(parent_state.snapshot_filename)
            slot_prep_ms = (time.perf_counter() - slot_op_start) * 1000.0

            backend_start = time.perf_counter()
            backend_response = await self._request_json("POST", "/v1/responses", forward_request)
            backend_ms = (time.perf_counter() - backend_start) * 1000.0

            response_id = str(backend_response.get("id") or f"resp_{int(time.time() * 1000)}_{self._trace_seq}")
            snapshot_filename = f"{response_id}.bin"

            save_start = time.perf_counter()
            save_result = await self.save_slot(snapshot_filename)
            save_ms = (time.perf_counter() - save_start) * 1000.0

            output_items = []
            for item in backend_response.get("output", []):
                if isinstance(item, dict):
                    output_items.append(self.sanitize_output_item(item))

            conversation_items = full_input + copy.deepcopy(output_items)
            self.lineage[response_id] = ResponseState(
                response_id=response_id,
                conversation_items=conversation_items,
                snapshot_filename=snapshot_filename,
                instructions_hash=instructions_hash,
                tools_hash=tools_hash,
                created_at=time.time(),
            )
            self.last_response_id = response_id

            self._trace_seq += 1
            trace_entry = {
                "trace_id": self._trace_seq,
                "timestamp": time.time(),
                "relation": relation,
                "model": self.model,
                "slot_id": self.slot_id,
                "previous_response_id": previous_response_id,
                "response_id": response_id,
                "instructions_hash": instructions_hash,
                "tools_hash": tools_hash,
                "delta_input_items": len(delta_input),
                "full_input_items": len(full_input),
                "slot_prepare_ms": slot_prep_ms,
                "backend_ms": backend_ms,
                "slot_save_ms": save_ms,
                "erase_result": erase_result,
                "restore_result": restore_result,
                "save_result": save_result,
                "backend_usage": backend_response.get("usage"),
                "backend_timings": backend_response.get("timings"),
            }
            with self.trace_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(trace_entry, sort_keys=True) + "\n")

        return {
            "response_id": response_id,
            "usage": self._usage(backend_response.get("usage")),
            "output_items": output_items,
            "backend_response": backend_response,
        }


async def handle_models(request: web.Request) -> web.Response:
    state: ProxyState = request.app["state"]
    return web.json_response(state.model_catalog())


async def handle_health(request: web.Request) -> web.Response:
    state: ProxyState = request.app["state"]
    backend_status = None
    try:
        backend_status = await state.backend_health()
    except Exception as exc:  # pragma: no cover - debug surface
        backend_status = {"status": "error", "message": str(exc)}
    return web.json_response(
        {
            "ok": True,
            "model": state.model,
            "backend": state.backend_base_url,
            "slot_id": state.slot_id,
            "known_lineage": len(state.lineage),
            "backend_health": backend_status,
        }
    )


async def handle_http_responses(request: web.Request) -> web.Response:
    state: ProxyState = request.app["state"]
    payload = await request.json()
    response = await state.forward_http_responses(payload)
    return web.json_response(response)


async def handle_ws_responses(request: web.Request) -> web.StreamResponse:
    state: ProxyState = request.app["state"]
    ws = web.WebSocketResponse(heartbeat=30.0, max_msg_size=32 * 1024 * 1024)
    await ws.prepare(request)

    async for msg in ws:
        if msg.type == WSMsgType.TEXT:
            try:
                payload = json.loads(msg.data)
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "error": {"message": "invalid JSON"}})
                continue

            if payload.get("type") != "response.create":
                await ws.send_json(
                    {
                        "type": "error",
                        "error": {"message": f"unsupported websocket message type: {payload.get('type')}"},
                    }
                )
                continue

            try:
                result = await state.process_create(payload)
            except Exception as exc:
                await ws.send_json({"type": "error", "error": {"message": str(exc)}})
                continue

            await ws.send_json({"type": "response.created", "response": {"id": result["response_id"]}})
            for item in result["output_items"]:
                await ws.send_json({"type": "response.output_item.done", "item": item})
            await ws.send_json(
                {
                    "type": "response.completed",
                    "response": {"id": result["response_id"], "usage": result["usage"]},
                }
            )
        elif msg.type == WSMsgType.ERROR:
            break

    return ws


async def on_startup(app: web.Application) -> None:
    await app["state"].open()


async def on_cleanup(app: web.Application) -> None:
    await app["state"].close()


def build_app(state: ProxyState) -> web.Application:
    app = web.Application()
    app["state"] = state
    app.router.add_get("/health", handle_health)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_post("/v1/responses", handle_http_responses)
    app.router.add_get("/v1/responses", handle_ws_responses)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19114)
    parser.add_argument("--backend", required=True, help="Base URL of the llama.cpp backend, for example http://127.0.0.1:19094")
    parser.add_argument("--model", default="qwen3.5-4b-ws-exp")
    parser.add_argument("--display-name", default="Qwen3.5 4B WS Experiment")
    parser.add_argument("--description", default="Experimental websocket lineage proxy model")
    parser.add_argument("--context-window", type=int, default=32768)
    parser.add_argument("--auto-compact-token-limit", type=int, default=28000)
    parser.add_argument("--truncation-limit", type=int, default=26000)
    parser.add_argument("--slot-id", type=int, default=0)
    parser.add_argument("--slot-save-dir", default=str(ROOT_DIR / ".marathon" / "ws-lineage-slot-saves"))
    parser.add_argument("--log-dir", default=str(ROOT_DIR / "logs" / "ws-lineage-proxy"))
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    state = ProxyState(
        backend_base_url=args.backend,
        model=args.model,
        display_name=args.display_name,
        description=args.description,
        context_window=args.context_window,
        auto_compact_token_limit=args.auto_compact_token_limit,
        truncation_limit=args.truncation_limit,
        slot_id=args.slot_id,
        slot_save_dir=Path(args.slot_save_dir).resolve(),
        log_dir=Path(args.log_dir).resolve(),
        debug=args.debug,
    )
    app = build_app(state)
    web.run_app(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
