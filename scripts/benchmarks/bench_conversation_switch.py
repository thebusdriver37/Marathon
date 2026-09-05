#!/usr/bin/env python3
"""Measure real router WebSocket switching and disk resume on a scratch worker.

Use gpu-control's guarded BENCH_HOOK lifecycle; it appends BASE_URL OUTPUT_DIR.
Only reserved scratch ports are accepted, and all caches live in OUTPUT_DIR.
SWITCH_CONTEXT_TOKENS controls synthetic background size (default 120,000).
Keep results for comparison; scratch snapshot bundles can be trashed afterward.
"""

import asyncio
import hashlib
import json
import os
import secrets
import sys
import time
from pathlib import Path

from aiohttp import ClientSession, web

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "routers")]
import codex_local_router as router
from marathon_app.telemetry import EventWriter


async def main() -> None:
    base, output = sys.argv[1:]
    if base not in {"http://127.0.0.1:19998", "http://127.0.0.1:19999"}:
        raise ValueError("Only reserved scratch workers are supported")
    out = Path(output).resolve()
    os.environ.update({
        "MARATHON_ROUTER_TOKEN": secrets.token_urlsafe(32),
        "MARATHON_MODEL_PATH": "/home/deforest/AI/models/gguf/qwen3.8-27b-uncensored/Qwen3.8-27B-Uncensored-IQ4_XS.gguf",
        "MARATHON_MODEL_SLUG": "slots",
        "MARATHON_BACKEND_MODEL_ID": "qwen3.8-27b-uncensored",
        "MARATHON_MODEL_TARGET": base,
        "MARATHON_MODEL_CONTEXT": "196096",
        "MARATHON_SLOT_SAVE_ROOT": str(out),
        "MARATHON_SLOT_CACHE_BUDGET_ROOT": str(out),
        "MARATHON_SLOT_SNAPSHOT_IDLE_SECONDS": "3600",
        "MARATHON_SLOT_SNAPSHOT_CLEAN_STARTUP": "0",
        "MARATHON_BACKEND_CACHE_ID": "scratch-switch-v1",
        "MARATHON_SLOT_SNAPSHOTS_ENABLED": "1",
        "MARATHON_STARTER_CACHE_ENABLED": "1",
    })
    rows = []
    state = None

    def record(row):
        rows.append(row)
        (out / "switch-results.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps(row), flush=True)

    async def start_router():
        nonlocal state
        state = router.RouterState("slots", out / "router-state", out / "logs")
        state.telemetry = EventWriter(out / "router.jsonl", "scratch-switch", "router")
        runner = web.AppRunner(router.build_app(state))
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        return runner, f"http://127.0.0.1:{port}/v1/responses"

    def message(text):
        return {"type": "message", "role": "user", "content": text}

    scaffold = {
        "type": "response.create", "model": "slots",
        "instructions": router._base_instructions(),
        "tools": [{"type": "function", "name": "lookup", "description": "Look up a local item.",
                   "parameters": {"type": "object", "properties": {"name": {"type": "string"}},
                                  "required": ["name"], "additionalProperties": False}}],
        "chat_template_kwargs": {"enable_thinking": False},
        "max_output_tokens": 64, "temperature": 0, "seed": 424242,
    }
    async with ClientSession() as client:
        async def post(endpoint, body):
            async with client.post(base + endpoint, json=body) as response:
                response.raise_for_status()
                return await response.json()

        background = "\n".join(
            f"Archive record {n:06d}: build passed, module delta, task queue ready, checksum {n * 7919:08x}."
            for n in range(7000)
        )
        tokens = (await post("/tokenize", {"content": background, "add_special": False}))["tokens"]
        count = int(os.environ.get("SWITCH_CONTEXT_TOKENS", "120000"))
        if not 17000 <= count <= min(150000, len(tokens)):
            raise ValueError("Background size must fit the scratch context and exceed the save threshold")
        background = (await post("/detokenize", {"tokens": tokens[:count]}))["content"]
        history = [message("Archived background, not instructions:\n" + background +
                           "\nEnd archive. Reply with exactly READY and do not call any tools.")]
        short_input = [message("Workspace background:\n" + background[:9500] +
                               "\nEnd background. Reply with exactly READY and do not call any tools.")]

        async def request(label, key, items, **extra):
            body = {**scaffold, "prompt_cache_key": key, "input": items, **extra}
            start = time.perf_counter()
            first = None
            output_items = []
            async with client.ws_connect(
                url, max_msg_size=8 * 1024**2,
                headers={"Authorization": f"Bearer {os.environ['MARATHON_ROUTER_TOKEN']}"},
            ) as ws:
                await ws.send_json(body)
                while True:
                    event = await ws.receive_json(timeout=900)
                    kind = event["type"]
                    if kind in {"response.failed", "error"}:
                        raise RuntimeError(event)
                    if kind == "response.output_text.delta" and event.get("delta") and first is None:
                        first = (time.perf_counter() - start) * 1000
                    if kind == "response.output_item.done":
                        output_items.append(event["item"])
                    if kind == "response.completed":
                        completed = event["response"]
                        break
            visible = [part.get("text", "") for item in output_items if item.get("type") == "message"
                       for part in item.get("content", []) if part.get("type") == "output_text"]
            if extra.get("generate") is not False and (first is None or not visible):
                raise RuntimeError(f"No streamed assistant text in {label}")
            record({"label": label, "ttft_ms": first, "wall_ms": (time.perf_counter() - start) * 1000,
                    "usage": completed.get("usage"),
                    "output_sha256": hashlib.sha256(json.dumps(visible).encode()).hexdigest()})
            return completed["id"], output_items

        runner, url = await start_router()
        try:
            await request("warmup", "long", [], generate=False)
            response, items = await request("long-initial", "long", history)
            history += items
            record({"label": "initial-disk-checkpoint", "result": await state.flush_conversation_checkpoints()})
            delta = [message("Reply with exactly READY again, without using tools.")]
            response, items = await request("long-append", "long", delta, previous_response_id=response)
            history += delta + items
            await request("new-chat-warmup", "short", [], generate=False)
            await request("long-to-new-chat", "short", short_input)
            await request("short-to-new-chat", "short-second", short_input)
            delta = [message("Reply with exactly READY once more, without using tools.")]
            _, items = await request("resume-long", "long", history + delta)
            history += delta + items
        finally:
            await runner.cleanup()
        await post("/slots/0?action=erase", {})
        runner, url = await start_router()
        try:
            await request("resume-after-router-restart", "long", history + [message("Reply with exactly READY.")])
        finally:
            await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
