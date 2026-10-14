#!/usr/bin/env python3
"""Exercise nonempty warmups against a normally leased GPU worker.

Uses synthetic history and exact recall checks. No worker eviction, model
configuration changes, or personal conversation data.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import uuid

from aiohttp import ClientSession

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def message(text, role="user"):
    return {"type": "message", "role": role,
            "content": [{"type": "input_text", "text": text}]}


async def exercise(runtime, output, repeats, records, warmup_input):
    rows = []
    scaffold = {
        "type": "response.create", "model": runtime.model.alias,
        "instructions": "Follow the user's instructions. Preserve facts from earlier messages.",
        "tools": [], "reasoning": {"effort": "none"}, "max_output_tokens": 64,
        "seed": 424242,
    }

    async with ClientSession() as client:
        async def request(label, key, items, expected=None, **extra):
            payload = {**scaffold, "prompt_cache_key": key, "input": items, **extra}
            started = time.perf_counter()
            first = None
            outputs = []
            async with client.ws_connect(
                runtime.router_url + "/v1/responses", headers=runtime.router_headers,
                max_msg_size=8 * 1024**2,
            ) as socket:
                await socket.send_json(payload)
                while True:
                    event = await socket.receive_json(timeout=180)
                    kind = event["type"]
                    if kind in {"response.failed", "error"}:
                        raise RuntimeError(event)
                    if kind == "response.output_text.delta" and event.get("delta") and first is None:
                        first = (time.perf_counter() - started) * 1000
                    if kind == "response.output_item.done":
                        outputs.append(event["item"])
                    if kind == "response.completed":
                        response = event["response"]
                        break
            elapsed = (time.perf_counter() - started) * 1000
            text = "".join(part.get("text", "") for item in outputs if item.get("type") == "message"
                           for part in item.get("content", []) if part.get("type") == "output_text")
            if expected is not None and (text.strip() != expected or first is None):
                raise RuntimeError(f"{label}: expected {expected!r}, got {text!r}")
            traces = [json.loads(line) for line in runtime.run_log.read_text().splitlines()]
            trace = next(e["data"] for e in reversed(traces)
                         if e.get("event") == "router.response.completed"
                         and e["data"].get("response_id") == response["id"])
            row = {
                "label": label, "passed": True, "wall_ms": elapsed, "first_text_ms": first,
                "warmup_input": warmup_input,
                "slot": trace.get("slot"), "backend_timings": trace.get("backend_timings"),
                "usage": response.get("usage"),
                "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
            rows.append(row)
            (output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(json.dumps({k: v for k, v in row.items() if k not in {"slot", "usage"}}), flush=True)
            return response["id"], outputs

        background = "\n".join(
            f"Record {i:04d}: module delta; build passed; checksum {i * 7919:08x}; queue idle."
            for i in range(records)
        )
        for repeat in range(repeats):
            key = f"warmup-benchmark-{uuid.uuid4()}"
            canary = f"amber-river-{731 + repeat}"
            history = [message("Initial developer context remains in chronological order.", "developer"),
                       message(f"Reference records:\n{background}\nThe cache canary is {canary}. "
                               "Remember it and reply only READY.")]
            _, generated = await request(f"{repeat}:initial", key, history, "READY")
            history += generated
            prefix_count = 1 if warmup_input == "base" else len(history)
            warm, _ = await request(f"{repeat}:idle-warmup", key, history[:prefix_count], generate=False)
            delta = [message("Reply only with the cache canary from the reference records.")]
            response, generated = await request(f"{repeat}:after-warmup", key,
                                                history[prefix_count:] + delta, canary,
                                                previous_response_id=warm)
            history += delta + generated
            # Ordinary continuation is a control: it already reuses the live slot.
            await request(f"{repeat}:ordinary-continuation", key,
                          [message("Reply only with the same cache canary again.")], canary,
                          previous_response_id=response)
            # A fork starts a new conversation key with the same supplied history.
            fork_key = f"fork-benchmark-{uuid.uuid4()}"
            warm, _ = await request(f"{repeat}:fork-warmup", fork_key, history[:prefix_count], generate=False)
            await request(f"{repeat}:after-fork-warmup", fork_key,
                          history[prefix_count:] + [message("Reply only with the cache canary from the reference records.")],
                          canary, previous_response_id=warm)


def exercise_native(output, repeats):
    """Measure /new inside one actual terminal session and worker lease."""
    from e2e_smoke import LAUNCHER, Terminal, read_events

    project = output / "project"
    project.mkdir()
    instance = f"new-chat-benchmark-{os.getpid()}"
    environment = dict(os.environ, TERM="xterm-256color",
                       MARATHON_CODEX_HOME=str(output / "codex-home"),
                       MARATHON_STOCK_CODEX_HOME=str(output / "stock-codex-home"),
                       MARATHON_AUTO_INSTALL_CLI="0")
    sessions = output / "codex-home/instances" / instance / "sessions"
    terminal = Terminal([str(LAUNCHER), "--instance", instance, "codex", "--no-alt-screen"],
                        project, environment, output / "terminal.txt")
    rows = []

    def traces():
        return [e for p in (output / "runs").rglob("*.jsonl") for e in read_events(p)]

    def warmups():
        return [e for e in traces() if e.get("event") == "router.response.completed"
                and e["data"].get("relation") == "warmup"]

    try:
        terminal.wait(lambda: "| Ready |" in terminal.text and warmups())
        for repeat in range(repeats + 1):
            if repeat:
                count = len(warmups())
                terminal.prompt("/new")
                terminal.wait(lambda: len(warmups()) > count)
            expected = f"NEW_CHAT_OK_{repeat}"

            def completed():
                return next((e["payload"] for p in sessions.rglob("*.jsonl") for e in read_events(p)
                             if e.get("type") == "event_msg"
                             and e.get("payload", {}).get("type") == "task_complete"
                             and e["payload"].get("last_agent_message", "").strip() == expected), None)

            started = time.perf_counter()
            terminal.prompt(f"Reply exactly {expected}. Do not use tools.")
            turn = terminal.wait(completed)
            elapsed = (time.perf_counter() - started) * 1000
            trace = next(e["data"] for e in reversed(traces())
                         if e.get("event") == "router.response.completed"
                         and e["data"].get("backend_ms", 0) > 0)
            row = {"label": "initial" if repeat == 0 else f"{repeat}:new-chat", "passed": True,
                   "wall_ms": elapsed, "first_activity_ms": turn.get("time_to_first_token_ms"),
                   "slot": trace["slot"], "backend_timings": trace["backend_timings"],
                   "output_sha256": hashlib.sha256(expected.encode()).hexdigest()}
            rows.append(row)
            (output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(json.dumps({k: v for k, v in row.items() if k != "slot"}), flush=True)
    finally:
        terminal.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-gpu", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--records", type=int, default=160)
    parser.add_argument("--native-new-chat", action="store_true",
                        help="measure /new in the real terminal instead of the protocol fixture")
    parser.add_argument("--warmup-input", choices=("base", "history"), default="base",
                        help="base matches native startup; history exercises idle-history preparation")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not args.run_gpu:
        parser.error("live inference requires --run-gpu")
    if args.repeats < 1 or not 1 <= args.records <= 300:
        parser.error("repeats must be positive and records must be 1-300")
    output = args.output_dir.resolve() if args.output_dir else Path(tempfile.mkdtemp(prefix="marathon-warmup-"))
    if args.output_dir:
        output.mkdir(parents=True, exist_ok=False)
    os.environ["MARATHON_RUNS_DIR"] = str(output / "runs")
    print(f"Evidence: {output}", flush=True)
    if args.native_new_chat:
        exercise_native(output, args.repeats)
        return
    from marathon_app.catalog import backend_for, discover_models
    from marathon_app.runtime import Runtime, automatic_launch_instance, load_selection
    from marathon_app.ui import _initial_selection

    selection = _initial_selection(discover_models(), load_selection())
    backend = backend_for(selection.model, selection.profile)
    if backend.kind != "llama_swap_pool" or not backend.supports_slots:
        parser.error("select a local pool model with slot caching before running this evaluator")
    runtime = Runtime(selection.model, selection.profile, automatic_launch_instance())
    try:
        runtime.start(lazy_pool=True)
        asyncio.run(exercise(runtime, output, args.repeats, args.records, args.warmup_input))
    finally:
        runtime.cleanup()


if __name__ == "__main__":
    main()
