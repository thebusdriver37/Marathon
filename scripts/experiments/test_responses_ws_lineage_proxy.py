#!/usr/bin/env python3
"""Exercise the experimental Responses websocket lineage proxy.

The test intentionally branches from an older response id after a few linear
turns. The proxy trace is then inspected to verify that the branch request used
slot restore and only processed the new suffix on the backend.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import websockets

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "scripts" / "routers"))

from codex_local_router import _base_instructions  # type: ignore  # noqa: E402


def entry_usage(entry: dict[str, Any]) -> dict[str, Any]:
    backend = entry.get("backend")
    if isinstance(backend, dict):
        usage = backend.get("usage")
        if isinstance(usage, dict):
            return usage
    usage = entry.get("backend_usage")
    return usage if isinstance(usage, dict) else {}


def message_item(text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": text}],
    }


async def recv_until_completed(ws: Any) -> dict[str, Any]:
    created_id: str | None = None
    output_items: list[dict[str, Any]] = []
    usage: dict[str, Any] | None = None
    while True:
        raw = await ws.recv()
        payload = json.loads(raw)
        msg_type = payload.get("type")
        if msg_type == "response.created":
            created_id = payload["response"]["id"]
        elif msg_type == "response.output_item.done":
            item = payload.get("item")
            if isinstance(item, dict):
                output_items.append(item)
        elif msg_type == "response.completed":
            usage = payload.get("response", {}).get("usage")
            return {
                "response_id": payload["response"]["id"],
                "created_id": created_id,
                "usage": usage,
                "output_items": output_items,
            }
        elif msg_type == "error":
            raise RuntimeError(payload.get("error", {}).get("message") or "proxy returned error")


async def run_test(args: argparse.Namespace) -> list[dict[str, Any]]:
    trace_path = Path(args.trace_log).resolve()
    if trace_path.exists():
        trace_path.unlink()

    instructions = _base_instructions() if args.include_base_instructions else ""
    uri = args.ws_url
    response_ids: dict[str, str] = {}
    steps = [
        {
            "label": "root",
            "previous": None,
            "text": "ROOT CONTEXT " * args.root_words,
        },
        {
            "label": "step2",
            "previous": "root",
            "text": "alpha " * args.step_words,
        },
        {
            "label": "step3",
            "previous": "step2",
            "text": "beta " * args.step_words,
        },
        {
            "label": "step4",
            "previous": "step3",
            "text": "gamma " * args.step_words,
        },
        {
            "label": "branch-from-root",
            "previous": "root",
            "text": args.branch_text,
        },
    ]

    async with websockets.connect(uri, max_size=32 * 1024 * 1024) as ws:
        for step in steps:
            payload: dict[str, Any] = {
                "type": "response.create",
                "model": args.model,
                "instructions": instructions,
                "input": [message_item(step["text"])],
                "tools": [],
                "prompt_cache_key": args.prompt_cache_key,
            }
            previous_label = step["previous"]
            if previous_label is not None:
                payload["previous_response_id"] = response_ids[previous_label]
            await ws.send(json.dumps(payload, separators=(",", ":")))
            result = await recv_until_completed(ws)
            response_ids[step["label"]] = result["response_id"]
            print(f"{step['label']}: response_id={result['response_id']} usage={result['usage']}")

    if not trace_path.exists():
        raise RuntimeError(f"trace log not found: {trace_path}")

    entries = []
    with trace_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    if len(entries) < len(steps):
        raise RuntimeError(f"expected at least {len(steps)} trace entries, got {len(entries)}")

    recent = entries[-len(steps):]
    root = recent[0]
    branch = recent[-1]
    if branch.get("relation") != "branch":
        raise RuntimeError(f"expected final relation to be 'branch', got {branch.get('relation')!r}")
    if branch.get("restore_result") is None:
        raise RuntimeError("expected branch turn to restore a slot snapshot")

    root_usage = entry_usage(root)
    branch_usage = entry_usage(branch)
    root_cached_tokens = (((root_usage.get("input_tokens_details") or {}).get("cached_tokens")))
    branch_cached_tokens = (((branch_usage.get("input_tokens_details") or {}).get("cached_tokens")))

    root_prompt_n = int(((root.get("backend_timings") or {}).get("prompt_n") or 0))
    branch_prompt_n = int(((branch.get("backend_timings") or {}).get("prompt_n") or 0))
    root_backend_ms = float(root.get("backend_ms") or 0.0)
    branch_backend_ms = float(branch.get("backend_ms") or 0.0)

    if isinstance(branch_cached_tokens, int) and branch_cached_tokens > 0:
        if isinstance(root_cached_tokens, int) and root_cached_tokens != 0:
            raise RuntimeError(
                f"expected root cached_tokens to be 0, got {root_cached_tokens}"
            )
        assertion_text = (
            "branch turn restored an ancestor snapshot and reported cached prompt tokens."
        )
    elif root_prompt_n > 0 and branch_prompt_n > 0:
        if branch_prompt_n >= root_prompt_n:
            raise RuntimeError(
                f"expected branch prompt_n ({branch_prompt_n}) to be lower than root prompt_n ({root_prompt_n})"
            )
        assertion_text = (
            "branch turn restored an ancestor snapshot and processed fewer prompt tokens than the root turn."
        )
    else:
        if root_backend_ms <= 0 or branch_backend_ms <= 0:
            raise RuntimeError(
                f"expected positive backend_ms values, got root={root_backend_ms} branch={branch_backend_ms}"
            )
        if branch_backend_ms >= root_backend_ms:
            raise RuntimeError(
                f"expected branch backend_ms ({branch_backend_ms}) to be lower than root backend_ms ({root_backend_ms})"
            )
        assertion_text = (
            "branch turn restored an ancestor snapshot and completed faster than the root turn."
        )

    print("\nTrace summary:")
    for entry in recent:
        timings = entry.get("backend_timings") or {}
        slot = entry.get("slot") or {}
        slot_prepare_ms = slot.get("prepare_ms")
        if slot_prepare_ms is None:
            slot_prepare_ms = entry.get("slot_prepare_ms")
        backend_ms = entry.get("backend_ms")
        print(
            f"- relation={entry.get('relation')} response_id={entry.get('response_id')} "
            f"prompt_n={timings.get('prompt_n', 'n/a')} cache_n={timings.get('cache_n', 'n/a')} "
            f"cached_tokens={((entry_usage(entry).get('input_tokens_details') or {}).get('cached_tokens', 'n/a'))} "
            f"slot_prepare_ms={(slot_prepare_ms if slot_prepare_ms is not None else 0.0):.2f} "
            f"backend_ms={(backend_ms if backend_ms is not None else 0.0):.2f}"
        )

    print(f"\nAssertion passed: {assertion_text}")
    return recent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ws-url", default="ws://127.0.0.1:19114/v1/responses")
    parser.add_argument("--model", default="qwen3.5-4b-ws-exp")
    parser.add_argument("--trace-log", required=True)
    parser.add_argument("--prompt-cache-key", default="marathon-ws-lineage-test")
    parser.add_argument("--root-words", type=int, default=700)
    parser.add_argument("--step-words", type=int, default=80)
    parser.add_argument("--branch-text", default="branch tiny question")
    parser.add_argument("--include-base-instructions", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    asyncio.run(run_test(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
