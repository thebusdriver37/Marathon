"""Frozen pre-fix hot paths from Marathon 19daa5f1a5c6ed2e16f637fb2c40e2a964a87ab0.

Synthetic benchmark baseline only. Never import as the application router.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from typing import Any

@dataclass(frozen=True)
class SseEvent:
    name: str | None
    data: str


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


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


def _input_fingerprint_summary(
    items: list[Any],
    max_fingerprints: int,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    digest.update(b"[")
    fingerprints: list[str] = []
    encoded_bytes = 2
    for index, item in enumerate(items):
        encoded = _stable_json(item).encode("utf-8")
        if index:
            digest.update(b",")
            encoded_bytes += 1
        digest.update(encoded)
        encoded_bytes += len(encoded)
        if index < max_fingerprints:
            fingerprints.append(hashlib.sha256(encoded).hexdigest())
    digest.update(b"]")
    return {
        "count": len(items),
        "bytes": encoded_bytes,
        "hash": digest.hexdigest(),
        "fingerprints": tuple(fingerprints),
    }


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


class RouterState:
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
