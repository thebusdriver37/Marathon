"""Metadata-only import of Codex's own session telemetry."""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


def _session_root() -> Path:
    codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")).expanduser()
    return codex_home / "sessions"


def snapshot_sessions() -> dict[Path, int]:
    root = _session_root()
    if not root.exists():
        return {}
    result: dict[Path, int] = {}
    for path in root.rglob("*.jsonl"):
        try:
            result[path] = path.stat().st_size
        except OSError:
            pass
    return result


def _usage(payload: dict[str, Any], key: str) -> dict[str, int]:
    value = ((payload.get("info") or {}).get(key) or {})
    return {
        name: int(amount)
        for name, amount in value.items()
        if isinstance(amount, (int, float))
    }


def summarize_session_changes(
    before: dict[Path, int], *, cwd: Path | None = None
) -> list[dict[str, Any]]:
    """Summarize appended Codex events without retaining conversation content."""

    after = snapshot_sessions()
    summaries: list[dict[str, Any]] = []
    for path, size in after.items():
        offset = before.get(path, 0)
        if size <= offset:
            continue
        event_counts: Counter[str] = Counter()
        tool_calls: Counter[str] = Counter()
        pending_tools: dict[str, tuple[str, datetime | None]] = {}
        tool_metrics: list[dict[str, Any]] = []
        efforts: set[str] = set()
        first_total: dict[str, int] | None = None
        first_last: dict[str, int] | None = None
        last_total: dict[str, int] | None = None
        model: str | None = None
        context_window: int | None = None
        session_id: str | None = None
        session_cwd: str | None = None
        with path.open("rb") as handle:
            if offset:
                handle.seek(offset - 1)
                ended_on_boundary = handle.read(1) == b"\n"
                handle.seek(offset)
                if not ended_on_boundary:
                    handle.readline()  # Ignore a partial line from a concurrent Codex write.
            for raw in handle:
                try:
                    item = json.loads(raw)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                kind = str(item.get("type") or "unknown")
                payload = item.get("payload") or {}
                subtype = str(payload.get("type") or "")
                event_counts[f"{kind}.{subtype}" if subtype else kind] += 1
                if kind == "session_meta":
                    session_id = str(payload.get("id") or "") or session_id
                    session_cwd = str(payload.get("cwd") or "") or session_cwd
                elif kind == "turn_context":
                    model = str(payload.get("model") or "") or model
                    session_cwd = str(payload.get("cwd") or "") or session_cwd
                    effort = payload.get("effort")
                    if isinstance(effort, str) and effort:
                        efforts.add(effort)
                elif kind == "event_msg" and subtype == "token_count":
                    total = _usage(payload, "total_token_usage")
                    recent = _usage(payload, "last_token_usage")
                    if total:
                        if first_total is None:
                            first_total = total
                            first_last = recent
                        last_total = total
                    context = (payload.get("info") or {}).get("model_context_window")
                    if isinstance(context, int):
                        context_window = context
                elif kind == "response_item" and subtype in {
                    "function_call", "custom_tool_call", "local_shell_call"
                }:
                    name = str(payload.get("name") or subtype)
                    tool_calls[name] += 1
                    call_id = str(payload.get("call_id") or payload.get("id") or "")
                    timestamp = item.get("timestamp")
                    parsed_time: datetime | None = None
                    if isinstance(timestamp, str):
                        try:
                            parsed_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                        except ValueError:
                            pass
                    if call_id:
                        pending_tools[call_id] = (name, parsed_time)
                elif kind == "response_item" and subtype in {
                    "function_call_output", "custom_tool_call_output", "local_shell_call_output"
                }:
                    call_id = str(payload.get("call_id") or payload.get("id") or "")
                    started_tool = pending_tools.pop(call_id, None)
                    timestamp = item.get("timestamp")
                    duration_ms: float | None = None
                    name = "unknown"
                    if started_tool:
                        name, started_at = started_tool
                        if started_at is not None and isinstance(timestamp, str):
                            try:
                                ended_at = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                                duration_ms = max(0.0, (ended_at - started_at).total_seconds() * 1000.0)
                            except ValueError:
                                pass
                    tool_metrics.append(
                        {
                            "call_id": call_id or None,
                            "name": name,
                            "duration_ms": duration_ms,
                            "status": payload.get("status") or "completed",
                        }
                    )

        token_delta: dict[str, int] = {}
        if first_total is not None and last_total is not None:
            # Infer the pre-launch cumulative baseline from Codex's first appended
            # last-turn counters. This stays correct for both new and resumed logs.
            for key, final in last_total.items():
                baseline = first_total.get(key, 0) - (first_last or {}).get(key, 0)
                token_delta[key] = max(0, final - baseline)
        if cwd is not None:
            try:
                matches_cwd = session_cwd is not None and Path(session_cwd).resolve() == cwd.resolve()
            except OSError:
                matches_cwd = False
            if not matches_cwd:
                continue
        summaries.append(
            {
                "session_id": session_id,
                "session_file": str(path),
                "cwd": session_cwd,
                "bytes_appended": size - offset,
                "model": model,
                "context_window": context_window,
                "reasoning_efforts": sorted(efforts),
                "token_delta": token_delta,
                "tool_calls": dict(tool_calls),
                "tool_metrics": tool_metrics,
                "event_counts": dict(event_counts),
            }
        )
    return summaries
