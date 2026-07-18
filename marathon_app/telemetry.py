"""Durable, privacy-conscious telemetry for Marathon runs.

The canonical format is append-only JSON Lines.  Every process writes complete
events under an advisory file lock, so a power loss may lose the last event but
cannot invalidate the rest of the run.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import threading
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .codex_telemetry import refresh_legacy_tool_metrics, summarize_active_sessions


SCHEMA_VERSION = 1
_SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization:\s*bearer\s+)[^\s]+"),
    re.compile(r"(?i)\b(sk-[a-z0-9_-]{12,})\b"),
    re.compile(r"(?i)(token|secret|password|api[_-]?key)(\s*[=:]\s*)[^\s,;]+"),
)
_LLAMA_PROMPT_TIMING = re.compile(
    r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens"
)
_LLAMA_DECODE_TIMING = re.compile(
    r"(?:^|\|)\s*eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens",
    re.MULTILINE,
)
_DS4_PROMPT_TIMING = re.compile(
    r"chat ctx=\d+\.\.\d+:(\d+).*prompt done\s+([\d.]+)s"
)
_DS4_FINISH_TIMING = re.compile(
    r"chat ctx=\d+\.\.\d+:\d+\s+gen=(\d+).*finish=\S+\s+([\d.]+)s"
)


def _xdg_state_home() -> Path:
    configured = os.environ.get("XDG_STATE_HOME")
    return Path(configured).expanduser() if configured else Path.home() / ".local" / "state"


def runs_dir() -> Path:
    configured = os.environ.get("MARATHON_RUNS_DIR")
    return Path(configured).expanduser() if configured else _xdg_state_home() / "marathon" / "runs"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def redact_text(value: str, limit: int = 8_192) -> str:
    """Bound operational output and remove common credential shapes."""

    value = value.replace("\x00", "")
    for pattern in _SECRET_PATTERNS:
        if pattern.groups >= 2:
            value = pattern.sub(lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]", value)
        else:
            value = pattern.sub("[REDACTED]", value)
    if len(value) > limit:
        return value[:limit] + f"… [truncated {len(value) - limit} chars]"
    return value


def _safe_value(value: Any, depth: int = 0) -> Any:
    if depth > 8:
        return "[depth limit]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _safe_value(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_value(item, depth + 1) for item in value]
    return redact_text(repr(value), 1_024)


class EventWriter:
    """A lightweight multi-process JSONL event writer."""

    def __init__(self, path: Path | None, run_id: str | None, source: str) -> None:
        self.path = path
        self.run_id = run_id
        self.source = source
        self._seq = 0
        self._lock = threading.Lock()
        self.dropped_events = 0

    @classmethod
    def from_env(cls, source: str) -> "EventWriter":
        path = os.environ.get("MARATHON_RUN_LOG")
        return cls(Path(path).expanduser() if path else None, os.environ.get("MARATHON_RUN_ID"), source)

    @property
    def enabled(self) -> bool:
        return self.path is not None and self.run_id is not None

    def emit(self, event: str, data: dict[str, Any] | None = None, *, level: str = "info") -> None:
        if not self.enabled or self.path is None:
            return
        with self._lock:
            self._seq += 1
            entry = {
                "schema": SCHEMA_VERSION,
                "ts": _utc_now(),
                "mono_ns": time.monotonic_ns(),
                "run_id": self.run_id,
                "source": self.source,
                "pid": os.getpid(),
                "seq": self._seq,
                "level": level,
                "event": event,
                "data": _safe_value(data or {}),
            }
            encoded = (json.dumps(entry, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX)
                    view = memoryview(encoded)
                    while view:
                        written = os.write(descriptor, view)
                        view = view[written:]
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                finally:
                    os.close(descriptor)
            except OSError:
                self.dropped_events += 1


def create_run_writer(model_id: str) -> EventWriter:
    directory = runs_dir()
    directory.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:12]
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^a-z0-9.-]+", "-", model_id.lower()).strip("-")[:64]
    return EventWriter(directory / f"{stamp}_{slug}_{run_id}.jsonl", run_id, "runtime")


def list_runs() -> list[Path]:
    directory = runs_dir()
    return sorted(directory.glob("*.jsonl"), key=lambda path: path.stat().st_mtime_ns if path.exists() else 0)


def resolve_run(value: str | Path | None = None) -> Path:
    if value is None or str(value) in {"", "last", "latest"}:
        available = list_runs()
        if not available:
            raise FileNotFoundError(f"no Marathon traces found under {runs_dir()}")
        return available[-1]
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    matches = [path for path in list_runs() if str(value) in path.name]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"no Marathon trace matches: {value}")
    raise ValueError(f"ambiguous run '{value}': {', '.join(path.name for path in matches[-5:])}")


def read_events(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                yield {"event": "telemetry.invalid_line", "level": "error", "data": {"line": line_number}}
                continue
            if isinstance(value, dict):
                yield value


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def summarize_run(path: Path, *, live: bool = False) -> dict[str, Any]:
    events = list(read_events(path))
    counts = Counter(str(item.get("event", "unknown")) for item in events)
    started = next((item for item in events if item.get("event") == "run.started"), None)
    completed = next((item for item in reversed(events) if item.get("event") == "run.completed"), None)
    ready = next((item for item in reversed(events) if item.get("event") == "runtime.ready"), None)
    start_data = (started or {}).get("data") or {}
    end_data = (completed or {}).get("data") or {}

    router_responses = [item.get("data") or {} for item in events if item.get("event") == "router.response.completed"]
    direct_turns = [item.get("data") or {} for item in events if item.get("event") == "direct.turn.completed"]
    codex_sessions = [
        refresh_legacy_tool_metrics(item.get("data") or {})
        for item in events
        if item.get("event") == "codex.session.completed"
    ]
    frontend_starts = [
        item
        for item in events
        if item.get("event") == "frontend.started"
        and (item.get("data") or {}).get("frontend") == "codex"
    ]
    frontend_completions = [
        item
        for item in events
        if item.get("event") == "frontend.completed"
        and (item.get("data") or {}).get("frontend") == "codex"
    ]
    active_codex_sessions: list[dict[str, Any]] = []
    if live and frontend_starts and (
        not frontend_completions
        or (_number(frontend_starts[-1].get("mono_ns")) or 0)
        > (_number(frontend_completions[-1].get("mono_ns")) or 0)
    ):
        active_start = frontend_starts[-1]
        active_data = active_start.get("data") or {}
        timestamp = active_start.get("ts")
        cwd = active_data.get("cwd")
        if isinstance(timestamp, str) and isinstance(cwd, str):
            try:
                started_at = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                active_codex_sessions = summarize_active_sessions(
                    started_at,
                    cwd=Path(cwd),
                )
            except (OSError, ValueError):
                active_codex_sessions = []
    codex_sessions.extend(active_codex_sessions)
    usage_totals: Counter[str] = Counter()
    backend_latencies: list[float] = []
    warmups = 0
    prompt_tokens = 0.0
    prompt_ms = 0.0
    generated_tokens = 0.0
    generated_ms = 0.0
    router_tool_counts: Counter[str] = Counter()
    for response in router_responses:
        is_warmup = response.get("relation") == "warmup"
        if is_warmup:
            warmups += 1
        usage = response.get("usage") or response.get("backend", {}).get("usage") or {}
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            if isinstance(usage.get(key), (int, float)):
                usage_totals[key] += int(usage[key])
        latency = _number(response.get("latency_ms") or response.get("backend", {}).get("latency_ms"))
        if latency is not None and not is_warmup:
            backend_latencies.append(latency)
        timings = response.get("backend_timings") or response.get("backend", {}).get("timings") or {}
        prompt_tokens += _number(timings.get("prompt_n")) or 0.0
        prompt_ms += _number(timings.get("prompt_ms")) or 0.0
        generated_tokens += _number(timings.get("predicted_n")) or 0.0
        generated_ms += _number(timings.get("predicted_ms")) or 0.0
        router_tool_counts.update((response.get("output") or {}).get("tool_calls") or {})

    if prompt_ms == 0 or generated_ms == 0:
        process_prompt_tokens = 0.0
        process_prompt_ms = 0.0
        process_generated_tokens = 0.0
        process_generated_ms = 0.0
        ds4_prompt_seconds: float | None = None
        for event in events:
            if event.get("event") != "process.output":
                continue
            data = event.get("data") or {}
            process = data.get("process")
            message = data.get("message")
            if not isinstance(message, str):
                continue
            if process == "llama":
                prompt_match = _LLAMA_PROMPT_TIMING.search(message)
                if prompt_match:
                    process_prompt_ms += float(prompt_match.group(1))
                    process_prompt_tokens += int(prompt_match.group(2))
                decode_match = _LLAMA_DECODE_TIMING.search(message)
                if decode_match:
                    process_generated_ms += float(decode_match.group(1))
                    process_generated_tokens += int(decode_match.group(2))
            elif process == "ds4-coordinator":
                prompt_match = _DS4_PROMPT_TIMING.search(message)
                if prompt_match:
                    ds4_prompt_seconds = float(prompt_match.group(2))
                    process_prompt_tokens += int(prompt_match.group(1))
                    process_prompt_ms += ds4_prompt_seconds * 1000.0
                    continue
                finish_match = _DS4_FINISH_TIMING.search(message)
                if finish_match:
                    total_seconds = float(finish_match.group(2))
                    decode_seconds = max(0.0, total_seconds - (ds4_prompt_seconds or 0.0))
                    process_generated_tokens += int(finish_match.group(1))
                    process_generated_ms += decode_seconds * 1000.0
                    ds4_prompt_seconds = None
        if prompt_ms == 0:
            prompt_ms, prompt_tokens = process_prompt_ms, process_prompt_tokens
        if generated_ms == 0:
            generated_ms, generated_tokens = process_generated_ms, process_generated_tokens

    direct_ttft = [
        value for value in (_number(turn.get("ttft_ms")) for turn in direct_turns)
        if value is not None
    ]

    gpu_samples = [item.get("data") or {} for item in events if item.get("event") == "hardware.gpu.sample"]
    gpu_power: list[float] = []
    gpu_utilization: list[float] = []
    gpu_peak_memory = 0.0
    gpu_peak_temp = 0.0
    energy_wh = 0.0
    previous_sample: tuple[float, float] | None = None
    gpu_events = [item for item in events if item.get("event") == "hardware.gpu.sample"]
    for event, sample in zip(gpu_events, gpu_samples):
        total_power = 0.0
        for gpu in sample.get("gpus", []):
            power = _number(gpu.get("power_w"))
            utilization = _number(gpu.get("utilization_pct"))
            memory = _number(gpu.get("memory_used_mib"))
            temperature = _number(gpu.get("temperature_c"))
            if power is not None:
                gpu_power.append(power)
                total_power += power
            if utilization is not None:
                gpu_utilization.append(utilization)
            gpu_peak_memory = max(gpu_peak_memory, memory or 0.0)
            gpu_peak_temp = max(gpu_peak_temp, temperature or 0.0)
        mono_ns = _number(event.get("mono_ns"))
        if mono_ns is not None and previous_sample is not None:
            previous_mono, previous_power = previous_sample
            interval_hours = max(0.0, (mono_ns - previous_mono) / 1_000_000_000 / 3600)
            energy_wh += ((previous_power + total_power) / 2.0) * interval_hours
        if mono_ns is not None:
            previous_sample = (mono_ns, total_power)

    errors = [item for item in events if item.get("level") in {"error", "critical"} or str(item.get("event", "")).endswith(".error")]
    error_events = []
    for item in errors[-10:]:
        data = item.get("data") or {}
        error_events.append(
            {
                "ts": item.get("ts"),
                "event": item.get("event"),
                "message": data.get("error") or data.get("message") or data.get("stderr"),
            }
        )
    duration = _number(end_data.get("duration_s"))
    if duration is None and events:
        first_mono = _number(events[0].get("mono_ns"))
        last_mono = _number(events[-1].get("mono_ns"))
        if first_mono is not None and last_mono is not None:
            duration = max(0.0, (last_mono - first_mono) / 1_000_000_000)

    tool_counts: Counter[str] = Counter()
    codex_usage: Counter[str] = Counter()
    reasoning_efforts: Counter[str] = Counter()
    tool_durations: list[float] = []
    tool_failures: list[dict[str, Any]] = []
    for session in codex_sessions:
        tool_counts.update(session.get("tool_calls") or {})
        reasoning_efforts.update(session.get("reasoning_efforts") or [])
        for tool in session.get("tool_metrics") or []:
            tool_duration = _number(tool.get("duration_ms"))
            if tool_duration is not None:
                tool_durations.append(tool_duration)
            if tool.get("status") == "failed":
                tool_failures.append(tool)
        for key, value in (session.get("token_delta") or {}).items():
            if isinstance(value, (int, float)):
                codex_usage[key] += int(value)

    configured_rate = (start_data.get("telemetry") or {}).get("electricity_rate_usd_kwh")
    try:
        electricity_rate = float(configured_rate) if configured_rate not in (None, "") else None
    except (TypeError, ValueError):
        electricity_rate = None

    return {
        "path": path,
        "run_id": (started or {}).get("run_id") or (events[0].get("run_id") if events else "unknown"),
        "complete": completed is not None,
        "active": live and completed is None,
        "model": start_data.get("model", {}).get("id") or start_data.get("model_id") or "unknown",
        "profile": start_data.get("profile", {}).get("id") or start_data.get("profile_id") or "unknown",
        "context": ((ready or {}).get("data") or {}).get("context")
        or start_data.get("profile", {}).get("requested_context")
        or start_data.get("context"),
        "duration_s": duration or 0.0,
        "event_count": len(events),
        "event_types": counts,
        "router_turns": len(router_responses) - warmups,
        "router_warmups": warmups,
        "direct_turns": len(direct_turns),
        "codex_sessions": len(codex_sessions),
        "active_codex_sessions": len(active_codex_sessions),
        "usage": dict(usage_totals),
        "codex_usage": dict(codex_usage),
        "tool_calls": dict(tool_counts),
        "reasoning_efforts": dict(reasoning_efforts),
        "avg_tool_duration_ms": sum(tool_durations) / len(tool_durations)
        if tool_durations else None,
        "tool_failures": len(tool_failures),
        "failed_tools": dict(
            Counter(str(tool.get("name") or "unknown") for tool in tool_failures)
        ),
        "tool_failure_details": tool_failures[-10:],
        "avg_backend_latency_ms": sum(backend_latencies) / len(backend_latencies) if backend_latencies else None,
        "prompt_tps": prompt_tokens / (prompt_ms / 1000.0) if prompt_ms > 0 else None,
        "decode_tps": generated_tokens / (generated_ms / 1000.0) if generated_ms > 0 else None,
        "avg_direct_ttft_ms": sum(direct_ttft) / len(direct_ttft) if direct_ttft else None,
        "gpu_samples": len(gpu_samples),
        "avg_gpu_power_w": sum(gpu_power) / len(gpu_power) if gpu_power else None,
        "avg_gpu_utilization_pct": sum(gpu_utilization) / len(gpu_utilization) if gpu_utilization else None,
        "energy_wh": energy_wh,
        "estimated_gpu_energy_cost_usd": energy_wh / 1000.0 * electricity_rate
        if electricity_rate is not None else None,
        "peak_gpu_memory_mib": gpu_peak_memory or None,
        "peak_gpu_temperature_c": gpu_peak_temp or None,
        "errors": len(errors),
        "error_events": error_events,
        "dropped_events": end_data.get("dropped_events", 0),
        "router_tool_calls": dict(router_tool_counts),
    }
