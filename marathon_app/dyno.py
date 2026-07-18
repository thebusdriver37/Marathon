"""Deterministic, machine-local tuning for Marathon runtime profiles.

Dyno deliberately tunes inference settings rather than model weights. It runs a
small bounded candidate set, keeps every server in the foreground process tree,
and publishes only the selected profile into the user's local configuration.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import re
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, TextIO

from .catalog import Model, Profile, backend_for, server_command, settings


SCHEMA_VERSION = 1
OBJECTIVES = {
    "balanced": ("Balanced", "Balance responsiveness, context, and efficiency."),
    "speed": ("Fastest responses", "Prioritize prompt and generation throughput."),
    "context": ("Longest context", "Prioritize the largest context that loads reliably."),
    "quality": ("Quality / reliability", "Prefer higher-precision cache and conservative settings."),
    "efficiency": ("Lowest power", "Prioritize useful throughput per watt."),
}


@dataclass(frozen=True)
class Candidate:
    id: str
    label: str
    profile: Profile


@dataclass(frozen=True)
class TrialResult:
    candidate: Candidate
    success: bool
    load_seconds: float = 0.0
    loaded_context: int = 0
    prompt_tps: float = 0.0
    decode_tps: float = 0.0
    latency_seconds: float = 0.0
    average_power_w: float = 0.0
    energy_wh: float = 0.0
    average_gpu_utilization: float = 0.0
    peak_gpu_memory_mib: float = 0.0
    peak_temperature_c: float = 0.0
    quality_pass: bool = False
    error: str = ""

    @property
    def efficiency(self) -> float:
        if self.average_power_w <= 0:
            return 0.0
        return (0.35 * self.prompt_tps + 0.65 * self.decode_tps) / self.average_power_w


@dataclass(frozen=True)
class TuningSummary:
    model: Model
    objective: str
    winner: TrialResult
    results: tuple[TrialResult, ...]
    frontier: tuple[str, ...]
    result_dir: Path
    profile_path: Path


def _xdg_config_home() -> Path:
    value = os.environ.get("XDG_CONFIG_HOME")
    return Path(value).expanduser() if value else Path.home() / ".config"


def _xdg_state_home() -> Path:
    value = os.environ.get("XDG_STATE_HOME")
    return Path(value).expanduser() if value else Path.home() / ".local" / "state"


def config_dir() -> Path:
    override = os.environ.get("MARATHON_DYNO_CONFIG_DIR")
    return Path(override).expanduser() if override else _xdg_config_home() / "marathon" / "dyno"


def state_dir() -> Path:
    override = os.environ.get("MARATHON_DYNO_STATE_DIR")
    return Path(override).expanduser() if override else _xdg_state_home() / "marathon" / "dyno"


def profile_path(model: Model) -> Path:
    slug = re.sub(r"[^a-z0-9.-]+", "-", model.id.lower()).strip("-")
    return config_dir() / "profiles" / f"{slug}.json"


def _model_identity(model: Model) -> dict[str, object]:
    shard_match = re.search(r"-00001-of-(\d{5})\.gguf$", model.path.name, re.IGNORECASE)
    if shard_match:
        stem = re.sub(
            r"-00001-of-\d{5}\.gguf$", "", model.path.name, flags=re.IGNORECASE
        )
        files = sorted(model.path.parent.glob(f"{stem}-*-of-*.gguf"))
    else:
        files = [model.path]
    shards = []
    for path in files:
        try:
            stat = path.stat()
        except OSError:
            continue
        shards.append(
            {"name": path.name, "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        )
    return {
        "id": model.id,
        "path": str(model.path.expanduser().resolve()),
        "size_bytes": model.size_bytes,
        "mtime_ns": model.path.stat().st_mtime_ns if model.path.exists() else None,
        "shards": shards,
    }


def _backend_identity(model: Model) -> dict[str, object]:
    try:
        path = backend_for(model).server.expanduser().resolve()
        stat = path.stat()
        return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    except (OSError, ValueError):
        return {}


@lru_cache(maxsize=1)
def machine_identity() -> dict[str, object]:
    gpus: list[dict[str, str]] = []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,memory.total,pci.bus_id",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
        for line in result.stdout.splitlines():
            values = [part.strip() for part in line.split(",", 3)]
            if len(values) == 4:
                gpus.append(dict(zip(("uuid", "name", "memory_mib", "pci_bus_id"), values)))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    cpu_model = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except (OSError, IndexError):
        pass
    return {
        "hostname": platform.node(),
        "kernel": platform.release(),
        "cpu": cpu_model,
        "logical_cpus": os.cpu_count() or 1,
        "gpus": gpus,
    }


def environment_fingerprint(model: Model) -> str:
    payload = {
        "machine": machine_identity(),
        "model": _model_identity(model),
        "backend": _backend_identity(model),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _profile_from_dict(raw: dict[str, object]) -> Profile:
    return Profile(
        id=str(raw["id"]),
        display_name=str(raw["display_name"]),
        description=str(raw.get("description", "")),
        context=int(raw["context"]),
        batch=int(raw["batch"]),
        ubatch=int(raw["ubatch"]),
        parallel=int(raw.get("parallel", 1)),
        gpu_layers=str(raw.get("gpu_layers", "auto")),
        split_mode=str(raw.get("split_mode", "layer")),
        tensor_split=str(raw.get("tensor_split", "")),
        main_gpu=int(raw.get("main_gpu", 0)),
        cache_k=str(raw.get("cache_k", "f16")),
        cache_v=str(raw.get("cache_v", "f16")),
        flash_attention=str(raw.get("flash_attention", "on")),
        extra_args=tuple(str(item) for item in raw.get("extra_args", [])),
        confidence=str(raw.get("confidence", "tuned")),
        frontends=tuple(str(item) for item in raw.get("frontends", ["direct"])),
        tool_thinking_budget=(
            max(0, int(raw["tool_thinking_budget"]))
            if raw.get("tool_thinking_budget") is not None
            else None
        ),
    )


def _sanitize_tuned_profile(model: Model, profile: Profile) -> Profile:
    """Drop obsolete DeepSeek settings copied from older tuned profiles."""

    if model.family.id != "deepseek-v4-flash":
        return profile
    checkpoint_flags = {"-ctxcp", "--ctx-checkpoints", "--swa-checkpoints"}
    values = list(profile.extra_args)
    sanitized: list[str] = []
    index = 0
    while index < len(values):
        value = values[index]
        if value == "--swa-full":
            index += 1
            continue
        if (
            value in checkpoint_flags
            and index + 1 < len(values)
            and values[index + 1] == "0"
        ):
            index += 2
            continue
        sanitized.append(value)
        index += 1
    default_profile = next(
        (
            candidate
            for candidate in model.family.profiles
            if candidate.id == model.family.default_profile
        ),
        None,
    )
    tool_thinking_budget = profile.tool_thinking_budget
    if tool_thinking_budget is None and default_profile is not None:
        tool_thinking_budget = default_profile.tool_thinking_budget
    return replace(
        profile,
        cache_k="f16",
        cache_v="f16",
        flash_attention="off",
        tool_thinking_budget=tool_thinking_budget,
        extra_args=tuple(sanitized),
    )


def load_tuned_profiles(model: Model) -> tuple[Profile, ...]:
    path = profile_path(model)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    if payload.get("schema") != SCHEMA_VERSION:
        return ()
    if payload.get("environment_fingerprint") != environment_fingerprint(model):
        return ()
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict):
        return ()
    loaded: list[Profile] = []
    for objective in OBJECTIVES:
        item = profiles.get(objective)
        if not isinstance(item, dict) or not isinstance(item.get("profile"), dict):
            continue
        try:
            loaded.append(
                _sanitize_tuned_profile(model, _profile_from_dict(item["profile"]))
            )
        except (KeyError, TypeError, ValueError):
            continue
    return tuple(loaded)


def _without_args(args: Iterable[str], flags: set[str]) -> tuple[str, ...]:
    values = list(args)
    result: list[str] = []
    index = 0
    while index < len(values):
        value = values[index]
        if value in flags:
            index += 2
            continue
        result.append(value)
        index += 1
    return tuple(result)


def _variant(
    base: Profile,
    candidate_id: str,
    label: str,
    *,
    context: int | None = None,
    batch: int | None = None,
    ubatch: int | None = None,
    cache: str | None = None,
    add_args: tuple[str, ...] = (),
) -> Candidate:
    controlled = {
        "--threads", "--threads-batch", "--poll", "--spec-type",
        "--spec-ngram-mod-n-match", "--spec-ngram-mod-n-min", "--spec-ngram-mod-n-max",
    }
    extra_args = _without_args(base.extra_args, controlled) + add_args
    profile = replace(
        base,
        id=f"dyno-trial-{candidate_id}",
        display_name=label,
        description="Temporary Dyno trial.",
        context=context or base.context,
        batch=batch or base.batch,
        ubatch=ubatch or base.ubatch,
        cache_k=cache or base.cache_k,
        cache_v=cache or base.cache_v,
        extra_args=extra_args,
        confidence="experimental",
    )
    return Candidate(candidate_id, label, profile)


def _vram_pressure(model: Model) -> float:
    total_mib = 0.0
    for gpu in machine_identity().get("gpus", []):
        if not isinstance(gpu, dict):
            continue
        try:
            total_mib += float(gpu.get("memory_mib", 0))
        except (TypeError, ValueError):
            continue
    if total_mib <= 0:
        return 0.0
    return model.size_bytes / (total_mib * 1024**2)


def candidate_profiles(model: Model, base: Profile, objective: str) -> tuple[Candidate, ...]:
    """Create a small, deterministic search space adapted to the objective."""

    if objective not in OBJECTIVES:
        raise ValueError(f"unknown Dyno objective: {objective}")
    baseline = _variant(base, "baseline", "Known-good baseline")
    pressure = _vram_pressure(model)
    if pressure >= 0.75:
        # Large micro-batches reserve several GiB of compute buffers. Models
        # that nearly fill aggregate VRAM need to search logical batch size
        # first while holding the known-good micro-batch constant.
        moderate_batch = min(1024, max(base.batch, 768))
        moderate_ubatch = base.ubatch
        large_batch = min(2048, max(moderate_batch * 2, 1536))
        large_ubatch = base.ubatch
    else:
        moderate_batch = min(2048, max(base.batch, 1024))
        moderate_ubatch = min(512, max(base.ubatch, 256))
        large_batch = min(4096, max(moderate_batch * 2, 2048))
        large_ubatch = min(512, max(moderate_ubatch * 2, 512))
    cpu_threads = max(4, min(os.cpu_count() or 4, 32))

    if objective == "context":
        contexts = tuple(dict.fromkeys((base.context, min(base.context * 2, 262_144), min(base.context * 4, 262_144))))
        return tuple(
            _variant(
                base,
                f"context-{context // 1024}k",
                f"{context // 1024}K context",
                context=context,
                batch=moderate_batch,
                ubatch=moderate_ubatch,
                cache="q8_0",
            )
            for context in contexts
        )

    if objective == "quality":
        return (
            _variant(base, "quality-baseline", "Higher-precision baseline", cache="f16"),
            _variant(base, "quality-moderate", "Higher-precision responsive", batch=moderate_batch, ubatch=moderate_ubatch, cache="f16"),
            _variant(base, "quality-threaded", "Higher-precision threaded", batch=moderate_batch, ubatch=moderate_ubatch, cache="f16", add_args=("--threads", str(cpu_threads), "--threads-batch", str(cpu_threads))),
        )

    candidates = [
        baseline,
        _variant(base, "moderate-batch", "Responsive batches", batch=moderate_batch, ubatch=moderate_ubatch),
        _variant(base, "large-batch", "Large prompt batches", batch=large_batch, ubatch=large_ubatch),
    ]
    if objective == "speed":
        candidates.append(
            _variant(
                base,
                "ngram-spec",
                "N-gram speculative",
                batch=moderate_batch,
                ubatch=moderate_ubatch,
                add_args=(
                    "--spec-type", "ngram-mod",
                    "--spec-ngram-mod-n-match", "24",
                    "--spec-ngram-mod-n-min", "48",
                    "--spec-ngram-mod-n-max", "64",
                ),
            )
        )
    elif objective == "efficiency":
        candidates.append(
            _variant(base, "low-poll", "Lower CPU polling", batch=moderate_batch, ubatch=moderate_ubatch, cache="q8_0", add_args=("--poll", "20"))
        )
    else:
        candidates.append(
            _variant(base, "f16-cache", "Higher-precision cache", batch=moderate_batch, ubatch=moderate_ubatch, cache="f16")
        )
    return tuple(candidates)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _http_json(url: str, *, payload: dict[str, object] | None = None, timeout: float = 10) -> dict[str, object]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("backend returned a non-object response")
    return value


def _loaded_context(base_url: str, alias: str) -> int | None:
    try:
        payload = _http_json(f"{base_url}/v1/models", timeout=3)
    except (OSError, ValueError, urllib.error.URLError, json.JSONDecodeError):
        return None
    for key in ("data", "models"):
        items = payload.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            aliases = {str(item.get(field) or "") for field in ("id", "slug", "model", "name")}
            if alias not in aliases:
                continue
            meta = item.get("meta")
            if isinstance(meta, dict) and isinstance(meta.get("n_ctx"), int):
                return int(meta["n_ctx"])
            try:
                props = _http_json(f"{base_url}/props", timeout=3)
                generation = props.get("default_generation_settings")
                if isinstance(generation, dict) and isinstance(generation.get("n_ctx"), int):
                    return int(generation["n_ctx"])
            except (OSError, ValueError, urllib.error.URLError, json.JSONDecodeError):
                return None
            return None
    return None


def _parent_death_signal() -> None:
    if platform.system() == "Linux":
        ctypes.CDLL(None).prctl(1, signal.SIGTERM)


def _terminate(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=15)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def _fatal_log_message(path: Path) -> str | None:
    """Recognize backend assertions that can leave the crash handler wedged."""

    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 128 * 1024))
            text = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    patterns = (
        "GGML_ASSERT(",
        "CUDA error",
        "CUDA failure",
        "failed to load model",
        "error loading model",
    )
    for line in reversed(text.splitlines()):
        if any(pattern.lower() in line.lower() for pattern in patterns):
            return line.strip()[-240:]
    return None


class _GpuSampler:
    def __init__(self) -> None:
        self.stop = threading.Event()
        self.samples: list[tuple[float, list[tuple[float, float, float, float]]]] = []
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.stop.clear()
        self.thread = threading.Thread(target=self._run, name="marathon-dyno-gpu", daemon=True)
        self.thread.start()

    def finish(self) -> dict[str, float]:
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=3)
        power: list[float] = []
        utilization: list[float] = []
        memory: list[float] = []
        temperatures: list[float] = []
        energy_wh = 0.0
        previous: tuple[float, float] | None = None
        for timestamp, gpus in self.samples:
            total_power = sum(item[0] for item in gpus)
            power.append(total_power)
            utilization.extend(item[1] for item in gpus)
            memory.extend(item[2] for item in gpus)
            temperatures.extend(item[3] for item in gpus)
            if previous:
                energy_wh += (previous[1] + total_power) / 2 * (timestamp - previous[0]) / 3600
            previous = (timestamp, total_power)
        return {
            "average_power_w": sum(power) / len(power) if power else 0.0,
            "energy_wh": energy_wh,
            "average_gpu_utilization": sum(utilization) / len(utilization) if utilization else 0.0,
            "peak_gpu_memory_mib": max(memory, default=0.0),
            "peak_temperature_c": max(temperatures, default=0.0),
        }

    def _run(self) -> None:
        while not self.stop.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=power.draw,utilization.gpu,memory.used,temperature.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                    check=False,
                )
                gpus: list[tuple[float, float, float, float]] = []
                for line in result.stdout.splitlines():
                    values = [float(part.strip()) for part in line.split(",")]
                    if len(values) == 4:
                        gpus.append(tuple(values))  # type: ignore[arg-type]
                if gpus:
                    self.samples.append((time.monotonic(), gpus))
            except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
                pass
            self.stop.wait(0.5)


def _measurement_prompt(target_tokens: int) -> str:
    unit = (
        "A distributed service accepts work, validates inputs, records state, retries transient failures, "
        "and emits deterministic diagnostics for operators. "
    )
    body = (unit * max(1, target_tokens * 4 // len(unit) + 1))[: target_tokens * 4]
    return body + "\n\nWrite at least 300 words explaining the operational tradeoffs. Do not use tools."


def _response_metrics(result: dict[str, object], elapsed: float) -> tuple[float, float, int, bool]:
    timings = result.get("timings") if isinstance(result.get("timings"), dict) else {}
    usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
    prompt_tps = float(timings.get("prompt_per_second") or 0.0)
    decode_tps = float(timings.get("predicted_per_second") or 0.0)
    completion = int(usage.get("completion_tokens") or 0)
    if decode_tps <= 0 and completion > 0 and elapsed > 0:
        decode_tps = completion / elapsed
    choices = result.get("choices")
    content = ""
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict):
            content = str(message.get("content") or "")
    return prompt_tps, decode_tps, completion, completion >= 32 and len(content) >= 80


def _benchmark(base_url: str, alias: str, context: int, objective: str) -> dict[str, float | bool]:
    warmup = {
        "model": alias,
        "messages": [{"role": "user", "content": "Reply with exactly: ready"}],
        "temperature": 0.0,
        "max_tokens": 16,
        "stream": False,
    }
    _http_json(f"{base_url}/v1/chat/completions", payload=warmup, timeout=180)
    target_tokens = min(8192 if objective == "context" else 2048, max(512, context // 8))
    payload = {
        "model": alias,
        "messages": [{"role": "user", "content": _measurement_prompt(target_tokens)}],
        "temperature": 0.0,
        "seed": 3407,
        "max_tokens": 160,
        "stream": False,
    }
    prompt_rates: list[float] = []
    decode_rates: list[float] = []
    latencies: list[float] = []
    passes: list[bool] = []
    sampler = _GpuSampler()
    sampler.start()
    try:
        for _ in range(2):
            started = time.monotonic()
            result = _http_json(f"{base_url}/v1/chat/completions", payload=payload, timeout=600)
            elapsed = time.monotonic() - started
            prompt_tps, decode_tps, _completion, quality_pass = _response_metrics(result, elapsed)
            prompt_rates.append(prompt_tps)
            decode_rates.append(decode_tps)
            latencies.append(elapsed)
            passes.append(quality_pass)
    finally:
        hardware = sampler.finish()
    return {
        "prompt_tps": sum(prompt_rates) / len(prompt_rates),
        "decode_tps": sum(decode_rates) / len(decode_rates),
        "latency_seconds": sum(latencies) / len(latencies),
        "quality_pass": all(passes),
        **hardware,
    }


def _trial(
    model: Model,
    candidate: Candidate,
    objective: str,
    directory: Path,
    progress: Callable[[str], None] | None,
) -> TrialResult:
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    command = server_command(model, candidate.profile, backend_for(model))
    command[command.index("--port") + 1] = str(port)
    command[command.index("--host") + 1] = "127.0.0.1"
    log_path = directory / f"{candidate.id}.log"
    process: subprocess.Popen[str] | None = None
    log: TextIO | None = None
    load_started = time.monotonic()
    try:
        log = log_path.open("w", encoding="utf-8")
        environment = os.environ.copy()
        environment.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        environment.setdefault("CUDA_SCALE_LAUNCH_QUEUES", "4x")
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            preexec_fn=_parent_death_signal,
        )
        if progress:
            progress(f"Loading {candidate.label}")
        deadline = time.monotonic() + settings().health_timeout
        loaded_context: int | None = None
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"llama-server exited during load; see {log_path}")
            fatal = _fatal_log_message(log_path)
            if fatal:
                _terminate(process)
                raise RuntimeError(f"llama-server failed during load: {fatal}; see {log_path}")
            loaded_context = _loaded_context(base_url, model.alias)
            if loaded_context:
                break
            time.sleep(1)
        if not loaded_context:
            raise TimeoutError(f"model did not load within {settings().health_timeout}s")
        if loaded_context < candidate.profile.context:
            raise RuntimeError(
                f"backend loaded {loaded_context:,} context, below requested {candidate.profile.context:,}"
            )
        load_seconds = time.monotonic() - load_started
        if progress:
            progress(f"Benchmarking {candidate.label}")
        metrics = _benchmark(base_url, model.alias, loaded_context, objective)
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited during benchmark; see {log_path}")
        return TrialResult(
            candidate=candidate,
            success=bool(metrics["quality_pass"]),
            load_seconds=load_seconds,
            loaded_context=loaded_context,
            prompt_tps=float(metrics["prompt_tps"]),
            decode_tps=float(metrics["decode_tps"]),
            latency_seconds=float(metrics["latency_seconds"]),
            average_power_w=float(metrics["average_power_w"]),
            energy_wh=float(metrics["energy_wh"]),
            average_gpu_utilization=float(metrics["average_gpu_utilization"]),
            peak_gpu_memory_mib=float(metrics["peak_gpu_memory_mib"]),
            peak_temperature_c=float(metrics["peak_temperature_c"]),
            quality_pass=bool(metrics["quality_pass"]),
            error="" if metrics["quality_pass"] else "deterministic response gate failed",
        )
    except (OSError, ValueError, RuntimeError, TimeoutError, urllib.error.URLError) as error:
        return TrialResult(
            candidate=candidate,
            success=False,
            load_seconds=time.monotonic() - load_started,
            error=str(error),
        )
    finally:
        _terminate(process)
        if log:
            log.close()


def _dominates(left: TrialResult, right: TrialResult) -> bool:
    left_values = (left.prompt_tps, left.decode_tps, float(left.loaded_context), left.efficiency)
    right_values = (right.prompt_tps, right.decode_tps, float(right.loaded_context), right.efficiency)
    return all(a >= b for a, b in zip(left_values, right_values)) and any(
        a > b for a, b in zip(left_values, right_values)
    )


def pareto_frontier(results: Iterable[TrialResult]) -> tuple[TrialResult, ...]:
    successful = tuple(result for result in results if result.success and result.quality_pass)
    return tuple(
        candidate
        for candidate in successful
        if not any(_dominates(other, candidate) for other in successful if other is not candidate)
    )


def _normalized(value: float, values: list[float]) -> float:
    low, high = min(values), max(values)
    if high <= low:
        return 1.0
    return (value - low) / (high - low)


def select_winner(results: Iterable[TrialResult], objective: str) -> TrialResult:
    frontier = pareto_frontier(results)
    if not frontier:
        raise RuntimeError("no Dyno candidate passed the load and response gates")
    prompt = [item.prompt_tps for item in frontier]
    decode = [item.decode_tps for item in frontier]
    contexts = [float(item.loaded_context) for item in frontier]
    efficiency = [item.efficiency for item in frontier]
    precision = [1.0 if item.candidate.profile.cache_k == "f16" and item.candidate.profile.cache_v == "f16" else 0.0 for item in frontier]

    def score(item: TrialResult) -> float:
        p = _normalized(item.prompt_tps, prompt)
        d = _normalized(item.decode_tps, decode)
        c = _normalized(float(item.loaded_context), contexts)
        e = _normalized(item.efficiency, efficiency)
        q = 1.0 if item.candidate.profile.cache_k == "f16" and item.candidate.profile.cache_v == "f16" else 0.0
        if objective == "speed":
            return 0.40 * p + 0.60 * d
        if objective == "context":
            return 0.75 * c + 0.15 * p + 0.10 * d
        if objective == "quality":
            return 0.70 * _normalized(q, precision) + 0.15 * p + 0.15 * d
        if objective == "efficiency":
            return 0.80 * e + 0.10 * p + 0.10 * d
        return 0.30 * p + 0.35 * d + 0.20 * c + 0.15 * e

    return max(frontier, key=lambda item: (score(item), item.decode_tps, item.prompt_tps, item.candidate.id))


def _json_result(result: TrialResult) -> dict[str, object]:
    payload = asdict(result)
    payload["candidate"] = {
        "id": result.candidate.id,
        "label": result.candidate.label,
        "profile": asdict(result.candidate.profile),
    }
    payload["efficiency"] = result.efficiency
    return payload


def _publish_profile(model: Model, objective: str, winner: TrialResult) -> Path:
    path = profile_path(model)
    fingerprint = environment_fingerprint(model)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    if payload.get("environment_fingerprint") != fingerprint:
        payload = {}
    profiles = payload.get("profiles") if isinstance(payload.get("profiles"), dict) else {}
    label = OBJECTIVES[objective][0]
    published = replace(
        winner.candidate.profile,
        id=f"dyno-{objective}",
        display_name=f"Dyno · {label}",
        description=(
            f"Machine-tuned: {winner.prompt_tps:.1f} prompt tok/s · "
            f"{winner.decode_tps:.1f} decode tok/s."
        ),
        confidence="tuned",
    )
    profiles[objective] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile": asdict(published),
        "metrics": _json_result(winner),
    }
    payload = {
        "schema": SCHEMA_VERSION,
        "environment_fingerprint": fingerprint,
        "machine": machine_identity(),
        "model": _model_identity(model),
        "backend": _backend_identity(model),
        "profiles": profiles,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def run_tuning(
    model: Model,
    base: Profile,
    objective: str,
    progress: Callable[[str], None] | None = None,
) -> TuningSummary:
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown Dyno objective: {objective}")
    backend = backend_for(model)
    if backend.kind != "llama_cpp":
        raise ValueError(
            f"Dyno does not tune {backend.display_name} yet; use the model's "
            "verified Marathon profile"
        )
    try:
        from .runtime import _gpu_processes

        conflicts = _gpu_processes()
    except (OSError, ValueError):
        conflicts = []
    if conflicts:
        detail = "; ".join(f"PID {item['pid']} {item['name']}" for item in conflicts)
        raise RuntimeError(f"Dyno requires free GPUs; active compute processes: {detail}")

    stamp = (
        datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        + f"-{time.time_ns() % 1_000_000:06d}"
    )
    run_directory = state_dir() / f"{stamp}_{model.id}_{objective}"
    run_directory.mkdir(parents=True, exist_ok=False)
    candidates = candidate_profiles(model, base, objective)
    results: list[TrialResult] = []
    try:
        for index, candidate in enumerate(candidates, 1):
            if progress:
                progress(f"Trial {index}/{len(candidates)} · {candidate.label}")
            results.append(_trial(model, candidate, objective, run_directory, progress))
        winner = select_winner(results, objective)
        profile_file = _publish_profile(model, objective, winner)
        frontier = tuple(item.candidate.id for item in pareto_frontier(results))
        summary_payload = {
            "schema": SCHEMA_VERSION,
            "objective": objective,
            "environment_fingerprint": environment_fingerprint(model),
            "model": _model_identity(model),
            "winner": winner.candidate.id,
            "frontier": frontier,
            "results": [_json_result(item) for item in results],
            "profile_path": str(profile_file),
        }
        (run_directory / "summary.json").write_text(
            json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8"
        )
        return TuningSummary(
            model=model,
            objective=objective,
            winner=winner,
            results=tuple(results),
            frontier=frontier,
            result_dir=run_directory,
            profile_path=profile_file,
        )
    except BaseException as error:
        # Trial logs remain useful after interruption, but no profile is
        # published until every candidate has been scored.
        (run_directory / "partial-summary.json").write_text(
            json.dumps(
                {
                    "schema": SCHEMA_VERSION,
                    "objective": objective,
                    "model": _model_identity(model),
                    "error": f"{type(error).__name__}: {error}",
                    "results": [_json_result(item) for item in results],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        raise
