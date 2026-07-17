"""Foreground process supervision for Marathon's local inference runtime."""

from __future__ import annotations

import contextlib
import ctypes
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from typing import Callable, Iterator, TextIO

from .catalog import Model, Profile, ROOT_DIR, server_command, settings
from .telemetry import EventWriter, create_run_writer, redact_text, runs_dir


def _xdg_path(env_name: str, fallback: Path) -> Path:
    value = os.environ.get(env_name)
    return Path(value).expanduser() if value else fallback


CONFIG_DIR = _xdg_path("XDG_CONFIG_HOME", Path.home() / ".config") / "marathon"
USER_STATE_DIR = _xdg_path("XDG_STATE_HOME", Path.home() / ".local" / "state") / "marathon"
RUNTIME_DIR = _xdg_path(
    "XDG_RUNTIME_DIR", Path("/tmp") / f"marathon-{os.getuid()}"
) / "marathon"
AI_ROOT = settings().ai_root
ROUTER_STATE_DIR = AI_ROOT / "cache" / "marathon" / "router"
SLOT_ROOT = Path(
    os.environ.get("MARATHON_SLOT_SAVE_ROOT", AI_ROOT / "cache" / "marathon" / "slots")
).expanduser()
SELECTION_FILE = CONFIG_DIR / "selection.json"
SESSION_FILE = RUNTIME_DIR / "session.json"
LOCK_FILE = RUNTIME_DIR / "runtime.lock"


def ensure_dirs() -> None:
    for path in (CONFIG_DIR, USER_STATE_DIR / "logs", runs_dir(), RUNTIME_DIR, ROUTER_STATE_DIR, SLOT_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def load_selection() -> dict[str, str]:
    try:
        value = json.loads(SELECTION_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {key: str(value[key]) for key in ("model", "profile", "frontend") if value.get(key)}


def save_selection(model: Model, profile: Profile, frontend: str) -> None:
    ensure_dirs()
    temporary = SELECTION_FILE.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {"schema": 1, "model": model.id, "profile": profile.id, "frontend": frontend},
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    temporary.replace(SELECTION_FILE)


def _http_json(url: str, timeout: float = 3) -> dict[str, object]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _loaded_model_context(payload: dict[str, object], model_alias: str) -> int | None:
    for key in ("data", "models"):
        items = payload.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            aliases = {
                str(item.get(field) or "")
                for field in ("id", "slug", "model", "name")
            }
            if model_alias not in aliases:
                continue
            meta = item.get("meta")
            if isinstance(meta, dict):
                context = meta.get("n_ctx")
                if isinstance(context, int) and context > 0:
                    return context
    return None


def _model_is_loaded(payload: dict[str, object], model_alias: str) -> bool:
    for key in ("data", "models"):
        items = payload.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            if model_alias in {
                str(item.get(field) or "")
                for field in ("id", "slug", "model", "name")
            }:
                return True
    return False


def _props_context_window(payload: dict[str, object]) -> int | None:
    settings_payload = payload.get("default_generation_settings")
    if not isinstance(settings_payload, dict):
        return None
    context = settings_payload.get("n_ctx")
    return context if isinstance(context, int) and context > 0 else None


def _port_pid(port: int) -> int | None:
    if shutil.which("ss"):
        result = subprocess.run(
            ["ss", "-ltnp", f"( sport = :{port} )"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        import re

        match = re.search(r"pid=(\d+)", result.stdout)
        return int(match.group(1)) if match else None
    return None


def _cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except OSError:
        return ""


def _gpu_processes() -> list[dict[str, str]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    processes: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) == 3 and parts[0].isdigit():
            processes.append({"pid": parts[0], "name": parts[1], "memory_mib": parts[2]})
    return processes


def stop_legacy_services() -> None:
    """Stop only the known Paddock service and Marathon's old detached backend."""

    if shutil.which("systemctl"):
        subprocess.run(
            ["systemctl", "--user", "stop", "paddock.service"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        subprocess.run(
            ["systemctl", "--user", "disable", "paddock.service"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    backend_script = ROOT_DIR / "scripts" / "ops" / "backend.sh"
    if backend_script.is_file():
        subprocess.run(
            [str(backend_script), "stop"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def request_stop() -> bool:
    """Ask an active foreground Marathon supervisor to shut down."""

    ensure_dirs()
    stopped = False
    try:
        session = json.loads(SESSION_FILE.read_text(encoding="utf-8"))
        pid = int(session.get("supervisor_pid", 0))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pid = 0
    if pid > 1 and "marathon_app" in _cmdline(pid):
        try:
            os.kill(pid, signal.SIGTERM)
            stopped = True
        except ProcessLookupError:
            pass
    elif pid:
        SESSION_FILE.unlink(missing_ok=True)
    stop_legacy_services()
    return stopped


def _set_parent_death_signal() -> None:
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None)
        libc.prctl(1, signal.SIGTERM)


class Runtime:
    def __init__(self, model: Model, profile: Profile) -> None:
        self.model = model
        self.profile = profile
        self.config = settings()
        self._context_window = profile.context
        self.llama: subprocess.Popen[str] | None = None
        self.router: subprocess.Popen[str] | None = None
        self._logs: list[TextIO] = []
        self._log_threads: list[threading.Thread] = []
        self._recent_model_lines: deque[str] = deque(maxlen=120)
        self._sample_stop = threading.Event()
        self._sampler: threading.Thread | None = None
        self._journal_cursor: str | None = None
        self._last_kernel_poll = 0.0
        self.telemetry: EventWriter | None = None
        self._run_started_mono: float | None = None
        self._lock: TextIO | None = None
        self._owns_lock = False
        self._cleaned = False
        self._old_handlers: dict[int, object] = {}

    @property
    def llama_url(self) -> str:
        return f"http://{self.config.llama_host}:{self.config.llama_port}"

    @property
    def router_url(self) -> str:
        return f"http://{self.config.router_host}:{self.config.router_port}"

    @property
    def log_dir(self) -> Path:
        return USER_STATE_DIR / "logs"

    @property
    def model_log(self) -> Path:
        return self.log_dir / "llama-server.log"

    @property
    def router_log(self) -> Path:
        return self.log_dir / "router.log"

    @property
    def catalog_file(self) -> Path:
        return RUNTIME_DIR / "codex-models.json"

    @property
    def run_id(self) -> str | None:
        return self.telemetry.run_id if self.telemetry else None

    @property
    def run_log(self) -> Path | None:
        return self.telemetry.path if self.telemetry else None

    @property
    def context_window(self) -> int:
        return self._context_window

    @property
    def context_reserve_tokens(self) -> int:
        """Leave model-scaled room for tool results and the next generation."""

        configured = os.environ.get("MARATHON_CONTEXT_RESERVE_TOKENS")
        if configured:
            try:
                return min(self.context_window // 2, max(1, int(configured)))
            except ValueError:
                pass
        reserve = max(12_288, min(32_768, self.context_window // 8))
        return min(self.context_window // 2, reserve)

    @property
    def auto_compact_token_limit(self) -> int:
        return max(1, self.context_window - self.context_reserve_tokens)

    @property
    def truncation_limit(self) -> int:
        configured = os.environ.get("MARATHON_COMPACTION_GUARD_TOKENS")
        guard: int | None = None
        if configured:
            try:
                guard = max(1, int(configured))
            except ValueError:
                pass
        if guard is None:
            guard = max(2_048, min(8_192, self.context_window // 20))
        return max(1, self.auto_compact_token_limit - guard)

    def acquire(self) -> None:
        ensure_dirs()
        self._lock = LOCK_FILE.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self._lock.seek(0)
            owner = self._lock.read().strip()
            self._lock.close()
            self._lock = None
            detail = ""
            try:
                metadata = json.loads(owner)
                pid = metadata.get("pid")
                terminal = metadata.get("terminal") or "an unknown terminal"
                detail = f" on {terminal}" + (f" (PID {pid})" if pid else "")
            except (json.JSONDecodeError, AttributeError):
                pass
            raise RuntimeError(
                f"Marathon is already open{detail}. Return to that terminal, "
                "or run 'marathon stop' if you want to close it."
            ) from error
        self._owns_lock = True
        self._lock.seek(0)
        self._lock.truncate()
        self._lock.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "terminal": os.ttyname(sys.stdin.fileno()) if sys.stdin.isatty() else None,
                    "started_at": int(time.time()),
                }
            )
        )
        self._lock.flush()

    def _install_handlers(self) -> None:
        for signum in (signal.SIGTERM, signal.SIGHUP):
            self._old_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, self._signal_exit)

    def _signal_exit(self, signum: int, _frame: object) -> None:
        self.cleanup()
        raise SystemExit(128 + signum)

    def _check_conflicts(self) -> None:
        for port in (self.config.llama_port, self.config.router_port):
            pid = _port_pid(port)
            if pid:
                raise RuntimeError(f"port {port} is still occupied by PID {pid}: {_cmdline(pid)}")
        processes = _gpu_processes()
        if processes:
            detail = "; ".join(
                f"PID {item['pid']} {item['name']} ({item['memory_mib']} MiB)" for item in processes
            )
            raise RuntimeError(f"GPU compute processes are already active: {detail}")

    def _open_log(self, path: Path) -> TextIO:
        handle = path.open("w", encoding="utf-8")
        self._logs.append(handle)
        return handle

    def record(self, event: str, data: dict[str, object] | None = None, *, level: str = "info") -> None:
        if self.telemetry:
            self.telemetry.emit(event, data, level=level)

    def _capture_output(self, source: str, stream: TextIO, legacy_log: TextIO) -> None:
        capture = os.environ.get("MARATHON_TELEMETRY_PROCESS_OUTPUT", "1").lower() not in {
            "0", "false", "no", "off"
        }
        try:
            for line in stream:
                legacy_log.write(line)
                legacy_log.flush()
                message = line.rstrip("\r\n")
                if source == "llama" and message:
                    self._recent_model_lines.append(message)
                if capture and message:
                    level = "error" if any(word in message.lower() for word in ("error", "failed", "xid")) else "info"
                    self.record("process.output", {"process": source, "message": redact_text(message)}, level=level)
        except (OSError, ValueError) as error:
            self.record("process.capture.error", {"process": source, "error": str(error)}, level="error")

    def _spawn(
        self, command: list[str], log: TextIO, env: dict[str, str], source: str
    ) -> subprocess.Popen[str]:
        process = subprocess.Popen(
            command,
            cwd=ROOT_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            start_new_session=True,
            preexec_fn=_set_parent_death_signal,
        )
        assert process.stdout is not None
        thread = threading.Thread(
            target=self._capture_output,
            args=(source, process.stdout, log),
            name=f"marathon-{source}-log",
            daemon=True,
        )
        thread.start()
        self._log_threads.append(thread)
        self.record(
            "process.started",
            {"process": source, "pid": process.pid, "command": command},
        )
        return process

    def start(self, progress: Callable[[str], None] | None = None) -> None:
        self.acquire()
        self._install_handlers()
        self.telemetry = create_run_writer(self.model.id)
        self._run_started_mono = time.monotonic()
        llama_command = server_command(self.model, self.profile)
        slot_path = SLOT_ROOT / self.model.alias
        llama_command.extend(["--slot-save-path", str(slot_path)])
        self.record(
            "run.started",
            {
                "model": {
                    "id": self.model.id,
                    "display_name": self.model.display_name,
                    "path": str(self.model.path),
                    "size_bytes": self.model.size_bytes,
                    "mtime_ns": self.model.path.stat().st_mtime_ns if self.model.path.exists() else None,
                    "quant": self.model.quant,
                    "family": self.model.family.id,
                },
                "profile": {
                    "id": self.profile.id,
                    "display_name": self.profile.display_name,
                    "requested_context": self.profile.context,
                    "batch": self.profile.batch,
                    "ubatch": self.profile.ubatch,
                    "parallel": self.profile.parallel,
                    "split_mode": self.profile.split_mode,
                    "tensor_split": self.profile.tensor_split,
                    "cache_k": self.profile.cache_k,
                    "cache_v": self.profile.cache_v,
                    "flash_attention": self.profile.flash_attention,
                    "tool_thinking_budget": self.profile.tool_thinking_budget,
                    "confidence": self.profile.confidence,
                },
                "llama_command": llama_command,
                "cwd": str(Path.cwd()),
                "python": sys.version.split()[0],
                "platform": sys.platform,
                "kernel": os.uname().release if hasattr(os, "uname") else None,
                "backend_binary": {
                    "path": llama_command[0],
                    "size_bytes": Path(llama_command[0]).stat().st_size
                    if Path(llama_command[0]).exists() else None,
                    "mtime_ns": Path(llama_command[0]).stat().st_mtime_ns
                    if Path(llama_command[0]).exists() else None,
                },
                "trace_disk": {
                    "free_bytes": shutil.disk_usage(runs_dir()).free,
                    "total_bytes": shutil.disk_usage(runs_dir()).total,
                },
                "telemetry": {
                    "process_output": os.environ.get("MARATHON_TELEMETRY_PROCESS_OUTPUT", "1"),
                    "sample_interval_s": os.environ.get("MARATHON_TELEMETRY_INTERVAL", "2"),
                    "capture_content": False,
                    "electricity_rate_usd_kwh": os.environ.get(
                        "MARATHON_ELECTRICITY_RATE_USD_KWH"
                    ),
                },
            },
        )
        self._start_sampler()
        stop_legacy_services()
        try:
            self._check_conflicts()
        except Exception as error:
            self.record("runtime.conflict", {"error": str(error)}, level="error")
            raise
        environment = os.environ.copy()
        environment.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        environment.setdefault("CUDA_SCALE_LAUNCH_QUEUES", "4x")
        if progress:
            progress("Starting llama.cpp")
        slot_path.mkdir(parents=True, exist_ok=True)
        load_started = time.monotonic()
        self.llama = self._spawn(
            llama_command, self._open_log(self.model_log), environment, "llama"
        )
        self._write_session()
        self._wait_for_model(progress)
        self.record(
            "backend.model.ready",
            {
                "load_ms": (time.monotonic() - load_started) * 1000.0,
                "loaded_context": self.context_window,
            },
        )
        if progress:
            progress("Starting Marathon router")
        router_env = environment.copy()
        router_env.update(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "MARATHON_AI_ROOT": str(AI_ROOT),
                "MARATHON_MODELS_DIR": str(self.config.model_root),
                "MARATHON_MODEL_PATH": str(self.model.path),
                "MARATHON_MODEL_SLUG": self.model.alias,
                "MARATHON_MODEL_DISPLAY_NAME": self.model.display_name,
                "MARATHON_MODEL_DESCRIPTION": f"{self.model.display_name} via Marathon",
                "MARATHON_MODEL_CONTEXT": str(self.context_window),
                "MARATHON_MODEL_AUTO_COMPACT_TOKEN_LIMIT": str(
                    self.auto_compact_token_limit
                ),
                "MARATHON_MODEL_TRUNCATION_LIMIT": str(self.truncation_limit),
                "MARATHON_MODEL_PORT": str(self.config.llama_port),
                "MARATHON_MODEL_TARGET": self.llama_url,
                "MARATHON_SLOT_SAVE_ROOT": str(SLOT_ROOT),
                "MARATHON_RUN_ID": str(self.run_id or ""),
                "MARATHON_RUN_LOG": str(self.run_log or ""),
            }
        )
        if self.profile.tool_thinking_budget is not None:
            router_env["MARATHON_MODEL_TOOL_THINKING_BUDGET_TOKENS"] = str(
                self.profile.tool_thinking_budget
            )
        configured_python = os.environ.get("MARATHON_ROUTER_PYTHON")
        python = (
            Path(configured_python).expanduser()
            if configured_python
            else ROOT_DIR / ".marathon" / "venv" / "bin" / "python3"
        )
        if not python.is_file() and configured_python:
            resolved_python = shutil.which(configured_python)
            python = Path(resolved_python) if resolved_python else python
        if not python.is_file():
            raise RuntimeError("Marathon Python environment is missing; run: marathon setup-deps")
        self.router = self._spawn(
            [
                str(python), str(ROOT_DIR / "scripts" / "routers" / "codex_local_router.py"),
                "--host", self.config.router_host, "--port", str(self.config.router_port),
                "--default-model", self.model.alias, "--state-dir", str(ROUTER_STATE_DIR),
                "--log-dir", str(self.log_dir),
            ],
            self._open_log(self.router_log),
            router_env,
            "router",
        )
        self._write_session()
        self._wait_for_router(progress)
        self._write_catalog()
        if progress:
            progress("Backend ready")
        self.record(
            "runtime.ready",
            {
                "startup_ms": (time.monotonic() - (self._run_started_mono or time.monotonic())) * 1000.0,
                "context": self.context_window,
                "llama_pid": self.llama.pid if self.llama else None,
                "router_pid": self.router.pid if self.router else None,
            },
        )

    @staticmethod
    def _parse_number(value: str) -> float | int | None:
        value = value.strip()
        try:
            number = float(value)
        except ValueError:
            return None
        return int(number) if number.is_integer() else number

    def _sample_gpus(self) -> None:
        fields = (
            "index,name,uuid,pci.bus_id,driver_version,utilization.gpu,memory.used,memory.total,"
            "temperature.gpu,power.draw,power.limit,clocks.current.sm,pstate"
        )
        try:
            result = subprocess.run(
                ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as error:
            self.record("hardware.gpu.error", {"error": str(error)}, level="error")
            return
        # A foreground Ctrl-C is also delivered to an in-flight nvidia-smi
        # child. That is an orderly Marathon shutdown, not a GPU failure.
        if result.returncode in {-signal.SIGINT, -signal.SIGTERM}:
            return
        if result.returncode != 0:
            self.record(
                "hardware.gpu.error",
                {"returncode": result.returncode, "stderr": redact_text(result.stderr)},
                level="error",
            )
            return
        names = (
            "index", "name", "uuid", "pci_bus_id", "driver_version", "utilization_pct", "memory_used_mib",
            "memory_total_mib", "temperature_c", "power_w", "power_limit_w",
            "sm_clock_mhz", "pstate",
        )
        gpus: list[dict[str, object]] = []
        for line in result.stdout.splitlines():
            values = [part.strip() for part in line.split(",")]
            if len(values) != len(names):
                continue
            gpu: dict[str, object] = {}
            for name, value in zip(names, values):
                if name in {"name", "uuid", "pci_bus_id", "driver_version", "pstate"}:
                    gpu[name] = value
                else:
                    parsed = self._parse_number(value)
                    gpu[name] = parsed if parsed is not None else value
            gpus.append(gpu)
        self.record("hardware.gpu.sample", {"gpus": gpus})

    def _sample_host(self) -> None:
        memory: dict[str, int] = {}
        try:
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                key, raw = line.split(":", 1)
                if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
                    memory[f"{key.lower()}_kib"] = int(raw.strip().split()[0])
        except (OSError, ValueError, IndexError):
            pass
        self.record(
            "hardware.host.sample",
            {
                "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
                **memory,
            },
        )

    def _initialize_kernel_cursor(self) -> None:
        if not shutil.which("journalctl"):
            return
        try:
            result = subprocess.run(
                ["journalctl", "-k", "-n", "0", "--show-cursor", "--no-pager"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if line.startswith("-- cursor: "):
                    self._journal_cursor = line.removeprefix("-- cursor: ").strip()

    def _sample_kernel_events(self) -> None:
        if not self._journal_cursor or time.monotonic() - self._last_kernel_poll < 10:
            return
        self._last_kernel_poll = time.monotonic()
        try:
            result = subprocess.run(
                [
                    "journalctl", "-k", f"--after-cursor={self._journal_cursor}",
                    "--show-cursor", "--output=json", "--no-pager",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return
        if result.returncode != 0:
            return
        keywords = ("nvrm", "xid", "pcie", "aer:", "fallen off", "gpu has fallen")
        for line in result.stdout.splitlines():
            if line.startswith("-- cursor: "):
                self._journal_cursor = line.removeprefix("-- cursor: ").strip()
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            message = str(item.get("MESSAGE") or "")
            if any(keyword in message.lower() for keyword in keywords):
                self.record(
                    "system.kernel.alert",
                    {
                        "message": redact_text(message),
                        "priority": item.get("PRIORITY"),
                        "identifier": item.get("SYSLOG_IDENTIFIER"),
                        "kernel_timestamp": item.get("__REALTIME_TIMESTAMP"),
                    },
                    level="error",
                )

    def _sample_backend_metrics(self) -> None:
        try:
            with urllib.request.urlopen(f"{self.llama_url}/metrics", timeout=2) as response:
                body = response.read().decode("utf-8", errors="replace")
        except (OSError, urllib.error.URLError):
            return
        metrics: dict[str, float | int] = {}
        for line in body.splitlines():
            if not line or line.startswith("#") or " " not in line:
                continue
            key, raw_value = line.rsplit(None, 1)
            try:
                number = float(raw_value)
            except ValueError:
                continue
            if key.startswith(("llamacpp:", "llama_", "process_")):
                metrics[key] = int(number) if number.is_integer() else number
        if metrics:
            self.record("backend.metrics.sample", {"metrics": metrics})

    def _sample_loop(self) -> None:
        try:
            interval = max(0.5, float(os.environ.get("MARATHON_TELEMETRY_INTERVAL", "2")))
        except ValueError:
            interval = 2.0
        sample_backend = os.environ.get(
            "MARATHON_BACKEND_METRICS_ENABLED", "0"
        ).lower() in {"1", "true", "yes", "on"}
        while not self._sample_stop.is_set():
            started = time.monotonic()
            self._sample_gpus()
            self._sample_host()
            if sample_backend:
                self._sample_backend_metrics()
            self._sample_kernel_events()
            elapsed = time.monotonic() - started
            self._sample_stop.wait(max(0.1, interval - elapsed))

    def _start_sampler(self) -> None:
        self._sample_stop.clear()
        self._initialize_kernel_cursor()
        self._sampler = threading.Thread(
            target=self._sample_loop,
            name="marathon-telemetry-sampler",
            daemon=True,
        )
        self._sampler.start()

    def _wait_for_model(self, progress: Callable[[str], None] | None) -> None:
        deadline = time.monotonic() + self.config.health_timeout
        while time.monotonic() < deadline:
            if self.llama and self.llama.poll() is not None:
                raise RuntimeError(f"llama-server exited while loading; see {self.model_log}")
            try:
                payload = _http_json(f"{self.llama_url}/v1/models")
                loaded_context = _loaded_model_context(payload, self.model.alias)
                if _model_is_loaded(payload, self.model.alias):
                    if loaded_context is None:
                        try:
                            loaded_context = _props_context_window(
                                _http_json(f"{self.llama_url}/props")
                            )
                        except (
                            OSError,
                            ValueError,
                            urllib.error.URLError,
                            json.JSONDecodeError,
                        ):
                            pass
                    self._context_window = loaded_context or self.profile.context
                    return
            except (OSError, ValueError, urllib.error.URLError, json.JSONDecodeError):
                pass
            if progress:
                progress(self.latest_model_status())
            time.sleep(1)
        raise TimeoutError(f"model did not become ready within {self.config.health_timeout}s")

    def _wait_for_router(self, progress: Callable[[str], None] | None) -> None:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if self.router and self.router.poll() is not None:
                raise RuntimeError(f"router exited during startup; see {self.router_log}")
            try:
                if _http_json(f"{self.router_url}/health").get("ok") is True:
                    return
            except (OSError, ValueError, urllib.error.URLError, json.JSONDecodeError):
                pass
            if progress:
                progress("Waiting for the local API router")
            time.sleep(0.5)
        raise TimeoutError("Marathon router did not become ready within 30 seconds")

    def latest_model_status(self) -> str:
        interesting = [
            line.strip() for line in list(self._recent_model_lines)[-80:]
            if any(token in line.lower() for token in ("load", "cuda", "buffer", "graph", "slot"))
        ]
        return interesting[-1][-120:] if interesting else "Loading model weights"

    def _write_catalog(self) -> None:
        payload = _http_json(f"{self.router_url}/v1/models")
        temporary = self.catalog_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.catalog_file)

    def _write_session(self) -> None:
        ensure_dirs()
        payload = {
            "schema": 1,
            "supervisor_pid": os.getpid(),
            "llama_pid": self.llama.pid if self.llama else None,
            "router_pid": self.router.pid if self.router else None,
            "model": self.model.id,
            "profile": self.profile.id,
            "context": self.context_window,
            "started_at": int(time.time()),
            "run_id": self.run_id,
            "run_log": str(self.run_log) if self.run_log else None,
        }
        temporary = SESSION_FILE.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(SESSION_FILE)

    def status(self) -> dict[str, object]:
        return {
            "model": self.model.display_name,
            "model_id": self.model.id,
            "profile": self.profile.display_name,
            "profile_id": self.profile.id,
            "context": self.context_window,
            "router_url": f"{self.router_url}/v1",
            "llama_pid": self.llama.pid if self.llama and self.llama.poll() is None else None,
            "router_pid": self.router.pid if self.router and self.router.poll() is None else None,
            "run_id": self.run_id,
            "run_log": str(self.run_log) if self.run_log else None,
        }

    @contextlib.contextmanager
    def frontend_signals(self) -> Iterator[None]:
        old = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            yield
        finally:
            signal.signal(signal.SIGINT, old)

    def cleanup(self) -> None:
        if self._cleaned:
            return
        self._cleaned = True
        self._sample_stop.set()
        if self._sampler is not None:
            self._sampler.join(timeout=5)
        for process in (self.router, self.llama):
            self._terminate(process)
        for thread in self._log_threads:
            thread.join(timeout=3)
        for handle in self._logs:
            with contextlib.suppress(OSError):
                handle.close()
        duration = (
            time.monotonic() - self._run_started_mono
            if self._run_started_mono is not None
            else 0.0
        )
        self.record(
            "run.completed",
            {
                "duration_s": duration,
                "router_returncode": self.router.poll() if self.router else None,
                "llama_returncode": self.llama.poll() if self.llama else None,
                "dropped_events": self.telemetry.dropped_events if self.telemetry else 0,
            },
        )
        if self._owns_lock:
            SESSION_FILE.unlink(missing_ok=True)
            self.catalog_file.unlink(missing_ok=True)
        for signum, handler in self._old_handlers.items():
            signal.signal(signum, handler)
        if self._lock is not None:
            with contextlib.suppress(OSError):
                self._lock.seek(0)
                self._lock.truncate()
                fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
                self._lock.close()
            self._lock = None
            self._owns_lock = False

    def _terminate(self, process: subprocess.Popen[str] | None) -> None:
        if process is None or process.poll() is not None:
            return
        name = "router" if process is self.router else "llama"
        started = time.monotonic()
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        deadline = time.monotonic() + self.config.stop_timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                self.record(
                    "process.stopped",
                    {
                        "process": name,
                        "pid": process.pid,
                        "returncode": process.returncode,
                        "stop_ms": (time.monotonic() - started) * 1000.0,
                        "forced": False,
                    },
                )
                return
            time.sleep(0.2)
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=5)
        self.record(
            "process.stopped",
            {
                "process": name,
                "pid": process.pid,
                "returncode": process.returncode,
                "stop_ms": (time.monotonic() - started) * 1000.0,
                "forced": True,
            },
            level="error",
        )

    def __enter__(self) -> "Runtime":
        return self

    def __exit__(self, _kind: object, _value: object, _traceback: object) -> None:
        self.cleanup()
