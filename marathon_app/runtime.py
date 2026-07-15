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
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Iterator, TextIO

from .catalog import Model, Profile, ROOT_DIR, server_command, settings


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
    for path in (CONFIG_DIR, USER_STATE_DIR / "logs", RUNTIME_DIR, ROUTER_STATE_DIR, SLOT_ROOT):
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
        self.llama: subprocess.Popen[bytes] | None = None
        self.router: subprocess.Popen[bytes] | None = None
        self._logs: list[TextIO] = []
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
    def context_window(self) -> int:
        return self._context_window

    @property
    def auto_compact_token_limit(self) -> int:
        return max(1, self.context_window * 9 // 10)

    @property
    def truncation_limit(self) -> int:
        return max(1, self.context_window * 85 // 100)

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

    def _spawn(self, command: list[str], log: TextIO, env: dict[str, str]) -> subprocess.Popen[bytes]:
        return subprocess.Popen(
            command,
            cwd=ROOT_DIR,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            preexec_fn=_set_parent_death_signal,
        )

    def start(self, progress: Callable[[str], None] | None = None) -> None:
        self.acquire()
        self._install_handlers()
        stop_legacy_services()
        self._check_conflicts()
        environment = os.environ.copy()
        environment.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        environment.setdefault("CUDA_SCALE_LAUNCH_QUEUES", "4x")
        if progress:
            progress("Starting llama.cpp")
        slot_path = SLOT_ROOT / self.model.alias
        slot_path.mkdir(parents=True, exist_ok=True)
        llama_command = server_command(self.model, self.profile)
        llama_command.extend(["--slot-save-path", str(slot_path)])
        self.llama = self._spawn(
            llama_command, self._open_log(self.model_log), environment
        )
        self._write_session()
        self._wait_for_model(progress)
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
            }
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
        )
        self._write_session()
        self._wait_for_router(progress)
        self._write_catalog()
        if progress:
            progress("Backend ready")

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
        try:
            lines = self.model_log.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return "Loading model weights"
        interesting = [
            line.strip() for line in lines[-80:]
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
        for process in (self.router, self.llama):
            self._terminate(process)
        for handle in self._logs:
            with contextlib.suppress(OSError):
                handle.close()
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

    def _terminate(self, process: subprocess.Popen[bytes] | None) -> None:
        if process is None or process.poll() is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        deadline = time.monotonic() + self.config.stop_timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                return
            time.sleep(0.2)
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=5)

    def __enter__(self) -> "Runtime":
        return self

    def __exit__(self, _kind: object, _value: object, _traceback: object) -> None:
        self.cleanup()
