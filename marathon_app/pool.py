"""Process-owned reservations for the broker's exclusive local workers."""

import fcntl
import hashlib
import json
import os
import re
import csv
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TextIO

from .catalog import Backend
import yaml


def _output(args: list[str]) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True,
                          timeout=5).stdout


def _worker_gpu_blockers(backend: Backend) -> dict[str, str]:
    """Check physical devices, including jobs outside Marathon's lease system.

    Broker configuration is the source of GPU placement. Only PIDs in the
    exact configured container are reusable, never arbitrary llama processes.
    Configured pools fail closed when occupancy cannot be established.
    """
    if backend.identity_config is None:
        return {}
    try:
        models = yaml.safe_load(backend.identity_config.read_text())["models"]
        # These inventories are independent. Keep them fresh for every lease,
        # but overlap process/driver startup instead of paying for it serially.
        with ThreadPoolExecutor(max_workers=3) as executor:
            gpu_query = executor.submit(_output, [
                "nvidia-smi", "--query-gpu=index,uuid,memory.free",
                "--format=csv,noheader,nounits",
            ])
            process_query = executor.submit(_output, [
                "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                "--format=csv,noheader,nounits",
            ])
            container_query = executor.submit(_output, ["docker", "ps", "--format", "{{.Names}}"])
            gpu_output = gpu_query.result()
            process_output = process_query.result()
            running = set(container_query.result().splitlines())
        devices = {}
        for row in csv.reader(gpu_output.splitlines()):
            index, uuid, free = (item.strip() for item in row)
            devices[index] = devices[uuid] = (uuid, int(free))
        processes = []
        for row in csv.reader(process_output.splitlines()):
            uuid, pid, memory = (item.strip() for item in row)
            processes.append((uuid, int(pid), int(memory)))
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError, subprocess.SubprocessError):
        return dict.fromkeys(backend.pool_models, "cannot verify GPU availability")
    blocked = {}
    for model_id in backend.pool_models:
        try:
            model = models[model_id]
            command = model["cmd"].replace("${MODEL_ID}", model_id)
            for key, value in model.get("macros", {}).items():
                command = command.replace("${" + str(key) + "}", str(value))
            args = shlex.split(command)
            placement = args[args.index("--gpus") + 1].strip('\"')
            if not placement.startswith("device="):
                raise ValueError("GPU placement must be explicit")
            gpu_ids = placement.removeprefix("device=").split(",")
            container = args[args.index("--name") + 1]
            own_pids = set()
            if container in running:
                own_pids = {int(line.strip()) for line in _output(
                    ["docker", "top", container, "-eo", "pid"]
                ).splitlines()[1:] if line.strip()}
            for gpu in gpu_ids:
                uuid, free = devices[gpu.strip()]
                occupants = [(pid, memory) for device, pid, memory in processes if device == uuid]
                foreign = [pid for pid, _ in occupants if pid not in own_pids]
                reusable = sum(memory for pid, memory in occupants if pid in own_pids)
                if foreign:
                    blocked[model_id] = f"GPU {gpu} occupied by other jobs (PIDs {foreign})"
                    break
                if free + reusable < backend.pool_min_free_vram_mib:
                    blocked[model_id] = f"GPU {gpu} has insufficient available VRAM"
                    break
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
            blocked[model_id] = "cannot verify worker GPU placement or ownership"
    return blocked


def release_pool_worker(handle: TextIO) -> None:
    """Clear advisory ownership while still locked, then release the lease.

    Lock-file existence or a live recorded PID does not establish ownership:
    only flock does. Keep the inode in place so waiters cannot lock a second
    file for the same worker.
    """
    if handle.closed:
        return
    try:
        handle.seek(0)
        handle.truncate()
        handle.flush()
    finally:
        handle.close()


def acquire_pool_worker(
    backend: Backend, runtime_dir: Path, instance: str | None
) -> tuple[TextIO, str]:
    pool_dir = runtime_dir / "backend-pools" / backend.id
    pool_dir.mkdir(parents=True, exist_ok=True)
    blocked = _worker_gpu_blockers(backend)
    for model_id in backend.pool_models:
        if model_id in blocked:
            continue
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", model_id).strip("-")
        suffix = hashlib.sha256(model_id.encode("utf-8")).hexdigest()[:12]
        handle = (pool_dir / f"{safe_name or 'worker'}-{suffix}.lock").open("a+")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            continue
        try:
            handle.seek(0)
            handle.truncate()
            json.dump({"pid": os.getpid(), "instance": instance, "model": model_id,
                       "started_at": int(time.time())}, handle)
            handle.flush()
        except BaseException:
            handle.close()
            raise
        return handle, model_id
    raise RuntimeError(
        "No free local worker. Workers are already assigned to local Marathon sessions "
        "or their GPUs are unavailable. "
        + "; ".join(f"{worker}: {reason}" for worker, reason in blocked.items())
    )
