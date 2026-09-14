"""Process-owned reservations for the broker's exclusive local workers."""

import fcntl
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import TextIO

from .catalog import Backend


def acquire_pool_worker(
    backend: Backend, runtime_dir: Path, instance: str | None
) -> tuple[TextIO, str]:
    pool_dir = runtime_dir / "backend-pools" / backend.id
    pool_dir.mkdir(parents=True, exist_ok=True)
    for model_id in backend.pool_models:
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
        f"No free local worker. All {len(backend.pool_models)} workers are already assigned "
        "to local Marathon sessions. Choose an external model or retry when a worker is free."
    )
