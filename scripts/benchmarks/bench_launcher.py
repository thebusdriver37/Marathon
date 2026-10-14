#!/usr/bin/env python3
"""Measure launcher imports and optional read-only probes on the current machine.

For real terminal readiness, model replies, and model-picker HTTP requests, use
scripts/evals/startup.py --run-gpu --catalog-requests 30 as well.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def samples(action, count):
    values = []
    for _ in range(count):
        start = time.perf_counter()
        action()
        values.append((time.perf_counter() - start) * 1000)
    return {"median_ms": statistics.median(values), "samples_ms": values}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--gpu-probes", action="store_true",
                        help="query real NVIDIA/Docker availability without leasing or starting workers")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    imports = []
    for _ in range(args.repeats):
        result = subprocess.run(
            [sys.executable, "-c", "import time; t=time.perf_counter(); "
             "import marathon_app.__main__; print((time.perf_counter()-t)*1000)"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        imports.append(float(result.stdout.strip()))
    report = {"launcher_import": {"median_ms": statistics.median(imports), "samples_ms": imports}}
    from marathon_app.catalog import backends
    from marathon_app.model_identity import pool_identity
    from marathon_app.pool import _worker_gpu_blockers

    for backend in backends().values():
        if backend.identity_config is None:
            continue
        # Warm the identity once, as startup does before opening the model picker.
        expected = pool_identity(backend)
        if expected is None or "identity unavailable" in expected[0]:
            raise RuntimeError(f"Cannot verify identity for {backend.id}")

        def identity():
            if pool_identity(backend) != expected:
                raise RuntimeError("Model identity changed during benchmark")

        def availability():
            blocked = _worker_gpu_blockers(backend)
            if any("cannot verify" in reason for reason in blocked.values()):
                raise RuntimeError("GPU availability probe failed during benchmark")

        row = {"identity": samples(identity, args.repeats)}
        if args.gpu_probes:
            row["gpu_availability"] = samples(availability, args.repeats)
        report[backend.id] = row
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
