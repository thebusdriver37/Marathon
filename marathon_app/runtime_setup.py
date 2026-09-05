"""Optional, revision-pinned model bundles without machine-local dependencies."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

if TYPE_CHECKING:
    from .catalog import Model, Profile

MANIFEST = Path(__file__).resolve().parents[1] / "config" / "model_bundles.toml"


def missing_runtime_tools(frontend: str) -> tuple[str, ...]:
    if frontend == "codex" and sys.platform.startswith("linux") and not shutil.which("bwrap"):
        return ("bubblewrap (Linux sandbox)",)
    return ()


def missing_build_tools(*, runtime_installed: bool, frontend_installed: bool, cuda: bool | None = None) -> tuple[str, ...]:
    """Fail before a large bundle download when source prerequisites are absent."""
    required = []
    if not runtime_installed:
        required.extend(("git", "cmake"))
        if cuda is None:
            mode = os.environ.get("MARATHON_GPU_BACKEND", "auto")
            cuda = mode == "cuda" or (mode == "auto" and bool(shutil.which("nvidia-smi") or shutil.which("nvcc")))
        if cuda and not shutil.which(os.environ.get("CUDA_COMPILER", "nvcc")):
            required.append("nvcc (CUDA toolkit)")
    if not frontend_installed:
        required.extend(("git", "cargo", "cmake"))
    missing = [tool for tool in required if shutil.which(tool) is None]
    if not runtime_installed or not frontend_installed:
        if not any(shutil.which(tool) for tool in ("c++", "g++", "clang++")):
            missing.append("C++ compiler")
        if not shutil.which("pkg-config"):
            missing.append("pkg-config and OpenSSL development libraries")
        elif subprocess.run(["pkg-config", "--exists", "openssl"], check=False).returncode:
            missing.append("OpenSSL development libraries (libssl-dev on Ubuntu)")
    return tuple(dict.fromkeys(missing))


def model_bundle(bundle_id: str) -> dict:
    with MANIFEST.open("rb") as handle:
        bundles = tomllib.load(handle)
    if bundle_id not in bundles:
        raise ValueError(f"Unknown model bundle: {bundle_id}")
    bundle = bundles[bundle_id]
    for asset in bundle["files"]:
        name = asset["filename"]
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("Bundle filenames must stay inside their model folder")
    return bundle


def bundle_matches_model(bundle_id: str, path: Path) -> bool:
    target = next(asset for asset in model_bundle(bundle_id)["files"] if asset["role"] == "model")
    return path.name == target["filename"]


@dataclass(frozen=True)
class AvailableGpu:
    index: int
    uuid: str
    name: str
    free_mib: int


def eligible_gpus(bundle_id: str) -> tuple[AvailableGpu, ...]:
    # This source-packaged runtime is tested on Linux CUDA, not Metal or ROCm.
    if not sys.platform.startswith("linux"):
        return ()
    bundle = model_bundle(bundle_id)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ()
    if result.returncode:
        return ()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    allowed = None if visible is None else {value.strip() for value in visible.split(",")}
    found = []
    for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if len(row) != 4:
            continue
        index, uuid, name, memory = [value.strip() for value in row]
        try:
            gpu = AvailableGpu(int(index), uuid, name, int(memory))
        except ValueError:
            continue
        if allowed is not None and index not in allowed and uuid not in allowed:
            continue
        if name in bundle["gpu_names"] and gpu.free_mib >= bundle["minimum_free_vram_mib"]:
            found.append(gpu)
    return tuple(sorted(found, key=lambda gpu: (-gpu.free_mib, gpu.index)))


def prepare_bundle_profile(model: Model, profile: Profile) -> Profile:
    """Validate the complete bundle and select one free, explicitly supported GPU."""
    if not profile.bundle:
        return profile
    bundle = model_bundle(profile.bundle)
    if not bundle_matches_model(profile.bundle, model.path):
        raise ValueError("This profile requires the exact model from its setup bundle")
    for asset in bundle["files"]:
        path = model.path.parent / asset["filename"]
        if not path.is_file() or path.stat().st_size != asset["size_bytes"]:
            raise ValueError(f"Bundle file is missing or incomplete: {path}. Run marathon setup.")
        if asset["role"] == "projector" and (
            model.multimodal_projector is None
            or model.multimodal_projector.resolve() != path.resolve()
        ):
            raise ValueError("The tuned profile requires its matching vision projector")
    candidates = eligible_gpus(profile.bundle)
    if profile.gpus:
        if len(profile.gpus) != 1:
            raise ValueError("This tuned profile requires exactly one GPU")
        candidates = tuple(gpu for gpu in candidates if gpu.index == profile.gpus[0])
    if not candidates:
        raise ValueError(
            f"This preset needs one supported, available GPU with at least "
            f"{bundle['minimum_free_vram_mib']:,} MiB free. "
            "Use the Automatic profile for other hardware or less free memory."
        )
    return replace(profile, gpus=(candidates[0].index,), main_gpu=0)


def download_bundle(bundle_id: str, model_root: Path) -> Path:
    """Download all roles at fixed revisions and verify their actual file hashes."""
    from huggingface_hub import hf_hub_download

    bundle = model_bundle(bundle_id)
    destination = model_root / bundle_id
    destination.mkdir(parents=True, exist_ok=True)
    required = sum(
        asset["size_bytes"] for asset in bundle["files"]
        if not (destination / asset["filename"]).is_file()
        or (destination / asset["filename"]).stat().st_size != asset["size_bytes"]
    ) + 2 * 1024**3
    if shutil.disk_usage(destination).free < required:
        raise ValueError(f"Setup needs at least {required / 1024**3:.1f} GiB free for these downloads")
    target = None
    for asset in bundle["files"]:
        downloaded = Path(hf_hub_download(
            repo_id=asset["repository"], revision=asset["revision"],
            filename=asset["filename"], local_dir=destination,
        ))
        digest = hashlib.sha256()
        with downloaded.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
                digest.update(chunk)
        if downloaded.stat().st_size != asset["size_bytes"] or digest.hexdigest() != asset["sha256"]:
            raise ValueError(f"Checksum verification failed: {downloaded}. Setup stopped; no model was launched.")
        downloaded.with_suffix(".gguf.marathon.json").write_text(
            json.dumps({"schema": 1, **asset}, indent=2) + "\n", encoding="utf-8",
        )
        if asset["role"] == "model":
            target = downloaded
    if target is None:
        raise ValueError("Bundle has no target model")
    return target
