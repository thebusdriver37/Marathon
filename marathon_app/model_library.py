"""Persistent model roots and safe Hugging Face GGUF downloads."""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


RECOMMENDED_QWEN_REPOSITORY = "unsloth/Qwen3.8-27B-GGUF"


def is_model_sidecar(filename: str) -> bool:
    """Return whether a GGUF is helper weights rather than a chat model."""

    lowered = Path(filename).name.lower()
    return is_multimodal_projector(lowered) or bool(
        re.search(
            r"(?:^|[-_.])(?:mtp|dflash2?|dspark|eagle3)(?:[-_.]|$)",
            lowered,
        )
    )


def is_multimodal_projector(filename: str) -> bool:
    """Return whether a GGUF is a multimodal projector sidecar."""

    lowered = Path(filename).name.lower()
    return lowered.startswith("mmproj") or bool(
        re.search(
            r"(?:^|[-_.])vision[-_.](?:bf16|f16|f32|q8_0)(?:[-_.]|$)",
            lowered,
        )
    )


def _projector_sort_key(filename: str) -> tuple[int, int, str]:
    """Prefer portable F16 projector weights, then other full-quality formats."""

    lowered = Path(filename).name.lower()
    format_rank = 0 if re.search(r"(?:^|[-_.])f16(?:[-_.]|$)", lowered) else 1
    named_rank = 0 if lowered.startswith("mmproj") else 1
    return format_rank, named_rank, lowered


def find_multimodal_projector(model_path: Path) -> Path | None:
    """Find the best projector stored beside a local language model."""

    try:
        candidates = [
            path
            for path in model_path.parent.iterdir()
            if path.is_file()
            and path.suffix.lower() == ".gguf"
            and is_multimodal_projector(path.name)
        ]
    except OSError:
        return None
    return min(candidates, key=lambda path: _projector_sort_key(path.name), default=None)


@dataclass(frozen=True)
class HuggingFaceGguf:
    repository: str
    revision: str
    filename: str
    size_bytes: int | None
    quant: str
    sha256: str | None = None
    mmproj_filename: str | None = None
    mmproj_size_bytes: int | None = None
    mmproj_sha256: str | None = None


def _config_dir() -> Path:
    configured = os.environ.get("XDG_CONFIG_HOME")
    root = Path(configured).expanduser() if configured else Path.home() / ".config"
    return root / "marathon"


def model_library_file() -> Path:
    configured = os.environ.get("MARATHON_MODEL_LIBRARY_FILE")
    return Path(configured).expanduser() if configured else _config_dir() / "models.json"


def _normalized(path: str | os.PathLike[str]) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def load_registered_model_roots(path: Path | None = None) -> tuple[Path, ...]:
    try:
        payload = json.loads((path or model_library_file()).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    roots = payload.get("roots") if isinstance(payload, dict) else None
    if not isinstance(roots, list):
        return ()
    result: list[Path] = []
    for value in roots:
        if not isinstance(value, str) or not value.strip():
            continue
        root = _normalized(value)
        if root not in result:
            result.append(root)
    return tuple(result)


def save_registered_model_roots(
    roots: tuple[Path, ...] | list[Path], path: Path | None = None
) -> None:
    destination = path or model_library_file()
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalized: list[str] = []
    for root in roots:
        value = str(_normalized(root))
        if value not in normalized:
            normalized.append(value)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"schema": 1, "roots": normalized}, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def register_model_root(root: Path, path: Path | None = None) -> Path:
    normalized = _normalized(root)
    if not normalized.is_dir():
        raise ValueError(f"model folder does not exist: {normalized}")
    roots = list(load_registered_model_roots(path))
    if normalized not in roots:
        roots.append(normalized)
        save_registered_model_roots(roots, path)
    return normalized


def configured_model_roots(default_root: Path) -> tuple[Path, ...]:
    explicit = os.environ.get("MARATHON_MODEL_DIRS")
    if explicit:
        values = [value for value in explicit.split(os.pathsep) if value]
        return tuple(dict.fromkeys(_normalized(value) for value in values))
    legacy = os.environ.get("MARATHON_MODELS_DIR")
    if legacy:
        return (_normalized(legacy),)
    roots = [_normalized(default_root), *load_registered_model_roots()]
    return tuple(dict.fromkeys(roots))


def quant_from_filename(filename: str) -> str:
    normalized = re.sub(
        r"-\d{5}-of-\d{5}\.gguf$", "", filename, flags=re.IGNORECASE
    )
    matches = re.findall(
        r"(?:UD-)?(?:IQ\d(?:_[A-Z0-9]+)+(?:-[A-Z0-9]+)?|"
        r"Q\d(?:_[A-Z0-9]+)+(?:-[A-Z0-9]+)?)",
        normalized.upper(),
    )
    return matches[-1] if matches else "GGUF"


def list_huggingface_ggufs(repository: str) -> list[HuggingFaceGguf]:
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise RuntimeError(
            "Hugging Face support is missing; rerun 'marathon setup-deps'"
        ) from error

    repository = repository.strip().strip("/")
    if repository.count("/") != 1 or any(char.isspace() for char in repository):
        raise ValueError("enter a Hugging Face repository as owner/name")
    info = HfApi().model_info(repository, files_metadata=True)
    revision = str(info.sha)
    siblings = list(info.siblings or [])
    projector_siblings = [
        sibling
        for sibling in siblings
        if str(sibling.rfilename).lower().endswith(".gguf")
        and is_multimodal_projector(str(sibling.rfilename))
    ]
    projector = min(
        projector_siblings,
        key=lambda item: _projector_sort_key(str(item.rfilename)),
        default=None,
    )
    projector_filename = str(projector.rfilename) if projector is not None else None
    projector_size = (
        projector.size
        if projector is not None and isinstance(projector.size, int)
        else None
    )
    projector_lfs = getattr(projector, "lfs", None)
    projector_sha256 = getattr(projector_lfs, "sha256", None)
    files: list[HuggingFaceGguf] = []
    for sibling in siblings:
        filename = str(sibling.rfilename)
        lowered = Path(filename).name.lower()
        if not lowered.endswith(".gguf"):
            continue
        if is_model_sidecar(lowered):
            continue
        if re.search(r"-\d{5}-of-\d{5}\.gguf$", lowered):
            continue
        size = sibling.size if isinstance(sibling.size, int) else None
        lfs = getattr(sibling, "lfs", None)
        sha256 = getattr(lfs, "sha256", None)
        files.append(
            HuggingFaceGguf(
                repository=repository,
                revision=revision,
                filename=filename,
                size_bytes=size,
                quant=quant_from_filename(filename),
                sha256=sha256 if isinstance(sha256, str) else None,
                mmproj_filename=projector_filename,
                mmproj_size_bytes=projector_size,
                mmproj_sha256=(
                    projector_sha256
                    if isinstance(projector_sha256, str)
                    else None
                ),
            )
        )

    preferred = {
        "Q4_K_M": 0,
        "UD-Q4_K_XL": 1,
        "Q5_K_M": 2,
        "Q6_K": 3,
        "Q8_0": 4,
        "UD-Q8_K_XL": 5,
    }
    return sorted(
        files,
        key=lambda item: (preferred.get(item.quant, 100), item.size_bytes or 0, item.filename),
    )


def download_huggingface_gguf(model: HuggingFaceGguf, model_root: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError(
            "Hugging Face support is missing; rerun 'marathon setup-deps'"
        ) from error

    owner, name = model.repository.split("/", 1)
    destination = _normalized(model_root) / f"{owner}--{name}"
    destination.mkdir(parents=True, exist_ok=True)
    if model.size_bytes is not None:
        available = shutil.disk_usage(destination).free
        required = model.size_bytes + (model.mmproj_size_bytes or 0) + 2 * 1024**3
        if available < required:
            raise RuntimeError(
                f"download needs about {required / 1024**3:.1f} GiB free; "
                f"only {available / 1024**3:.1f} GiB is available"
            )

    downloaded = Path(
        hf_hub_download(
            repo_id=model.repository,
            filename=model.filename,
            revision=model.revision,
            local_dir=destination,
        )
    ).resolve()
    if model.mmproj_filename:
        hf_hub_download(
            repo_id=model.repository,
            filename=model.mmproj_filename,
            revision=model.revision,
            local_dir=destination,
        )
    provenance = downloaded.with_suffix(downloaded.suffix + ".marathon.json")
    provenance.write_text(
        json.dumps(
            {
                "schema": 1,
                "repository": model.repository,
                "revision": model.revision,
                "filename": model.filename,
                "size_bytes": model.size_bytes,
                "sha256": model.sha256,
                "multimodal_projector": (
                    {
                        "filename": model.mmproj_filename,
                        "size_bytes": model.mmproj_size_bytes,
                        "sha256": model.mmproj_sha256,
                    }
                    if model.mmproj_filename
                    else None
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return downloaded
