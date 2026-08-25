"""Persistent model roots and safe Hugging Face GGUF downloads."""

from __future__ import annotations

import functools
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path


RECOMMENDED_QWEN_REPOSITORY = "unsloth/Qwen3.8-27B-GGUF"
GGUF_METADATA_CACHE_SCHEMA = 1
GGUF_METADATA_CACHE_MAX_ENTRIES = 512


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


@dataclass(frozen=True)
class GgufMetadata:
    """Small, launch-relevant subset of a model's embedded GGUF metadata."""

    architecture: str | None = None
    name: str | None = None
    context_length: int | None = None


def _gguf_field_value(reader: object, key: str) -> object | None:
    fields = getattr(reader, "fields", {})
    field = fields.get(key) if isinstance(fields, dict) else None
    if field is None:
        return None
    return field.contents()


@functools.lru_cache(maxsize=256)
def _inspect_gguf_metadata_cached(
    resolved_path: str,
    mtime_ns: int,
    size_bytes: int,
) -> GgufMetadata:
    del mtime_ns, size_bytes
    try:
        from gguf import GGUFReader

        reader = GGUFReader(resolved_path)
        architecture_raw = _gguf_field_value(reader, "general.architecture")
        architecture = (
            str(architecture_raw).strip() if architecture_raw is not None else None
        )
        name = None
        for key in ("general.name", "general.basename"):
            value = _gguf_field_value(reader, key)
            if value is not None and str(value).strip():
                name = str(value).strip()
                break
        context_raw = (
            _gguf_field_value(reader, f"{architecture}.context_length")
            if architecture
            else None
        )
        context_length = (
            int(context_raw)
            if isinstance(context_raw, int) and context_raw > 0
            else None
        )
        return GgufMetadata(architecture, name, context_length)
    except (
        EOFError,
        ImportError,
        IndexError,
        KeyError,
        OSError,
        OverflowError,
        TypeError,
        ValueError,
    ):
        return GgufMetadata()


def gguf_metadata_cache_file() -> Path:
    configured = os.environ.get("MARATHON_GGUF_METADATA_CACHE")
    if configured:
        return Path(configured).expanduser()
    cache_home = os.environ.get("XDG_CACHE_HOME")
    root = Path(cache_home).expanduser() if cache_home else Path.home() / ".cache"
    return root / "marathon" / "gguf-metadata.json"


def _load_gguf_metadata_cache(path: Path) -> dict[str, dict[str, object]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict) or payload.get("schema") != GGUF_METADATA_CACHE_SCHEMA:
        return {}
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        return {}
    return {
        key: value
        for key, value in entries.items()
        if isinstance(key, str) and isinstance(value, dict)
    }


def _cached_gguf_metadata(
    entry: dict[str, object] | None,
    *,
    mtime_ns: int,
    size_bytes: int,
) -> GgufMetadata | None:
    if (
        entry is None
        or entry.get("mtime_ns") != mtime_ns
        or entry.get("size_bytes") != size_bytes
    ):
        return None
    architecture = entry.get("architecture")
    name = entry.get("name")
    context_length = entry.get("context_length")
    if architecture is not None and not isinstance(architecture, str):
        return None
    if name is not None and not isinstance(name, str):
        return None
    if context_length is not None and (
        not isinstance(context_length, int)
        or isinstance(context_length, bool)
        or context_length <= 0
    ):
        return None
    return GgufMetadata(architecture, name, context_length)


def _save_gguf_metadata_cache(
    path: Path,
    entries: dict[str, dict[str, object]],
) -> None:
    if len(entries) > GGUF_METADATA_CACHE_MAX_ENTRIES:
        def checked_ns(item: tuple[str, dict[str, object]]) -> int:
            value = item[1].get("checked_ns")
            return value if isinstance(value, int) and not isinstance(value, bool) else 0

        newest = sorted(
            entries.items(),
            key=checked_ns,
            reverse=True,
        )[:GGUF_METADATA_CACHE_MAX_ENTRIES]
        entries = dict(newest)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(
            json.dumps(
                {"schema": GGUF_METADATA_CACHE_SCHEMA, "entries": entries},
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except OSError:
        pass
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass


def read_gguf_metadata(
    path: Path,
    *,
    mtime_ns: int | None = None,
    size_bytes: int | None = None,
) -> GgufMetadata:
    """Read embedded model identity without loading tensor data into memory."""

    resolved = path.expanduser().resolve(strict=False)
    if mtime_ns is None or size_bytes is None:
        try:
            stat = resolved.stat()
        except OSError:
            return GgufMetadata()
        mtime_ns = stat.st_mtime_ns
        size_bytes = stat.st_size
    cache_path = gguf_metadata_cache_file()
    entries = _load_gguf_metadata_cache(cache_path)
    cache_key = str(resolved)
    cached = _cached_gguf_metadata(
        entries.get(cache_key),
        mtime_ns=mtime_ns,
        size_bytes=size_bytes,
    )
    if cached is not None:
        return cached

    metadata = _inspect_gguf_metadata_cached(cache_key, mtime_ns, size_bytes)
    entries[cache_key] = {
        "mtime_ns": mtime_ns,
        "size_bytes": size_bytes,
        "architecture": metadata.architecture,
        "name": metadata.name,
        "context_length": metadata.context_length,
        "checked_ns": time.time_ns(),
    }
    _save_gguf_metadata_cache(cache_path, entries)
    return metadata


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
