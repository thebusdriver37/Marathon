"""Model discovery and local-backend launch-profile construction."""

from __future__ import annotations

import copy
import functools
import os
import re
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 compatibility.
    import tomli as tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .model_library import (
    configured_model_roots,
    find_multimodal_projector,
    is_model_sidecar,
    quant_from_filename,
    read_gguf_metadata,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
CATALOG_PATH = Path(
    os.environ.get("MARATHON_CATALOG", ROOT_DIR / "config" / "runtime_catalog.toml")
).expanduser()
def user_catalog_path() -> Path:
    """Return the active machine-local catalog path.

    Resolve this at call time so tests and launchers can set
    ``MARATHON_USER_CATALOG`` after importing Marathon.
    """

    configured = os.environ.get("MARATHON_USER_CATALOG")
    if configured:
        return Path(configured).expanduser()
    return Path(
        os.environ.get(
            "XDG_CONFIG_HOME",
            os.path.join(os.path.expanduser("~"), ".config"),
        )
    ).expanduser() / "marathon" / "catalog.toml"


@dataclass(frozen=True)
class Settings:
    ai_root: Path
    model_root: Path
    model_roots: tuple[Path, ...]
    llama_host: str
    llama_port: int
    router_host: str
    router_port: int
    health_timeout: int
    stop_timeout: int
    prompt_cache_ram_mib: int
    slot_snapshots_enabled: bool
    slot_snapshot_max_count: int
    slot_snapshot_max_bytes: int
    slot_snapshot_clean_startup: bool


@dataclass(frozen=True)
class Backend:
    id: str
    display_name: str
    server: Path
    kind: str = "llama_cpp"
    worker: Path | None = None
    model_alias: str = ""
    layer_slices: tuple[str, ...] = ()
    gpu_ids: tuple[int, ...] = ()
    environment: tuple[tuple[str, str], ...] = ()
    supports_slots: bool = True


@dataclass(frozen=True)
class Profile:
    id: str
    display_name: str
    description: str
    context: int
    batch: int
    ubatch: int
    parallel: int
    gpu_layers: str
    split_mode: str
    tensor_split: str
    main_gpu: int
    cache_k: str
    cache_v: str
    flash_attention: str
    extra_args: tuple[str, ...]
    confidence: str
    frontends: tuple[str, ...]
    tool_thinking_budget: int | None = None
    parallel_tool_calls: bool = False
    backend: str | None = None
    temperature: float | None = None
    gpus: tuple[int, ...] = ()

    def supports(self, frontend: str) -> bool:
        return frontend in self.frontends


@dataclass(frozen=True)
class ReasoningLevel:
    effort: str
    description: str


@dataclass(frozen=True)
class Family:
    id: str
    display_name: str
    patterns: tuple[str, ...]
    backend: str
    default_profile: str
    profiles: tuple[Profile, ...]
    reasoning_levels: tuple[ReasoningLevel, ...] = ()
    default_reasoning_level: str | None = None


@dataclass(frozen=True)
class Model:
    id: str
    display_name: str
    path: Path
    size_bytes: int
    family: Family
    quant: str
    multimodal_projector: Path | None = None
    architecture: str | None = None
    native_context: int | None = None

    @property
    def alias(self) -> str:
        return self.id


@dataclass(frozen=True)
class ExternalModel:
    """An optional OpenAI-compatible model configured outside the repository."""

    id: str
    model: str
    display_name: str
    description: str
    base_url: str
    context: int
    auto_compact_token_limit: int
    truncation_limit: int
    api_key_env: str | None = None
    api_key_file: str | None = None
    supports_parallel_tool_calls: bool = False
    temperature: float | None = None
    input_modalities: tuple[str, ...] = ("text",)


def _merge_catalog(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge a user catalog over the base, keeping settings, backends, and
    families keyed by id while appending user-defined profiles to families."""

    merged = dict(base)
    for key, value in override.items():
        if key in {"backends", "families"} and isinstance(merged.get(key), list) and isinstance(value, list):
            merged[key] = _merge_keyed_list(merged[key], value)
        elif key in {"settings",} and isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def _merge_keyed_list(base_items: list[dict[str, Any]], override_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = [dict(item) for item in base_items]
    for override_item in override_items:
        item_id = override_item.get("id")
        for position, existing in enumerate(result):
            if existing.get("id") == item_id:
                merged_item = {**existing, **override_item}
                for table in ("profiles",):
                    if isinstance(existing.get(table), list) and isinstance(override_item.get(table), list):
                        merged_item[table] = _merge_keyed_list(existing[table], override_item[table])
                result[position] = merged_item
                break
        else:
            result.append(dict(override_item))
    return result


def _catalog_file_revision(path: Path) -> tuple[str, int | None, int | None]:
    resolved = path.expanduser().resolve()
    try:
        stat = resolved.stat()
    except OSError:
        return str(resolved), None, None
    return str(resolved), stat.st_mtime_ns, stat.st_size


@functools.lru_cache(maxsize=16)
def _load_catalog_cached(
    base_revision: tuple[str, int | None, int | None],
    user_revision: tuple[str, int | None, int | None] | None,
) -> dict[str, Any]:
    base = Path(base_revision[0])
    with base.open("rb") as handle:
        merged = tomllib.load(handle)
    if user_revision is not None and user_revision[1] is not None:
        local_catalog = Path(user_revision[0])
        with local_catalog.open("rb") as handle:
            merged = _merge_catalog(merged, tomllib.load(handle))
    return merged


def _catalog_snapshot(path: Path | None = None) -> dict[str, Any]:
    base = CATALOG_PATH if path is None else path
    user_revision = (
        _catalog_file_revision(user_catalog_path()) if path is None else None
    )
    return _load_catalog_cached(_catalog_file_revision(base), user_revision)


def load_catalog(path: Path | None = None) -> dict[str, Any]:
    """Load a catalog, reparsing only after either source file changes."""

    return copy.deepcopy(_catalog_snapshot(path))


def _external_context_defaults(context: int) -> tuple[int, int]:
    reserve = min(context // 2, max(12_288, min(32_768, context // 8)))
    auto_compact = max(1, context - reserve)
    guard = max(2_048, min(8_192, context // 20))
    return auto_compact, max(1, auto_compact - guard)


def external_models(catalog: dict[str, Any] | None = None) -> tuple[ExternalModel, ...]:
    """Load optional OpenAI-compatible models from the merged user catalog."""

    loaded = _catalog_snapshot() if catalog is None else catalog
    raw_models = loaded.get("external_models", [])
    if not isinstance(raw_models, list) or any(
        not isinstance(item, dict) for item in raw_models
    ):
        raise ValueError("external_models must be a list of tables")

    result: list[ExternalModel] = []
    seen: set[str] = set()
    for raw in raw_models:
        if raw.get("enabled", True) is False:
            continue
        model_id = str(raw.get("id") or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", model_id):
            raise ValueError(
                "external model id may only contain letters, numbers, '.', '_', and '-'"
            )
        if model_id in seen:
            raise ValueError(f"duplicate external model id: {model_id}")
        seen.add(model_id)

        upstream_model = str(raw.get("model") or "").strip()
        if not upstream_model:
            raise ValueError(f"external model {model_id} requires model")
        base_url = str(raw.get("base_url") or "").strip().rstrip("/")
        parsed_url = urlparse(base_url)
        if (
            parsed_url.scheme not in {"http", "https"}
            or not parsed_url.netloc
            or parsed_url.username is not None
            or parsed_url.password is not None
            or parsed_url.query
            or parsed_url.fragment
        ):
            raise ValueError(
                f"external model {model_id} requires an http(s) base_url "
                "without embedded credentials, query, or fragment"
            )

        try:
            context = int(raw["context"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"external model {model_id} requires a positive integer context"
            ) from exc
        if context <= 0:
            raise ValueError(
                f"external model {model_id} requires a positive integer context"
            )
        default_auto_compact, default_truncation = _external_context_defaults(context)
        auto_compact = int(
            raw.get("auto_compact_token_limit", default_auto_compact)
        )
        truncation = int(raw.get("truncation_limit", default_truncation))
        if not 0 < truncation <= auto_compact <= context:
            raise ValueError(
                f"external model {model_id} requires 0 < truncation_limit "
                "<= auto_compact_token_limit <= context"
            )

        api_key_env_raw = raw.get("api_key_env")
        api_key_env = (
            str(api_key_env_raw).strip() if api_key_env_raw is not None else None
        )
        if api_key_env and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", api_key_env):
            raise ValueError(
                f"external model {model_id} has an invalid api_key_env name"
            )
        api_key_file_raw = raw.get("api_key_file")
        api_key_file = (
            str(Path(str(api_key_file_raw)).expanduser())
            if api_key_file_raw is not None
            else None
        )

        raw_modalities = raw.get("input_modalities", ["text"])
        if not isinstance(raw_modalities, list):
            raise ValueError(
                f"external model {model_id} input_modalities must be a list"
            )
        modalities = tuple(
            dict.fromkeys(str(value).strip().lower() for value in raw_modalities)
        )
        if not modalities or any(value not in {"text", "image"} for value in modalities):
            raise ValueError(
                f"external model {model_id} input_modalities may contain only text and image"
            )

        temperature_raw = raw.get("temperature")
        result.append(
            ExternalModel(
                id=model_id,
                model=upstream_model,
                display_name=str(raw.get("display_name") or model_id).strip(),
                description=str(
                    raw.get("description")
                    or "OpenAI-compatible external model"
                ).strip(),
                base_url=base_url,
                context=context,
                auto_compact_token_limit=auto_compact,
                truncation_limit=truncation,
                api_key_env=api_key_env or None,
                api_key_file=api_key_file,
                supports_parallel_tool_calls=bool(
                    raw.get("supports_parallel_tool_calls", False)
                ),
                temperature=(
                    float(temperature_raw)
                    if temperature_raw is not None
                    else None
                ),
                input_modalities=modalities,
            )
        )
    return tuple(result)


def _resolve_ai_path(value: str | os.PathLike[str], ai_root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ai_root / path


def _setting_bool(environment_name: str, default: object) -> bool:
    value = os.environ.get(environment_name)
    if value is None:
        return bool(default)
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{environment_name} must be true or false")


def settings(catalog: dict[str, Any] | None = None) -> Settings:
    raw = (catalog or load_catalog())["settings"]
    ai_root = Path(
        os.environ.get("MARATHON_AI_ROOT", raw.get("ai_root", "~/AI"))
    ).expanduser()
    configured_model_root = os.environ.get("MARATHON_MODELS_DIR")
    model_root = _resolve_ai_path(configured_model_root or raw["model_root"], ai_root)
    return Settings(
        ai_root=ai_root,
        model_root=model_root,
        model_roots=configured_model_roots(model_root),
        llama_host=os.environ.get("MARATHON_LLAMA_HOST", raw["llama_host"]),
        llama_port=int(os.environ.get("MARATHON_LLAMA_PORT", raw["llama_port"])),
        router_host=os.environ.get("MARATHON_PROXY_HOST", raw["router_host"]),
        router_port=int(os.environ.get("MARATHON_PROXY_PORT", raw["router_port"])),
        health_timeout=int(
            os.environ.get(
                "MARATHON_BACKEND_START_TIMEOUT_SECONDS", raw["health_timeout"]
            )
        ),
        stop_timeout=int(raw["stop_timeout"]),
        prompt_cache_ram_mib=int(
            os.environ.get(
                "MARATHON_PROMPT_CACHE_RAM_MIB",
                raw.get("prompt_cache_ram_mib", 8192),
            )
        ),
        slot_snapshots_enabled=_setting_bool(
            "MARATHON_SLOT_SNAPSHOTS_ENABLED",
            raw.get("slot_snapshots_enabled", False),
        ),
        slot_snapshot_max_count=max(
            0,
            int(
                os.environ.get(
                    "MARATHON_SLOT_SNAPSHOT_MAX_COUNT",
                    raw.get("slot_snapshot_max_count", 16),
                )
            ),
        ),
        slot_snapshot_max_bytes=max(
            0,
            int(
                os.environ.get(
                    "MARATHON_SLOT_SNAPSHOT_MAX_BYTES",
                    raw.get("slot_snapshot_max_bytes", 32 * 1024**3),
                )
            ),
        ),
        slot_snapshot_clean_startup=_setting_bool(
            "MARATHON_SLOT_SNAPSHOT_CLEAN_STARTUP",
            raw.get("slot_snapshot_clean_startup", False),
        ),
    )


def backends(catalog: dict[str, Any] | None = None) -> dict[str, Backend]:
    loaded = catalog or load_catalog()
    ai_root = settings(loaded).ai_root
    result: dict[str, Backend] = {}
    for raw in loaded.get("backends", []):
        env_name = f"MARATHON_BACKEND_{raw['id'].upper().replace('-', '_')}"
        configured = os.environ.get(env_name)
        if raw["id"] == "upstream":
            configured = os.environ.get("LLAMACPP_BIN", configured)
        server = _resolve_ai_path(configured or raw["server"], ai_root)
        worker_value = raw.get("worker")
        worker_env = os.environ.get(f"{env_name}_WORKER")
        worker = (
            _resolve_ai_path(worker_env or worker_value, ai_root)
            if worker_env or worker_value
            else None
        )
        result[raw["id"]] = Backend(
            id=raw["id"],
            display_name=raw["display_name"],
            server=server,
            kind=str(raw.get("kind", "llama_cpp")),
            worker=worker,
            model_alias=str(raw.get("model_alias", "")),
            layer_slices=tuple(str(value) for value in raw.get("layer_slices", [])),
            gpu_ids=tuple(int(value) for value in raw.get("gpu_ids", [])),
            environment=tuple(
                (str(key), str(value))
                for key, value in raw.get("environment", {}).items()
            ),
            supports_slots=bool(raw.get("supports_slots", True)),
        )
    return result


def families(catalog: dict[str, Any] | None = None) -> tuple[Family, ...]:
    result: list[Family] = []
    for raw in (catalog or load_catalog()).get("families", []):
        raw_reasoning_levels = raw.get("reasoning_levels", [])
        if not isinstance(raw_reasoning_levels, list) or any(
            not isinstance(item, dict) for item in raw_reasoning_levels
        ):
            raise ValueError(
                f"family {raw['id']} reasoning_levels must be a list of tables"
            )
        reasoning_levels = tuple(
            ReasoningLevel(
                effort=str(item["effort"]).strip(),
                description=str(item.get("description", "")).strip(),
            )
            for item in raw_reasoning_levels
        )
        efforts = [level.effort for level in reasoning_levels]
        if any(not effort for effort in efforts):
            raise ValueError(f"family {raw['id']} has an empty reasoning effort")
        if len(efforts) != len(set(efforts)):
            raise ValueError(f"family {raw['id']} has duplicate reasoning efforts")
        default_reasoning_level = raw.get("default_reasoning_level")
        if default_reasoning_level is not None:
            default_reasoning_level = str(default_reasoning_level).strip()
            if default_reasoning_level not in efforts:
                raise ValueError(
                    f"family {raw['id']} default reasoning effort "
                    f"{default_reasoning_level!r} is not supported"
                )
        elif reasoning_levels:
            raise ValueError(
                f"family {raw['id']} has reasoning levels without a default"
            )
        profiles = tuple(
            Profile(
                id=item["id"],
                display_name=item["display_name"],
                description=item.get("description", ""),
                context=int(item["context"]),
                batch=int(item.get("batch", 2048)),
                ubatch=int(item.get("ubatch", 512)),
                parallel=int(item.get("parallel", 1)),
                gpu_layers=str(item.get("gpu_layers", "999")),
                split_mode=item.get("split_mode", "layer"),
                tensor_split=item.get("tensor_split", ""),
                main_gpu=int(item.get("main_gpu", 0)),
                cache_k=item.get("cache_k", "f16"),
                cache_v=item.get("cache_v", "f16"),
                flash_attention=item.get("flash_attention", "on"),
                extra_args=tuple(str(arg) for arg in item.get("extra_args", [])),
                confidence=item.get("confidence", "baseline"),
                frontends=tuple(item.get("frontends", ["direct"])),
                tool_thinking_budget=(
                    max(0, int(item["tool_thinking_budget"]))
                    if "tool_thinking_budget" in item
                    else None
                ),
                parallel_tool_calls=bool(item.get("parallel_tool_calls", False)),
                backend=(str(item["backend"]) if item.get("backend") else None),
                temperature=(
                    float(item["temperature"])
                    if item.get("temperature") is not None
                    else None
                ),
                gpus=tuple(int(gpu) for gpu in item.get("gpus", [])),
            )
            for item in raw.get("profiles", [])
        )
        result.append(
            Family(
                id=raw["id"],
                display_name=raw["display_name"],
                patterns=tuple(pattern.lower() for pattern in raw.get("patterns", [])),
                backend=raw["backend"],
                default_profile=raw["default_profile"],
                profiles=profiles,
                reasoning_levels=reasoning_levels,
                default_reasoning_level=default_reasoning_level,
            )
        )
    return tuple(result)


def _first_shard(path: Path) -> bool:
    match = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", path.name, re.IGNORECASE)
    return not match or match.group(1) == "00001"


def _model_sidecar(path: Path) -> bool:
    """Keep speculative draft and multimodal helper weights out of the picker."""

    return is_model_sidecar(path.name)


def _family_for(
    path: Path,
    known: tuple[Family, ...],
    *,
    metadata_name: str | None = None,
    architecture: str | None = None,
) -> Family:
    value = " ".join(
        part for part in (str(path), metadata_name, architecture) if part
    ).lower()
    for family in known:
        if family.id != "generic" and any(pattern in value for pattern in family.patterns):
            return family
    return next(family for family in known if family.id == "generic")


def _quant(name: str) -> str:
    return quant_from_filename(name)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _model_size(path: Path, discovered_sizes: dict[Path, int]) -> int:
    match = re.search(r"-00001-of-(\d{5})\.gguf$", path.name, re.IGNORECASE)
    if not match:
        return discovered_sizes[path]
    stem = re.sub(r"-00001-of-\d{5}\.gguf$", "", path.name, flags=re.IGNORECASE)
    pattern = re.compile(rf"^{re.escape(stem)}-\d{{5}}-of-\d{{5}}\.gguf$", re.IGNORECASE)
    return sum(
        size
        for shard, size in discovered_sizes.items()
        if shard.parent == path.parent and pattern.match(shard.name)
    )


def discover_models(model_root: Path | None = None) -> list[Model]:
    roots = (model_root,) if model_root is not None else settings().model_roots
    known = families()
    result: list[Model] = []
    ids: dict[str, int] = {}
    paths: list[Path] = []
    discovered_sizes: dict[Path, int] = {}
    discovered_mtimes: dict[Path, int] = {}
    seen_files: set[tuple[int, int]] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.gguf"):
            try:
                stat = path.stat()
                identity = (stat.st_dev, stat.st_ino)
            except OSError:
                continue
            if identity in seen_files:
                continue
            seen_files.add(identity)
            paths.append(path)
            discovered_sizes[path] = stat.st_size
            discovered_mtimes[path] = stat.st_mtime_ns
    for path in sorted(paths):
        if ".cache" in path.parts or path.name.lower().startswith("mmproj"):
            continue
        if _model_sidecar(path) or not _first_shard(path) or discovered_sizes[path] == 0:
            continue
        metadata = read_gguf_metadata(
            path,
            mtime_ns=discovered_mtimes[path],
            size_bytes=discovered_sizes[path],
        )
        family = _family_for(
            path,
            known,
            metadata_name=metadata.name,
            architecture=metadata.architecture,
        )
        quant = _quant(path.name)
        if family.id == "generic":
            base = re.sub(r"-\d{5}-of-\d{5}\.gguf$", "", path.name, flags=re.IGNORECASE)
            base = re.sub(r"\.gguf$", "", base, flags=re.IGNORECASE)
            model_id = _slug(base)
            display = metadata.name or base
        else:
            model_id = f"{family.id}-{_slug(quant)}"
            display = f"{family.display_name} {quant}"
        ids[model_id] = ids.get(model_id, 0) + 1
        if ids[model_id] > 1:
            model_id = f"{model_id}-{ids[model_id]}"
        result.append(
            Model(
                model_id,
                display,
                path,
                _model_size(path, discovered_sizes),
                family,
                quant,
                find_multimodal_projector(path),
                metadata.architecture,
                metadata.context_length,
            )
        )
    return sorted(result, key=lambda model: (model.family.id, model.display_name.lower()))


def profiles_for_model(model: Model) -> tuple[Profile, ...]:
    """Return shipped profiles followed by valid machine-local Dyno profiles."""

    # Keep the catalog independent from the optional tuning runner at import
    # time. dyno imports these dataclasses, so a local import avoids a cycle.
    try:
        from .dyno import load_tuned_profiles

        tuned = load_tuned_profiles(model)
    except (KeyError, OSError, ValueError, TypeError):
        tuned = ()
    shipped_ids = {profile.id for profile in model.family.profiles}
    return model.family.profiles + tuple(
        profile for profile in tuned if profile.id not in shipped_ids
    )


def find_model(query: str, models: list[Model] | None = None) -> Model:
    available = models if models is not None else discover_models()
    exact = [model for model in available if model.id == query]
    if exact:
        return exact[0]
    lowered = query.lower()
    matches = [
        model for model in available
        if lowered in model.id.lower() or lowered in model.display_name.lower()
    ]
    if not matches:
        raise ValueError(f"no installed model matches: {query}")
    if len(matches) > 1:
        raise ValueError(f"ambiguous model '{query}': {', '.join(model.id for model in matches)}")
    return matches[0]


def find_profile(model: Model, profile_id: str | None, frontend: str | None = None) -> Profile:
    profile_id = profile_id or model.family.default_profile
    available = profiles_for_model(model)
    for profile in available:
        if profile.id == profile_id:
            if frontend and not profile.supports(frontend):
                raise ValueError(
                    f"profile '{profile.id}' is not compatible with {frontend}; "
                    f"choose one of: {', '.join(p.id for p in available if p.supports(frontend))}"
                )
            return profile
    raise ValueError(
        f"unknown profile '{profile_id}' for {model.id}; "
        f"choose: {', '.join(profile.id for profile in available)}"
    )


def backend_for(model: Model, profile: Profile | None = None) -> Backend:
    backend_id = profile.backend if profile and profile.backend else model.family.backend
    backend = backends().get(backend_id)
    if backend is None:
        raise ValueError(f"backend '{backend_id}' is not configured")
    if not backend.server.is_file() or not os.access(backend.server, os.X_OK):
        raise ValueError(f"backend server is missing or not executable: {backend.server}")
    if backend.kind == "ds4_distributed" and (
        backend.worker is None
        or not backend.worker.is_file()
        or not os.access(backend.worker, os.X_OK)
    ):
        raise ValueError(
            f"DS4 worker is missing or not executable: {backend.worker or '(not configured)'}"
        )
    for key, path in backend_files(model, backend).items():
        if not path.is_file():
            raise ValueError(f"backend file from {key} is missing: {path}")
    return backend


def backend_environment(model: Model, backend: Backend) -> dict[str, str]:
    """Expand portable catalog values, while preserving explicit user overrides."""

    configured = settings()
    placeholders = {
        "{ai_root}": str(configured.ai_root),
        "{model_dir}": str(model.path.parent),
        "{model_path}": str(model.path),
    }
    result: dict[str, str] = {}
    for key, default in backend.environment:
        value = os.environ.get(key, default)
        for placeholder, replacement in placeholders.items():
            value = value.replace(placeholder, replacement)
        result[key] = os.path.expanduser(value)
    return result


def backend_files(model: Model, backend: Backend) -> dict[str, Path]:
    """Return catalog environment values that name required GGUF sidecars."""

    return {
        key: Path(value).expanduser()
        for key, value in backend_environment(model, backend).items()
        if key.endswith("_GGUF")
    }


def server_command(model: Model, profile: Profile, backend: Backend | None = None) -> list[str]:
    cfg = settings()
    selected = backend or backend_for(model, profile)
    if selected.kind != "llama_cpp":
        raise ValueError(
            f"backend '{selected.id}' uses {selected.kind}; it requires Marathon's "
            "multi-process runtime instead of a llama-server command"
        )
    command = [
        str(selected.server), "--model", str(model.path), "--alias", model.alias,
        "--host", cfg.llama_host, "--port", str(cfg.llama_port),
        "--ctx-size", str(profile.context), "--parallel", str(profile.parallel),
        "--n-gpu-layers", profile.gpu_layers, "--split-mode", profile.split_mode,
        "--batch-size", str(profile.batch), "--ubatch-size", str(profile.ubatch),
        "--cache-type-k", profile.cache_k, "--cache-type-v", profile.cache_v,
        "--flash-attn", profile.flash_attention, "--jinja", "--metrics",
        "--cache-prompt", "--cache-idle-slots", "--cache-ram",
        str(cfg.prompt_cache_ram_mib),
    ]
    if model.multimodal_projector is not None:
        command.extend(["--mmproj", str(model.multimodal_projector)])
    if profile.split_mode == "none":
        command.extend(["--main-gpu", str(profile.main_gpu)])
    elif profile.tensor_split:
        command.extend(["--tensor-split", profile.tensor_split])
    if profile.temperature is not None:
        command.extend(["--temp", str(profile.temperature)])
    command.extend(profile.extra_args)
    return command


def format_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"
