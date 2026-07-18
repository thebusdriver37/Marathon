"""Model discovery and local-backend launch-profile construction."""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
CATALOG_PATH = Path(
    os.environ.get("MARATHON_CATALOG", ROOT_DIR / "config" / "runtime_catalog.toml")
).expanduser()


@dataclass(frozen=True)
class Settings:
    ai_root: Path
    model_root: Path
    llama_host: str
    llama_port: int
    router_host: str
    router_port: int
    health_timeout: int
    stop_timeout: int


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

    def supports(self, frontend: str) -> bool:
        return frontend in self.frontends


@dataclass(frozen=True)
class Family:
    id: str
    display_name: str
    patterns: tuple[str, ...]
    backend: str
    default_profile: str
    profiles: tuple[Profile, ...]


@dataclass(frozen=True)
class Model:
    id: str
    display_name: str
    path: Path
    size_bytes: int
    family: Family
    quant: str

    @property
    def alias(self) -> str:
        return self.id


def load_catalog(path: Path | None = None) -> dict[str, Any]:
    with (path or CATALOG_PATH).open("rb") as handle:
        return tomllib.load(handle)


def _resolve_ai_path(value: str | os.PathLike[str], ai_root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ai_root / path


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
            )
        )
    return tuple(result)


def _first_shard(path: Path) -> bool:
    match = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", path.name, re.IGNORECASE)
    return not match or match.group(1) == "00001"


def _family_for(path: Path, known: tuple[Family, ...]) -> Family:
    value = str(path).lower()
    for family in known:
        if family.id != "generic" and any(pattern in value for pattern in family.patterns):
            return family
    return next(family for family in known if family.id == "generic")


def _quant(name: str) -> str:
    normalized = re.sub(r"-\d{5}-of-\d{5}\.gguf$", "", name, flags=re.IGNORECASE)
    matches = re.findall(
        r"(?:UD-)?(?:IQ\d(?:_[A-Z0-9]+)+(?:-[A-Z0-9]+)?|Q\d(?:_[A-Z0-9]+)+(?:-[A-Z0-9]+)?)",
        normalized.upper(),
    )
    return matches[-1] if matches else "GGUF"


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _model_size(path: Path) -> int:
    match = re.search(r"-00001-of-(\d{5})\.gguf$", path.name, re.IGNORECASE)
    if not match:
        return path.stat().st_size
    stem = re.sub(r"-00001-of-\d{5}\.gguf$", "", path.name, flags=re.IGNORECASE)
    return sum(shard.stat().st_size for shard in path.parent.glob(f"{stem}-*-of-*.gguf"))


def discover_models(model_root: Path | None = None) -> list[Model]:
    root = model_root or settings().model_root
    if not root.exists():
        return []
    known = families()
    result: list[Model] = []
    ids: dict[str, int] = {}
    for path in sorted(root.rglob("*.gguf")):
        if ".cache" in path.parts or path.name.lower().startswith("mmproj"):
            continue
        if not _first_shard(path) or path.stat().st_size == 0:
            continue
        family = _family_for(path, known)
        quant = _quant(path.name)
        if family.id == "generic":
            base = re.sub(r"-\d{5}-of-\d{5}\.gguf$", "", path.name, flags=re.IGNORECASE)
            base = re.sub(r"\.gguf$", "", base, flags=re.IGNORECASE)
            model_id = _slug(base)
            display = base
        else:
            model_id = f"{family.id}-{_slug(quant)}"
            display = f"{family.display_name} {quant}"
        ids[model_id] = ids.get(model_id, 0) + 1
        if ids[model_id] > 1:
            model_id = f"{model_id}-{ids[model_id]}"
        result.append(Model(model_id, display, path, _model_size(path), family, quant))
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


def backend_for(model: Model) -> Backend:
    backend = backends().get(model.family.backend)
    if backend is None:
        raise ValueError(f"backend '{model.family.backend}' is not configured")
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
    return backend


def server_command(model: Model, profile: Profile, backend: Backend | None = None) -> list[str]:
    cfg = settings()
    selected = backend or backend_for(model)
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
    ]
    if profile.split_mode == "none":
        command.extend(["--main-gpu", str(profile.main_gpu)])
    elif profile.tensor_split:
        command.extend(["--tensor-split", profile.tensor_split])
    command.extend(profile.extra_args)
    return command


def format_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"
