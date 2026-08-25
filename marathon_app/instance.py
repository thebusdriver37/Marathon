"""Portable named-instance configuration for Marathon."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .catalog import Profile, Settings, load_catalog, settings


INSTANCE_NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{0,62}")
INSTANCE_PORT_SPAN = 10_000


def normalize_instance_name(value: str | None) -> str | None:
    """Validate a CLI/catalog instance name and normalize the default alias."""

    if value is None:
        return None
    name = value.strip()
    if name == "default":
        return None
    if not INSTANCE_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            "instance names must start with a lowercase letter or number and "
            "contain only lowercase letters, numbers, '_' and '-' (63 characters maximum)"
        )
    return name


def instance_path(root: Path, name: str | None) -> Path:
    """Return an isolated tree for a name while preserving default paths."""

    return root if name is None else root / "instances" / name


def _derived_port(base: int, name: str) -> int:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    offset = 1 + int.from_bytes(digest[:4], "big") % INSTANCE_PORT_SPAN
    port = base + offset
    if port > 65_535:
        raise ValueError(
            f"cannot derive a port for instance '{name}' from base port {base}; "
            "configure explicit llama_port and router_port values"
        )
    return port


def _configured_port(
    raw: dict[str, Any],
    key: str,
    environment_key: str,
    base: int,
    name: str,
) -> int:
    if environment_key in os.environ:
        value = base
    elif key in raw:
        value = raw[key]
    else:
        value = _derived_port(base, name)
    if isinstance(value, bool):
        raise ValueError(f"instances.{name}.{key} must be a TCP port")
    try:
        port = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"instances.{name}.{key} must be a TCP port") from error
    if not 1 <= port <= 65_535:
        raise ValueError(f"instances.{name}.{key} must be between 1 and 65535")
    return port


def _configured_gpus(raw: dict[str, Any], name: str) -> tuple[int, ...] | None:
    if "gpus" not in raw:
        return None
    values = raw["gpus"]
    if not isinstance(values, list) or not values:
        raise ValueError(f"instances.{name}.gpus must be a non-empty list of GPU indexes")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise ValueError(f"instances.{name}.gpus must contain non-negative integers")
    gpus = tuple(values)
    if len(gpus) != len(set(gpus)):
        raise ValueError(f"instances.{name}.gpus must not contain duplicates")
    return gpus


@dataclass(frozen=True)
class InstanceConfig:
    """Resolved identity, hardware selection, and listener settings."""

    name: str | None
    settings: Settings
    gpus: tuple[int, ...] | None = None

    @property
    def label(self) -> str:
        return self.name or "default"

    @property
    def cli_arguments(self) -> tuple[str, ...]:
        return ("--instance", self.name) if self.name is not None else ()

    def apply_profile(self, profile: Profile) -> Profile:
        return replace(profile, gpus=self.gpus) if self.gpus is not None else profile


def resolve_instance(
    value: str | None,
    catalog_data: dict[str, Any] | None = None,
) -> InstanceConfig:
    """Resolve a named instance from the merged catalog.

    Named instances work without catalog entries.
    Their ports are stable functions of the name, and a personal catalog can
    pin GPUs or provide explicit ports when local policy requires it.
    """

    name = normalize_instance_name(value)
    loaded = load_catalog() if catalog_data is None else catalog_data
    base = settings(loaded)
    if name is None:
        return InstanceConfig(None, base)

    configured = loaded.get("instances", {})
    if not isinstance(configured, dict):
        raise ValueError("instances must be a table")
    raw = configured.get(name, {})
    if not isinstance(raw, dict):
        raise ValueError(f"instances.{name} must be a table")
    unknown = set(raw) - {"gpus", "llama_port", "router_port"}
    if unknown:
        raise ValueError(
            f"instances.{name} has unknown setting(s): {', '.join(sorted(unknown))}"
        )

    llama_port = _configured_port(
        raw, "llama_port", "MARATHON_LLAMA_PORT", base.llama_port, name
    )
    router_port = _configured_port(
        raw, "router_port", "MARATHON_PROXY_PORT", base.router_port, name
    )
    if llama_port == router_port:
        raise ValueError(
            f"instance '{name}' cannot use port {llama_port} for both backend and router"
        )
    return InstanceConfig(
        name,
        replace(base, llama_port=llama_port, router_port=router_port),
        _configured_gpus(raw, name),
    )
