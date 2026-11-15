"""Read model identity from the broker's launch configuration, never its alias."""

from functools import lru_cache
import hashlib
from pathlib import Path
import re
import shlex

import yaml

from .catalog import Backend


def pool_identity(backend: Backend) -> tuple[str, str] | None:
    """Check current contents on every discovery; parse only changed identities.

    Stable routing aliases are deliberately not treated as weight identities.
    Unsupported or inconsistent commands must not inherit a plausible stale name.
    """
    if backend.identity_config is None:
        return None
    try:
        return _configured_identity(backend.identity_config.read_text(), backend.pool_models)
    except (OSError, UnicodeError):
        return "Local pool (identity unavailable)", "Cannot verify the broker's configured model identity."


def pool_cache_layout(backend: Backend) -> tuple[Path, str]:
    """Return the host folder pool workers save slots to, and a hash of their commands.

    The broker config is the only source of truth. The router must look for
    snapshots where the workers really write them, and any change to a worker's
    launch command must retire snapshots that worker may no longer restore.
    """
    source = backend.identity_config
    try:
        models = yaml.safe_load(source.read_text())["models"]
        commands = sorted({models[worker]["cmd"] for worker in backend.pool_models})
        folders = set()
        for command in commands:
            args = shlex.split(command)
            [target] = _option_values(args, "--slot-save-path")
            [folder] = [
                volume.split(":")[0]
                for volume in _option_values(args, "--volume", "-v")
                if volume.split(":")[1:2] == [target]
            ]
            folders.add(folder)
    except (OSError, UnicodeError, KeyError, TypeError, ValueError, yaml.YAMLError) as error:
        raise ValueError(
            f"Cannot find where the pool workers in {source} save slot snapshots. Each "
            "worker needs one --slot-save-path that a --volume maps to a host folder."
        ) from error
    folder = folders.pop() if len(folders) == 1 else ", ".join(sorted(folders))
    if folders or "${" in folder or not Path(folder).is_absolute():
        raise ValueError(
            f"The pool workers in {source} must share one absolute slot snapshot "
            f"folder, not {folder}."
        )
    digest = hashlib.sha256("\0".join(commands).encode("utf-8")).hexdigest()
    return Path(folder), digest


def pool_serves_images(backend: Backend) -> bool:
    """Whether every selected worker loads a vision projector (`--mmproj`).

    The broker YAML, not a model file Marathon happens to find on disk, decides
    what the pool serves, so a stray copy of the weights cannot hide vision.
    """
    try:
        models = yaml.safe_load(backend.identity_config.read_text())["models"]
        return all(_option_values(shlex.split(models[worker]["cmd"]), "--mmproj")
                   for worker in backend.pool_models)
    except (AttributeError, OSError, UnicodeError, KeyError, TypeError, ValueError, yaml.YAMLError):
        return False


def _option_values(args: list[str], *names: str) -> list[str]:
    """Collect an option's values given as `--name value` or `--name=value`."""
    values = [args[i + 1] for i, arg in enumerate(args[:-1]) if arg in names]
    return values + [arg.split("=", 1)[1] for arg in args if arg.split("=", 1)[0] in names and "=" in arg]


@lru_cache(maxsize=8)
def _configured_identity(content: str, workers: tuple[str, ...]) -> tuple[str, str]:
    # Key by contents, not just mtime: even same-size edits, atomic replacement,
    # and edits that preserve timestamps must update the picker immediately.
    try:
        config = yaml.safe_load(content)
        identities = set()
        for worker in workers:
            command = config["models"][worker]["cmd"]
            if not isinstance(command, str):
                raise ValueError("model command is not text")
            args = shlex.split(command)

            def option(name: str) -> str:
                values = _option_values(args, name)
                if len(values) > 1:
                    raise ValueError("ambiguous model option")
                return values[0] if values else ""

            target, draft = option("--model"), option("--spec-draft-model")
            if not target or "${" in target:
                raise ValueError("no concrete model file")
            # Include mounts when checking consistency: identical container paths
            # can refer to entirely different host files on different workers.
            mounts = tuple(sorted(_option_values(args, "--volume", "-v")))
            identities.add((target, draft, mounts))
        if len(identities) != 1:
            return "Local pool (mixed identities)", "Pool workers do not share one model configuration."
        target, draft, _ = identities.pop()
        # Keep deployment details in the description, not the picker/footer.
        stem = Path(target).stem
        if re.search(r"qwen3[.]8[-_ ]27b(?:[-_ ]|$)", stem, re.IGNORECASE):
            name = "Qwen3.8 27B"
        elif re.search(r"qwen3[.]8[-_ ]flash[-_ ]next(?:[-_ ]|$)", stem, re.IGNORECASE):
            name = "Qwen3.8 Flash-Next"
        else:
            name = stem.replace("-", " ") + " (Local)"
        description = f"Configured target: {Path(target).name}."
        if draft:
            description += f" Drafter: {Path(draft).name}."
        return name, description
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError):
        return "Local pool (identity unavailable)", "Cannot verify the broker's configured model identity."
