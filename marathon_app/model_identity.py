"""Read model identity from the broker's launch configuration, never its alias."""

from pathlib import Path
import shlex

import yaml

from .catalog import Backend


def pool_identity(backend: Backend) -> tuple[str, str] | None:
    """Refresh on every discovery request; no network calls or worker startup.

    Stable routing aliases are deliberately not treated as weight identities.
    Unsupported or inconsistent commands must not inherit a plausible stale name.
    """
    if backend.identity_config is None:
        return None
    try:
        config = yaml.safe_load(backend.identity_config.read_text())
        identities = set()
        for worker in backend.pool_models:
            command = config["models"][worker]["cmd"]
            if not isinstance(command, str):
                raise ValueError("model command is not text")
            args = shlex.split(command)

            def option(name: str) -> str:
                values = [args[i + 1] for i, arg in enumerate(args[:-1]) if arg == name]
                values += [arg.split("=", 1)[1] for arg in args if arg.startswith(name + "=")]
                if len(values) > 1:
                    raise ValueError("ambiguous model option")
                return values[0] if values else ""

            target, draft = option("--model"), option("--spec-draft-model")
            if not target or "${" in target:
                raise ValueError("no concrete model file")
            # Include mounts when checking consistency: identical container paths
            # can refer to entirely different host files on different workers.
            mounts = tuple(sorted(
                [args[i + 1] for i, arg in enumerate(args[:-1]) if arg in ("--volume", "-v")]
                + [arg.split("=", 1)[1] for arg in args if arg.startswith("--volume=")]
            ))
            identities.add((target, draft, mounts))
        if len(identities) != 1:
            return "Local pool (mixed identities)", "Pool workers do not share one model configuration."
        target, draft, _ = identities.pop()
        name = Path(target).stem.replace("-", " ") + " (Local)"
        description = f"Configured target: {Path(target).name}."
        if draft:
            description += f" Drafter: {Path(draft).name}."
        return name, description
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError):
        return "Local pool (identity unavailable)", "Cannot verify the broker's configured model identity."
