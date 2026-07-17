#!/usr/bin/env bash
set -euo pipefail

# Create or refresh the Marathon Python venv at .marathon/venv/.
#
# The venv is self-contained; router Python dependencies are declared in
# scripts/requirements.txt.
#
# Usage: scripts/ops/setup_python_env.sh [--force]
#
# --force  recreate the venv from scratch even if it already exists.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="$ROOT_DIR/.marathon/venv"
REQUIREMENTS="$ROOT_DIR/scripts/requirements.txt"

force=0
if [[ "${1:-}" == "--force" ]]; then
  force=1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "error: python3 is not installed or not on PATH" >&2
  exit 1
fi

PY_VERSION="$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
if ! python3 -c 'import sys; raise SystemExit(sys.version_info < (3, 10))'; then
  echo "error: Python 3.10 or newer is required; found python3 $PY_VERSION." >&2
  exit 1
fi

if [[ "$force" == "1" && -d "$VENV_DIR" ]]; then
  echo "removing existing venv at $VENV_DIR" >&2
  rm -rf "$VENV_DIR"
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "creating venv at $VENV_DIR" >&2
  python3 -m venv "$VENV_DIR"
fi

PIP="$VENV_DIR/bin/pip"
"$PIP" install --quiet --upgrade pip
"$PIP" install --quiet -r "$REQUIREMENTS"

echo
echo "Marathon venv ready at $VENV_DIR"
"$VENV_DIR/bin/python3" -c '
import importlib.metadata as m
for pkg in ("trafilatura", "aiohttp", "rich", "prompt-toolkit"):
    try:
        print(f"  {pkg}: {m.version(pkg)}")
    except m.PackageNotFoundError:
        print(f"  {pkg}: MISSING")
'
