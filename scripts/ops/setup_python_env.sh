#!/usr/bin/env bash
set -euo pipefail

# Create or refresh the Marathon Python venv at .marathon/venv/.
#
# The venv is self-contained; router Python dependencies are declared in
# scripts/requirements.txt.
#
# Usage: scripts/ops/setup_python_env.sh [--force]
#
# --force  repair the venv and reinstall its dependencies without deleting it.

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

PYTHON="$VENV_DIR/bin/python3"
if [[ "$force" == "1" || ! -x "$PYTHON" ]] || ! "$PYTHON" -m pip --version >/dev/null 2>&1; then
  echo "preparing venv at $VENV_DIR" >&2
  # Ubuntu ships venv separately from ensurepip. Seeding pip ourselves also
  # repairs an interrupted first run without deleting any existing files.
  python3 -m venv --without-pip "$VENV_DIR"
fi

if ! "$PYTHON" -m pip --version >/dev/null 2>&1; then
  echo "installing Marathon's private package installer..." >&2
  "$PYTHON" - <<'PY'
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.request

# A pinned, hash-verified wheel, not an unversioned remote bootstrap script.
url = "https://files.pythonhosted.org/packages/44/3c/d717024885424591d5376220b5e836c2d5293ce2011523c9de23ff7bf068/pip-25.3-py3-none-any.whl"
expected = "9655943313a94722b7774661c21049070f6bbb0a1516bf02f7c8d5d9201514cd"
try:
    with tempfile.TemporaryDirectory(prefix="marathon-pip-") as directory:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("package installer checksum mismatch")
        wheel = Path(directory) / "pip-25.3-py3-none-any.whl"
        wheel.write_bytes(data)
        subprocess.run(
            [sys.executable, "-m", "pip", "--isolated", "--disable-pip-version-check",
             "install", "--no-index", "--no-deps", str(wheel)],
            env={**os.environ, "PYTHONPATH": str(wheel)}, check=True,
        )
except Exception as error:
    raise SystemExit(f"Could not prepare pip: {error}. Check your connection and rerun Marathon.")
PY
fi

pip_args=(--disable-pip-version-check install --quiet)
[[ "$force" != 1 ]] || pip_args+=(--force-reinstall)
"$PYTHON" -m pip "${pip_args[@]}" -r "$REQUIREMENTS"
cksum "$REQUIREMENTS" > "$VENV_DIR/.marathon-requirements"

echo
echo "Marathon venv ready at $VENV_DIR"
"$VENV_DIR/bin/python3" -c '
import importlib.metadata as m
for pkg in ("trafilatura", "aiohttp", "rich", "prompt-toolkit", "huggingface-hub", "gguf"):
    try:
        print(f"  {pkg}: {m.version(pkg)}")
    except m.PackageNotFoundError:
        print(f"  {pkg}: MISSING")
'
