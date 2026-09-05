#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INSTALL_BIN_DIR="${MARATHON_INSTALL_BIN_DIR:-$HOME/.local/bin}"
TARGET="$INSTALL_BIN_DIR/marathon"
SOURCE="$ROOT_DIR/bin/marathon"

usage() {
  cat <<USAGE
Usage: ./bin/marathon install [--force]

Installs the Marathon launcher as:
  $TARGET

Options:
  --force   Replace an existing non-symlink target

Environment:
  MARATHON_INSTALL_BIN_DIR   Override install dir (default: ~/.local/bin)
  MARATHON_CONFIGURE_SHELL   Set to 0 to leave shell startup files unchanged
USAGE
}

force=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help|help)
      usage
      exit 0
      ;;
    --force)
      force=1
      shift
      ;;
    *)
      echo "error: unknown install option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -x "$SOURCE" ]]; then
  chmod +x "$SOURCE"
fi

mkdir -p "$INSTALL_BIN_DIR"
if [[ -e "$TARGET" && ! -L "$TARGET" && "$force" != "1" ]]; then
  echo "error: $TARGET already exists and is not a symlink" >&2
  echo "rerun with --force to replace it, or set MARATHON_INSTALL_BIN_DIR" >&2
  exit 1
fi
ln -sfn "$SOURCE" "$TARGET"

echo "installed marathon -> $TARGET"

case ":$PATH:" in
  *":$INSTALL_BIN_DIR:"*)
    echo "ok: $INSTALL_BIN_DIR is on PATH"
    ;;
  *)
    if [[ "${MARATHON_CONFIGURE_SHELL:-1}" == 1 ]] && command -v python3 >/dev/null 2>&1; then
      python3 - "$INSTALL_BIN_DIR" <<'PY'
import os
from pathlib import Path
import shlex
import shutil
import sys

directory = shlex.quote(str(Path(sys.argv[1]).resolve()))
user_root = Path.home()
shell = Path(os.environ.get("SHELL", "/bin/bash")).name
snippet = (f'\n# Marathon CLI PATH\ncase ":$PATH:" in\n'
           f'  *:{directory}:*) ;;\n  *) export PATH={directory}:"$PATH" ;;\nesac\n')
if shell == "bash":
    login = next((user_root / name for name in (".bash_profile", ".bash_login", ".profile")
                  if (user_root / name).exists()), user_root / ".profile")
    paths = [user_root / ".bashrc", login]
elif shell == "zsh":
    paths = [Path(os.environ.get("ZDOTDIR", str(user_root))) / ".zshrc"]
elif shell == "fish":
    paths = [Path(os.environ.get("XDG_CONFIG_HOME", str(user_root / ".config")))
             / "fish/conf.d/marathon.fish"]
    snippet = f"\n# Marathon CLI PATH\nfish_add_path --path {directory}\n"
else:
    print(f"Shell {shell}: add {sys.argv[1]} to PATH, or use the checkout's bin/marathon.")
    raise SystemExit(0)
for path in paths:
    try:
        original = path.read_text() if path.exists() else ""
        if snippet in original:
            continue
        backup = path.with_name(path.name + ".before-marathon")
        if path.exists() and not backup.exists():
            shutil.copy2(path, backup)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(snippet)
        print(f"Configured PATH in {path}" + (f" (backup: {backup})" if original else ""))
    except OSError as error:
        print(f"Could not configure {path}: {error}. Use the checkout's bin/marathon.")
PY
    fi
    echo "Open a new terminal to use 'marathon' from any project."
    echo "This terminal can continue using: $SOURCE"
    ;;
esac
