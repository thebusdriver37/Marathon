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
    echo
    echo "warning: $INSTALL_BIN_DIR is not on PATH for this shell."
    echo "Add this to your shell profile, then open a new terminal:"
    echo
    echo "  export PATH=\"$INSTALL_BIN_DIR:\$PATH\""
    echo
    echo "For this terminal only, run:"
    echo
    echo "  export PATH=\"$INSTALL_BIN_DIR:\$PATH\""
    ;;
esac
