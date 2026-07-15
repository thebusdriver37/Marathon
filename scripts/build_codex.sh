#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODEX_DIR="${MARATHON_CODEX_DIR:-$ROOT_DIR/codex}"
PATCH_IN_PLACE="${MARATHON_PATCH_IN_PLACE:-0}"
BUILD_CODEX_DIR="$CODEX_DIR"
DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
INSTALL_BIN="${MARATHON_PATCHED_CODEX_BIN:-$DATA_HOME/marathon/bin/codex}"
BUILD_PROFILE="${MARATHON_CODEX_BUILD_PROFILE:-release}"
TEMP_TARGET=""

if [[ "$PATCH_IN_PLACE" != "1" ]]; then
  BUILD_CODEX_DIR="${MARATHON_PATCHED_CODEX_DIR:-$ROOT_DIR/.marathon/codex-patched}"
fi

"$ROOT_DIR/scripts/apply_codex_patches.sh"

export CODEX_SKIP_VENDORED_BWRAP="${CODEX_SKIP_VENDORED_BWRAP:-1}"
if [[ -z "${CARGO_TARGET_DIR:-}" && -z "${MARATHON_CODEX_TARGET_DIR:-}" ]]; then
  TEMP_TARGET="$(mktemp -d "${TMPDIR:-/tmp}/marathon-codex-target.XXXXXX")"
  trap 'rm -rf "$TEMP_TARGET"' EXIT
  export CARGO_TARGET_DIR="$TEMP_TARGET"
else
  export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$MARATHON_CODEX_TARGET_DIR}"
fi

(
  cd "$BUILD_CODEX_DIR/codex-rs"
  cargo build --profile "$BUILD_PROFILE" -p codex-cli
)

install -Dm755 "$CARGO_TARGET_DIR/$BUILD_PROFILE/codex" "$INSTALL_BIN"
if command -v strip >/dev/null 2>&1; then
  strip "$INSTALL_BIN"
fi

echo
echo "Codex source: $BUILD_CODEX_DIR"
echo "Codex binary: $INSTALL_BIN"
