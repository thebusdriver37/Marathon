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
INSTALL_TMP=""

if [[ "$PATCH_IN_PLACE" != "1" ]]; then
  BUILD_CODEX_DIR="${MARATHON_PATCHED_CODEX_DIR:-$ROOT_DIR/.marathon/codex-patched}"
fi

"$ROOT_DIR/scripts/apply_codex_patches.sh"

export CODEX_SKIP_VENDORED_BWRAP="${CODEX_SKIP_VENDORED_BWRAP:-1}"
if [[ -z "${CARGO_TARGET_DIR:-}" && -z "${MARATHON_CODEX_TARGET_DIR:-}" ]]; then
  TEMP_TARGET="$(mktemp -d "${TMPDIR:-/tmp}/marathon-codex-target.XXXXXX")"
  export CARGO_TARGET_DIR="$TEMP_TARGET"
else
  export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$MARATHON_CODEX_TARGET_DIR}"
fi

cleanup() {
  [[ -z "$INSTALL_TMP" ]] || rm -f "$INSTALL_TMP"
  [[ -z "$TEMP_TARGET" ]] || rm -rf "$TEMP_TARGET"
}
trap cleanup EXIT

if [[ "${MARATHON_CODEX_RUN_TESTS:-1}" == "1" ]]; then
  echo "-> Testing Marathon Codex patches..."
  MARATHON_PATCHED_CODEX_DIR="$BUILD_CODEX_DIR" \
    "$ROOT_DIR/scripts/test_codex_patches.sh"
fi

(
  cd "$BUILD_CODEX_DIR/codex-rs"
  cargo build --profile "$BUILD_PROFILE" -p codex-cli
)

candidate="$CARGO_TARGET_DIR/$BUILD_PROFILE/codex"
"$candidate" --version
"$candidate" --help >/dev/null
"$candidate" exec --help >/dev/null

install_dir="$(dirname "$INSTALL_BIN")"
mkdir -p "$install_dir"
INSTALL_TMP="$(mktemp "$install_dir/.codex.new.XXXXXX")"
install -m755 "$candidate" "$INSTALL_TMP"
if command -v strip >/dev/null 2>&1; then
  strip "$INSTALL_TMP"
fi

if [[ -x "$INSTALL_BIN" ]]; then
  backup_tmp="$(mktemp "$install_dir/.codex.previous.XXXXXX")"
  install -m755 "$INSTALL_BIN" "$backup_tmp"
  mv -f "$backup_tmp" "$INSTALL_BIN.previous"
  if [[ -f "$INSTALL_BIN.source" ]]; then
    cp -f "$INSTALL_BIN.source" "$INSTALL_BIN.previous.source"
  fi
fi

mv -f "$INSTALL_TMP" "$INSTALL_BIN"
INSTALL_TMP=""

source_tmp="$(mktemp "$install_dir/.codex.source.XXXXXX")"
git -C "$CODEX_DIR" rev-parse HEAD >"$source_tmp"
mv -f "$source_tmp" "$INSTALL_BIN.source"

echo
echo "Codex source: $BUILD_CODEX_DIR"
echo "Codex binary: $INSTALL_BIN"
echo "Codex commit: $(cat "$INSTALL_BIN.source")"
