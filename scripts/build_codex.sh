#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODEX_DIR="${MARATHON_CODEX_DIR:-$ROOT_DIR/codex}"
PATCH_IN_PLACE="${MARATHON_PATCH_IN_PLACE:-0}"
BUILD_CODEX_DIR="$CODEX_DIR"

if [[ "$PATCH_IN_PLACE" != "1" ]]; then
  BUILD_CODEX_DIR="${MARATHON_PATCHED_CODEX_DIR:-$ROOT_DIR/.marathon/codex-patched}"
fi

"$ROOT_DIR/scripts/apply_codex_patches.sh"

export CODEX_SKIP_VENDORED_BWRAP="${CODEX_SKIP_VENDORED_BWRAP:-1}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-${MARATHON_CODEX_TARGET_DIR:-$ROOT_DIR/.marathon/codex-target}}"

cargo +1.93.0 build \
  --manifest-path "$BUILD_CODEX_DIR/codex-rs/Cargo.toml" \
  -p codex-cli

echo
echo "Codex source: $BUILD_CODEX_DIR"
echo "Codex binary: $CARGO_TARGET_DIR/debug/codex"
