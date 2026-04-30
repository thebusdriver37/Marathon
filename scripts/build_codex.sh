#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODEX_DIR="${MARATHON_CODEX_DIR:-$ROOT_DIR/codex}"

"$ROOT_DIR/scripts/apply_codex_patches.sh"

export CODEX_SKIP_VENDORED_BWRAP="${CODEX_SKIP_VENDORED_BWRAP:-1}"

cargo +1.93.0 build \
  --manifest-path "$CODEX_DIR/codex-rs/Cargo.toml" \
  -p codex-cli

echo
echo "Codex binary: $CODEX_DIR/codex-rs/target/debug/codex"
