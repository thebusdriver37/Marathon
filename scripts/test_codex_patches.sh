#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_CODEX_DIR="${MARATHON_PATCHED_CODEX_DIR:-$ROOT_DIR/.marathon/codex-patched}"

if [[ ! -d "$BUILD_CODEX_DIR/codex-rs" ]]; then
  echo "error: patched Codex tree not found: $BUILD_CODEX_DIR" >&2
  exit 1
fi

(
  cd "$BUILD_CODEX_DIR/codex-rs"
  export RUST_MIN_STACK="${RUST_MIN_STACK:-8388608}"
  cargo test -p codex-utils-cli
  cargo test -p codex-protocol \
    token_usage_percentage_uses_the_full_runtime_window
  cargo test -p codex-tui \
    context_percentage_uses_the_full_runtime_window
  cargo test -p codex-tui \
    status_line_context_tokens_renders_live_context_count
  cargo test -p codex-tui turn_throughput
  cargo test -p codex-tui \
    status_line_tokens_per_second_renders_completed_turn_rate_snapshot
)

PYTHONPATH="$ROOT_DIR" "$ROOT_DIR/.marathon/venv/bin/python3" \
  -m unittest discover -s "$ROOT_DIR/tests" -v
