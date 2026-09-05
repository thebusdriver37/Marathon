#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_CODEX_DIR="${MARATHON_PATCHED_CODEX_DIR:-$ROOT_DIR/.marathon/codex-patched}"
TEMP_TARGET=""

if [[ ! -d "$BUILD_CODEX_DIR/codex-rs" ]]; then
  echo "error: patched Codex tree not found: $BUILD_CODEX_DIR" >&2
  exit 1
fi

cleanup() {
  if [[ -n "$TEMP_TARGET" && -d "$TEMP_TARGET" ]]; then
    rm -r -- "$TEMP_TARGET"
  fi
}
trap cleanup EXIT

if [[ -z "${CARGO_TARGET_DIR:-}" ]]; then
  if [[ -n "${MARATHON_CODEX_TARGET_DIR:-}" ]]; then
    export CARGO_TARGET_DIR="$MARATHON_CODEX_TARGET_DIR"
  else
    TEMP_TARGET="$(mktemp -d "${TMPDIR:-/tmp}/marathon-codex-target.XXXXXX")"
    export CARGO_TARGET_DIR="$TEMP_TARGET"
  fi
fi

# Codex is a large workspace. Test binaries do not need incremental state or
# debug symbols, which otherwise retain tens of gigabytes between runs.
export CARGO_INCREMENTAL="${CARGO_INCREMENTAL:-0}"
export CARGO_PROFILE_TEST_DEBUG="${CARGO_PROFILE_TEST_DEBUG:-0}"

(
  cd "$BUILD_CODEX_DIR/codex-rs"
  export RUST_MIN_STACK="${RUST_MIN_STACK:-8388608}"
  just test -p codex-utils-cli
  just test -p codex-protocol \
    token_usage_percentage_uses_the_full_runtime_window
  just test -p codex-app-server-protocol --lib
  just test -p codex-core --test all suite::compact::manual_compact
  just test -p codex-core --test all \
    responses_websocket_preserves_credit_usage_metadata
  just test -p codex-api --test sse_end_to_end \
    responses_stream_parses_items_and_completed_end_to_end
  just test -p codex-tui \
    context_percentage_uses_the_full_runtime_window
  just test -p codex-tui \
    status_line_context_tokens_renders_live_context_count
  just test -p codex-tui turn_throughput
  just test -p codex-app-server --test all \
    -E 'test(turn_start_emits_raw_response_completed_with_upstream_usage) | test(thread_compact_start_triggers_compaction_and_returns_empty_response)'
  just test -p codex-tui \
    status_line_tokens_per_second_renders_completed_turn_rate_snapshot
  just test -p codex-tui \
    status_line_tokens_per_second_tracks_generation_stage
  just test -p codex-tui distinguishes_unset_from_disabled
  just test -p codex-tui ignores_sqlite_candidate_from_another_provider
  just test -p codex-state \
    sqlite_sink_filters_noisy_targets_without_dropping_useful_diagnostics
)

PYTHONPATH="$ROOT_DIR" "$ROOT_DIR/.marathon/venv/bin/python3" \
  -m unittest discover -s "$ROOT_DIR/tests" -v
