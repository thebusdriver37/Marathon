#!/usr/bin/env bash
set -euo pipefail

# IDE socket fixtures must satisfy upstream's private-directory checks even
# when the invoking shell uses a group-writable umask.
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODEX_DIR="${MARATHON_CODEX_DIR:-$ROOT_DIR/codex}"
TEMP_TARGET=""

if ! cargo nextest --version >/dev/null 2>&1; then
  echo "error: native developer tests require cargo-nextest; see docs/DEVELOPMENT.md" >&2
  exit 1
fi

if [[ ! -d "$CODEX_DIR/codex-rs" ]]; then
  echo "error: Codex source not found: $CODEX_DIR/codex-rs" >&2
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
  cd "$CODEX_DIR/codex-rs"
  export RUST_MIN_STACK="${RUST_MIN_STACK:-8388608}"
  export NEXTEST_PROFILE="${NEXTEST_PROFILE:-local}"
  # Nextest can inject SSL_CERT_FILE while probing system roots. The native-TLS
  # unit tests need a clean environment; CA probes must retain their own fixtures.
  host_triple="$(rustc -vV | sed -n 's/^host: //p' | tr '[:lower:]-' '[:upper:]_')"
  env "CARGO_TARGET_${host_triple}_RUNNER=env -u SSL_CERT_FILE -u CODEX_CA_CERTIFICATE" \
    cargo nextest run --no-fail-fast -p codex-http-client --lib
  cargo nextest run --no-fail-fast -p codex-http-client --test ca_env
  cargo nextest run --no-fail-fast -p codex-core --lib local_runtime
  cargo nextest run --no-fail-fast -p codex-cli marathon_rejects_cloud_and_shared_server_commands
  cargo nextest run --no-fail-fast -p codex-utils-cli
  cargo nextest run --no-fail-fast -p codex-protocol \
    token_usage_percentage_uses_the_full_runtime_window
  cargo nextest run --no-fail-fast -p codex-app-server-protocol --lib
  cargo nextest run --no-fail-fast -p codex-core --test all suite::compact::manual_compact
  cargo nextest run --no-fail-fast -p codex-core --test all \
    responses_websocket_preserves_credit_usage_metadata
  cargo nextest run --no-fail-fast -p codex-core --test all \
    apply_patch_custom_tool_streaming_emits_updated_changes
  cargo nextest run --no-fail-fast -p codex-features \
    unstable_warning_event_only_mentions_enabled_under_development_features
  cargo nextest run --no-fail-fast -p codex-api --test sse_end_to_end \
    responses_stream_parses_items_and_completed_end_to_end
  cargo nextest run --no-fail-fast -p codex-tui \
    context_percentage_uses_the_full_runtime_window
  cargo nextest run --no-fail-fast -p codex-tui \
    status_line_context_tokens_renders_compact_usage_ratio
  cargo nextest run --no-fail-fast -p codex-tui turn_throughput
  cargo nextest run --no-fail-fast -p codex-tui --lib thread_title
  env -u NO_COLOR cargo nextest run --no-fail-fast -p codex-tui --lib \
    -E 'test(session_header) | test(status_snapshot) | test(onboarding::welcome) | test(terminal_title) | test(completed_global_chord)'
  cargo nextest run --no-fail-fast -p codex-app-server --test all \
    -E 'test(turn_start_emits_raw_response_completed_with_upstream_usage) | test(thread_compact_start_triggers_compaction_and_returns_empty_response)'
  cargo nextest run --no-fail-fast -p codex-app-server --test all \
    turn_start_streams_apply_patch_change_updates_v2
  cargo nextest run --no-fail-fast -p codex-tui \
    status_line_tokens_per_second_renders_completed_turn_rate_snapshot
  cargo nextest run --no-fail-fast -p codex-tui \
    status_line_limits_active_states_to_working_and_thinking
  cargo nextest run --no-fail-fast -p codex-tui \
    active_turn_footer_schedules_continuous_animation_frames
  cargo nextest run --no-fail-fast -p codex-tui \
    status_line_stays_stable_during_streaming_patch_lifecycle
  cargo nextest run --no-fail-fast -p codex-tui distinguishes_unset_from_disabled
  cargo nextest run --no-fail-fast -p codex-tui resolves_name_and_preview_from_server_list
  cargo nextest run --no-fail-fast -p codex-state \
    sqlite_sink_filters_noisy_targets_without_dropping_useful_diagnostics
)

PYTHONPATH="$ROOT_DIR" "$ROOT_DIR/.marathon/venv/bin/python3" \
  -m unittest discover -s "$ROOT_DIR/tests" -v
