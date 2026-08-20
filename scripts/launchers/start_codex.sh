#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
ROUTER_HOST="${MARATHON_PROXY_HOST:-127.0.0.1}"
ROUTER_PORT="${MARATHON_PROXY_PORT:-18111}"
CODEX_BIN="${MARATHON_CODEX_BIN:-}"
DEFAULT_CODEX_BIN="$DATA_HOME/marathon/bin/codex"
LOG_DIR="${MARATHON_LOG_DIR:-$ROOT_DIR/logs}"
STATE_DIR="${MARATHON_ROUTER_STATE_DIR:-$ROOT_DIR/.marathon/state}"
LOCAL_MODELS_FILE="${MARATHON_MODELS_FILE:-$ROOT_DIR/config/qwen_models.json}"
MODEL_PROVIDER_ID="${MARATHON_MODEL_PROVIDER_ID:-marathon-local}"
MODEL_PROVIDER_NAME="${MARATHON_MODEL_PROVIDER_NAME:-Marathon Local}"

mkdir -p "$LOG_DIR" "$STATE_DIR"

router_base="http://$ROUTER_HOST:$ROUTER_PORT"
router_v1="$router_base/v1"

if [[ -z "$CODEX_BIN" ]]; then
  if [[ -x "$DEFAULT_CODEX_BIN" ]]; then
    CODEX_BIN="$DEFAULT_CODEX_BIN"
  else
    CODEX_BIN="codex"
  fi
fi

backend_not_running() {
  cat >&2 <<MSG
Marathon backend is not running.

Start it first:
  marathon backend start 128k-single

Then run Marathon from your project:
  marathon
MSG
}

health_json="$(curl -fsS --max-time 2 "$router_base/health" 2>/dev/null || true)"
if [[ -z "$health_json" ]]; then
  backend_not_running
  exit 1
fi

health_value() {
  local expr="$1"
  HEALTH_JSON="$health_json" python3 - "$expr" <<'PY'
import json
import os
import sys

try:
    payload = json.loads(os.environ["HEALTH_JSON"])
except Exception:
    raise SystemExit(1)

value = payload
for part in sys.argv[1].split("."):
    if not part:
        continue
    if not isinstance(value, dict):
        value = None
        break
    value = value.get(part)

if value is None:
    raise SystemExit(1)
print(value)
PY
}

active_model="$(health_value current_model || true)"
backend_status="$(health_value backend_health.status || true)"

if [[ -z "$active_model" || "$backend_status" != "ok" ]]; then
  cat >&2 <<MSG
Marathon backend is not ready.

Check it with:
  marathon backend status

Or restart it:
  marathon backend restart 128k-single
MSG
  exit 1
fi

export CODEX_OSS_BASE_URL="$router_v1"

runtime_models_file="${MARATHON_RUNTIME_MODELS_FILE:-$STATE_DIR/codex_models.json}"
model_catalog="$(curl -fsS --max-time 3 "$router_v1/models" 2>/dev/null || true)"
if [[ -n "$model_catalog" ]]; then
  tmp_models_file="$(mktemp "$STATE_DIR/codex_models.XXXXXX")"
  printf '%s\n' "$model_catalog" >"$tmp_models_file"
  mv "$tmp_models_file" "$runtime_models_file"
  MODEL_CATALOG_FILE="$runtime_models_file"
else
  MODEL_CATALOG_FILE="$LOCAL_MODELS_FILE"
fi

if [[ -z "${CODEX_CLI_NAME:-}" ]]; then
  if command -v marathon >/dev/null 2>&1; then
    export CODEX_CLI_NAME="marathon"
  else
    export CODEX_CLI_NAME="$ROOT_DIR/bin/marathon"
  fi
fi

IFS=$'\t' read -r CODEX_HOME CODEX_SHARED_PROFILE < <(
  PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" python3 -m marathon_app.codex_home
)
export CODEX_HOME
if [[ -n "$CODEX_SHARED_PROFILE" ]]; then
  export CODEX_SQLITE_HOME="$CODEX_HOME"
fi

ensure_codex_home_config() {
  local config_path
  config_path="${CODEX_HOME:-}/config.toml"
  [[ -n "$config_path" ]] || return 0
  touch "$config_path"
  if rg -q '^tui\.status_line\s*=\s*\[[^]]*\]$' "$config_path" 2>/dev/null \
    && ! rg -q '^tui\.status_line\s*=\s*\[[^]]*"tokens-per-second"' "$config_path" 2>/dev/null; then
    perl -0pi -e 's/^(tui\.status_line\s*=\s*\[)/$1"tokens-per-second", /m' "$config_path"
    return 0
  fi
  if ! rg -q '^(status_line|tui\.status_line)\s*=' "$config_path" 2>/dev/null; then
    local tmp
    tmp="$(mktemp)"
    {
      printf 'tui.status_line = ["model-with-reasoning", "tokens-per-second", "context-remaining", "context-window-size", "context-tokens"]\n'
      cat "$config_path"
    } >"$tmp"
    mv "$tmp" "$config_path"
  fi
}

ensure_codex_home_config

WEB_SEARCH_MODE="${MARATHON_WEB_SEARCH_MODE:-cached}"
SEARXNG_URL="${MARATHON_SEARXNG_URL:-http://127.0.0.1:18093}"

if [[ "$WEB_SEARCH_MODE" != "disabled" ]]; then
  if ! curl -fsS --max-time 2 "$SEARXNG_URL/healthz" >/dev/null 2>&1; then
    echo "warning: SearXNG at $SEARXNG_URL is not reachable; web search will fail until you run 'marathon search up'" >&2
    echo "         set MARATHON_WEB_SEARCH_MODE=disabled to silence this warning" >&2
  fi
fi

MODEL_PROVIDER_CONFIG="model_providers.$MODEL_PROVIDER_ID={ name = \"$MODEL_PROVIDER_NAME\", base_url = \"$router_v1\", wire_api = \"responses\", requires_openai_auth = false, supports_websockets = true }"
COMMON_ARGS=(
  -c "$MODEL_PROVIDER_CONFIG"
  -c "model_provider=\"$MODEL_PROVIDER_ID\""
  -c "model_catalog_json=\"$MODEL_CATALOG_FILE\""
  -m "$active_model"
  -c "web_search=\"$WEB_SEARCH_MODE\""
  --disable image_generation
  --disable apps
  --disable plugins
)

if [[ "${1:-}" == "exec" ]]; then
  shift
  exec "$CODEX_BIN" exec --ignore-user-config "${COMMON_ARGS[@]}" "$@"
fi

PROFILE_ARGS=()
if [[ -n "$CODEX_SHARED_PROFILE" ]]; then
  PROFILE_ARGS=(--profile "$CODEX_SHARED_PROFILE")
fi

exec "$CODEX_BIN" "${PROFILE_ARGS[@]}" "${COMMON_ARGS[@]}" "$@"
