#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROUTER_HOST="${MARATHON_PROXY_HOST:-${CODEX_QWEN_PROXY_HOST:-127.0.0.1}}"
ROUTER_PORT="${MARATHON_PROXY_PORT:-${CODEX_QWEN_PROXY_PORT:-18111}}"
DEFAULT_MODEL="${MARATHON_DEFAULT_MODEL:-${CODEX_QWEN_DEFAULT_MODEL:-qwen3.6-27b-q4-128k}}"
LOCAL_CODEX_BIN="$ROOT_DIR/codex/codex-rs/target/debug/codex"
CODEX_BIN="${MARATHON_CODEX_BIN:-${CODEX_QWEN_CODEX_BIN:-}}"
LOG_DIR="${MARATHON_LOG_DIR:-$ROOT_DIR/logs}"
ROUTER_LOG_FILE="$LOG_DIR/codex_local_router.log"
STATE_DIR="${MARATHON_STATE_DIR:-${CODEX_QWEN_STATE_DIR:-$ROOT_DIR/.marathon/state}}"
ROUTER_PID_FILE="$STATE_DIR/codex-local-router.pid"
LOCAL_MODELS_FILE="${MARATHON_MODELS_FILE:-${CODEX_QWEN_MODELS_FILE:-$ROOT_DIR/config/qwen_models.json}}"

mkdir -p "$LOG_DIR"
mkdir -p "$STATE_DIR"

router_host="${ROUTER_HOST}:${ROUTER_PORT}"

if [[ -z "$CODEX_BIN" ]]; then
  if [[ -x "$LOCAL_CODEX_BIN" ]]; then
    CODEX_BIN="$LOCAL_CODEX_BIN"
  else
    CODEX_BIN="codex"
  fi
fi

port_owner_pid() {
  local port="$1"
  ss -ltnp "( sport = :$port )" 2>/dev/null | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | head -n1
}

pid_cmdline() {
  local pid="$1"
  ps -p "$pid" -o cmd= 2>/dev/null || true
}

json_model_matches() {
  local url="$1"
  local expected_model="$2"
  local payload
  payload="$(curl -fsS --max-time 2 "$url/v1/models" 2>/dev/null || true)"
  [[ -n "$payload" ]] || return 1
  PAYLOAD="$payload" python3 - "$expected_model" <<'PY'
import json
import os
import sys

expected = sys.argv[1]
try:
    payload = json.loads(os.environ["PAYLOAD"])
except Exception:
    raise SystemExit(1)

def matches(item):
    if not isinstance(item, dict):
        return False
    return any(item.get(key) == expected for key in ("id", "slug", "model", "name"))

for key in ("data", "models"):
    value = payload.get(key)
    if isinstance(value, list) and any(matches(item) for item in value):
        raise SystemExit(0)

raise SystemExit(1)
PY
}

is_expected_router_pid() {
  local pid="$1"
  local cmd
  cmd="$(pid_cmdline "$pid")"
  [[ -n "$cmd" ]] &&
    [[ "$cmd" == *"codex_local_router.py"* ]] &&
    [[ "$cmd" == *"--port $ROUTER_PORT"* ]] &&
    [[ "$cmd" == *"--default-model $DEFAULT_MODEL"* ]]
}

router_model_matches() {
  json_model_matches "http://$router_host" "$DEFAULT_MODEL"
}

router_identity_matches() {
  local pid
  pid="$(port_owner_pid "$ROUTER_PORT")"
  [[ -n "$pid" ]] || return 1
  is_expected_router_pid "$pid" && router_model_matches
}

router_healthy() {
  curl -fsS --max-time 2 "http://$router_host/health" >/dev/null 2>&1
}

warm_default_model() {
  local payload
  payload="$(cat <<JSON
{"model":"$DEFAULT_MODEL","input":[{"role":"user","content":[{"type":"input_text","text":"reply with exactly ok"}]}]}
JSON
)"
  curl -fsS --max-time 180 \
    -H 'content-type: application/json' \
    -d "$payload" \
    "http://$router_host/v1/responses" >/dev/null
}

start_router() {
  nohup python3 "$ROOT_DIR/scripts/routers/codex_local_router.py" \
    --host "$ROUTER_HOST" \
    --port "$ROUTER_PORT" \
    --default-model "$DEFAULT_MODEL" \
    --state-dir "$STATE_DIR" \
    --log-dir "$LOG_DIR" \
    >"$ROUTER_LOG_FILE" 2>&1 &
  echo "$!" >"$ROUTER_PID_FILE"
}

wait_for_router() {
  local attempts=60
  while (( attempts > 0 )); do
    if router_healthy && router_model_matches; then
      return 0
    fi
    sleep 1
    ((attempts-=1))
  done
  return 1
}

ensure_router() {
  local owner_pid owner_cmd tracked_pid

  if router_healthy; then
    if router_identity_matches; then
      return 0
    fi
    owner_pid="$(port_owner_pid "$ROUTER_PORT")"
    owner_cmd="$(pid_cmdline "$owner_pid")"
    if [[ "$owner_cmd" == *"codex_local_router.py"* ]]; then
      kill "$owner_pid" 2>/dev/null || true
      rm -f "$ROUTER_PID_FILE"
    else
      echo "error: router port $ROUTER_PORT is already in use by pid $owner_pid" >&2
      echo "cmd: $owner_cmd" >&2
      echo "stop that process or change MARATHON_PROXY_PORT before launching Codex." >&2
      exit 1
    fi
  fi

  if [[ -f "$ROUTER_PID_FILE" ]]; then
    tracked_pid="$(cat "$ROUTER_PID_FILE" 2>/dev/null || true)"
    if [[ -n "$tracked_pid" ]] && kill -0 "$tracked_pid" 2>/dev/null; then
      if is_expected_router_pid "$tracked_pid"; then
        echo "waiting for existing router pid $tracked_pid..." >&2
      else
        kill "$tracked_pid" 2>/dev/null || true
        rm -f "$ROUTER_PID_FILE"
        start_router
      fi
    else
      rm -f "$ROUTER_PID_FILE"
      start_router
    fi
  else
    start_router
  fi

  if ! wait_for_router; then
    echo "error: Codex local router failed to become ready at http://$router_host" >&2
    echo "router log: $ROUTER_LOG_FILE" >&2
    if [[ -f "$ROUTER_PID_FILE" ]]; then
      tracked_pid="$(cat "$ROUTER_PID_FILE" 2>/dev/null || true)"
      if [[ -n "$tracked_pid" ]]; then
        kill "$tracked_pid" 2>/dev/null || true
      fi
      rm -f "$ROUTER_PID_FILE"
    fi
    exit 1
  fi
}

export CODEX_OSS_BASE_URL="http://$ROUTER_HOST:$ROUTER_PORT/v1"
export CODEX_EXTRA_MODELS_PATH="${CODEX_EXTRA_MODELS_PATH:-$LOCAL_MODELS_FILE}"

if [[ "${MARATHON_USE_USER_CONFIG:-${CODEX_QWEN_USE_USER_CONFIG:-0}}" != "1" ]]; then
  export CODEX_HOME="${MARATHON_CODEX_HOME:-${CODEX_QWEN_CODEX_HOME:-$ROOT_DIR/.marathon/codex-home}}"
  mkdir -p "$CODEX_HOME"
fi

ensure_codex_home_config() {
  local config_path
  config_path="${CODEX_HOME:-}/config.toml"
  [[ -n "$config_path" ]] || return 0
  touch "$config_path"
  if rg -q '^tui\.status_line = \["model-name", "context-remaining", "context-window-size", "used-tokens"\]$' "$config_path" 2>/dev/null; then
    perl -0pi -e 's/^tui\.status_line = \["model-name", "context-remaining", "context-window-size", "used-tokens"\]$/tui.status_line = ["model-name", "context-remaining", "context-window-size", "context-tokens"]/m' "$config_path"
    return 0
  fi
  if ! rg -q '^(status_line|tui\.status_line)\s*=' "$config_path" 2>/dev/null; then
    local tmp
    tmp="$(mktemp)"
    {
      printf 'tui.status_line = ["model-name", "context-remaining", "context-window-size", "context-tokens"]\n'
      cat "$config_path"
    } >"$tmp"
    mv "$tmp" "$config_path"
  fi
}

ensure_router
ensure_codex_home_config

if ! warm_default_model; then
  echo "error: default model '$DEFAULT_MODEL' did not answer a warm-up request" >&2
  echo "router log: $ROUTER_LOG_FILE" >&2
  exit 1
fi

COMMON_ARGS=(
  --oss
  --local-provider lmstudio
  -m "$DEFAULT_MODEL"
  -c 'web_search="disabled"'
  --disable image_generation
  --disable apps
  --disable plugins
  --disable tool_search
)

if [[ "${1:-}" == "exec" ]]; then
  shift
  exec "$CODEX_BIN" exec --ignore-user-config "${COMMON_ARGS[@]}" "$@"
fi

exec "$CODEX_BIN" "${COMMON_ARGS[@]}" "$@"
