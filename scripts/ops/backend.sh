#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROUTER_HOST="${MARATHON_PROXY_HOST:-127.0.0.1}"
ROUTER_PORT="${MARATHON_PROXY_PORT:-18111}"
STATE_DIR="${MARATHON_ROUTER_STATE_DIR:-$ROOT_DIR/.marathon/state}"
LOG_DIR="${MARATHON_LOG_DIR:-$ROOT_DIR/logs}"
ROUTER_PID_FILE="$STATE_DIR/codex-local-router.pid"
ROUTER_LOG="$LOG_DIR/codex_local_router.log"
DEFAULT_PROFILE="${MARATHON_DEFAULT_PROFILE:-128k-single}"
START_TIMEOUT_SECONDS="${MARATHON_BACKEND_START_TIMEOUT_SECONDS:-360}"

usage() {
  cat <<USAGE
Usage: marathon backend <command> [profile] [path]

Commands:
  start [profile] [path]  Start the Marathon router and model backend
  stop                 Stop the Marathon router and model backends
  restart [profile]    Stop, then start the backend
  status               Show router and model backend health
  logs [target] [-f]   Tail logs for router or a model profile

Profiles:
  128k-single          Qwen3.6 27B Q4 128K on one GPU (default)
  128k                 Qwen3.6 27B Q4 128K
  fast                 Qwen3.6 27B Q4 32K
  a3b                  Qwen3.6 35B A3B 128K
  qwopus               Qwopus3.6 35B A3B v1 128K
  gemma                Gemma 4 26B A4B IT 128K
  custom [path]        Any local GGUF model served by llama.cpp

Examples:
  marathon backend start 128k-single
  marathon backend start qwopus
  marathon backend start gemma
  marathon backend start custom /path/to/model.gguf
  MARATHON_MODEL_SLUG=local-coder marathon backend start /path/to/model.gguf
  marathon backend status
  marathon backend logs -f
  marathon backend stop
USAGE
}

custom_slug() {
  printf '%s\n' "${MARATHON_MODEL_SLUG:-custom}"
}

custom_port() {
  printf '%s\n' "${MARATHON_MODEL_PORT:-18095}"
}

profile_slug() {
  case "${1:-$DEFAULT_PROFILE}" in
    128k-single|qwen3.6-27b-q4-128k-single)
      printf '%s\n' "qwen3.6-27b-q4-128k-single"
      ;;
    128k|qwen3.6-27b-q4-128k)
      printf '%s\n' "qwen3.6-27b-q4-128k"
      ;;
    fast|qwen3.6-27b-q4)
      printf '%s\n' "qwen3.6-27b-q4"
      ;;
    a3b|qwen3.6-35b-a3b)
      printf '%s\n' "qwen3.6-35b-a3b"
      ;;
    qwopus|qwopus3.6-35b-a3b-v1)
      printf '%s\n' "qwopus3.6-35b-a3b-v1"
      ;;
    gemma|gemma4|gemma4-26b-a4b-it-128k)
      printf '%s\n' "gemma4-26b-a4b-it-128k"
      ;;
    custom)
      custom_slug
      ;;
    *)
      echo "error: unknown backend profile: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
}

profile_port() {
  local custom
  custom="$(custom_slug)"
  case "$1" in
    qwen3.6-27b-q4-128k-single) printf '%s\n' "18094" ;;
    qwen3.6-27b-q4-128k) printf '%s\n' "18091" ;;
    qwen3.6-27b-q4) printf '%s\n' "18090" ;;
    qwen3.6-35b-a3b) printf '%s\n' "18092" ;;
    qwopus3.6-35b-a3b-v1) printf '%s\n' "18096" ;;
    gemma4-26b-a4b-it-128k) printf '%s\n' "18097" ;;
    "$custom") custom_port ;;
    *) return 1 ;;
  esac
}

profile_log() {
  printf '%s/%s.log\n' "$LOG_DIR" "$1"
}

looks_like_model_path() {
  case "$1" in
    *.gguf|*.GGUF|/*|./*|../*|~/*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

absolute_path() {
  local path="$1"
  case "$path" in
    "~/"*) path="$HOME/${path#~/}" ;;
  esac
  if [[ "$path" == /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s/%s\n' "$PWD" "$path"
  fi
}

validate_custom_profile() {
  if [[ -z "${MARATHON_MODEL_PATH:-}" ]]; then
    echo "error: custom model path is required." >&2
    echo "run: marathon backend start custom /path/to/model.gguf" >&2
    echo "or set MARATHON_MODEL_PATH=/path/to/model.gguf" >&2
    exit 1
  fi
  if [[ ! -f "$MARATHON_MODEL_PATH" ]]; then
    echo "error: custom model not found: $MARATHON_MODEL_PATH" >&2
    exit 1
  fi
  if [[ ! "$(custom_slug)" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "error: MARATHON_MODEL_SLUG may only contain letters, numbers, '.', '_', and '-'" >&2
    exit 1
  fi
  if [[ ! "$(custom_port)" =~ ^[0-9]+$ ]]; then
    echo "error: MARATHON_MODEL_PORT must be a number" >&2
    exit 1
  fi
}

router_base_url() {
  printf 'http://%s:%s\n' "$ROUTER_HOST" "$ROUTER_PORT"
}

port_owner_pid() {
  local port="$1"
  local pid=""
  if command -v ss >/dev/null 2>&1; then
    pid="$(ss -ltnp "( sport = :$port )" 2>/dev/null | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | head -n1)"
  elif command -v lsof >/dev/null 2>&1; then
    pid="$(lsof -nP -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null | head -n1)"
  fi
  [[ "$pid" =~ ^[0-9]+$ ]] && printf '%s\n' "$pid"
}

pid_cmdline() {
  ps -p "$1" -o cmd= 2>/dev/null || true
}

terminate_pid() {
  local pid="$1"
  local label="$2"
  [[ -n "$pid" ]] || return 0
  if ! kill -0 "$pid" 2>/dev/null; then
    return 0
  fi
  echo "stopping $label pid $pid..."
  kill "$pid" 2>/dev/null || true
  for _ in {1..12}; do
    if ! kill -0 "$pid" 2>/dev/null; then
      return 0
    fi
    sleep 1
  done
  kill -9 "$pid" 2>/dev/null || true
}

stop_router() {
  local pid=""
  if [[ -f "$ROUTER_PID_FILE" ]]; then
    pid="$(tr -cd '0-9' <"$ROUTER_PID_FILE" || true)"
    if [[ -n "$pid" ]]; then
      terminate_pid "$pid" "router"
    fi
  fi

  pid="$(port_owner_pid "$ROUTER_PORT" || true)"
  if [[ -n "$pid" ]]; then
    local cmd
    cmd="$(pid_cmdline "$pid")"
    if [[ "$cmd" == *"codex_local_router.py"* ]]; then
      terminate_pid "$pid" "router"
    else
      echo "warning: router port $ROUTER_PORT is owned by another process: $cmd" >&2
    fi
  fi

  rm -f "$ROUTER_PID_FILE"
}

stop_model_backends() {
  local slug port pid cmd
  for slug in qwen3.6-27b-q4-128k-single qwen3.6-27b-q4-128k qwen3.6-27b-q4 qwen3.6-35b-a3b qwopus3.6-35b-a3b-v1 gemma4-26b-a4b-it-128k "$(custom_slug)"; do
    port="$(profile_port "$slug" 2>/dev/null || true)"
    [[ -n "$port" ]] || continue
    pid="$(port_owner_pid "$port" || true)"
    if [[ -z "$pid" ]]; then
      rm -f "$STATE_DIR/$slug.pid"
      continue
    fi
    cmd="$(pid_cmdline "$pid")"
    if [[ "$cmd" == *"llama-server"* ]]; then
      terminate_pid "$pid" "$slug"
      rm -f "$STATE_DIR/$slug.pid"
    else
      echo "warning: model port $port is owned by another process: $cmd" >&2
    fi
  done

  local pid_file
  for pid_file in "$STATE_DIR"/*.pid; do
    [[ -e "$pid_file" ]] || continue
    [[ "$(basename "$pid_file")" != "codex-local-router.pid" ]] || continue
    slug="$(basename "$pid_file" .pid)"
    pid="$(tr -cd '0-9' <"$pid_file" || true)"
    [[ -n "$pid" ]] || continue
    cmd="$(pid_cmdline "$pid")"
    if [[ "$cmd" == *"llama-server"* ]]; then
      terminate_pid "$pid" "$slug"
      rm -f "$pid_file"
    fi
  done
}

stop_backend() {
  mkdir -p "$STATE_DIR"
  stop_router
  stop_model_backends
}

router_python() {
  if [[ -n "${MARATHON_ROUTER_PYTHON:-}" ]]; then
    if [[ -x "$MARATHON_ROUTER_PYTHON" ]] || command -v "$MARATHON_ROUTER_PYTHON" >/dev/null 2>&1; then
      printf '%s\n' "$MARATHON_ROUTER_PYTHON"
      return 0
    fi
    echo "error: MARATHON_ROUTER_PYTHON is not executable: $MARATHON_ROUTER_PYTHON" >&2
    exit 1
  fi

  local venv_python="$ROOT_DIR/.marathon/venv/bin/python3"
  if [[ -x "$venv_python" ]]; then
    printf '%s\n' "$venv_python"
    return 0
  fi
  echo "error: router Python environment is missing." >&2
  echo "run: marathon setup-deps" >&2
  exit 1
}

health_json() {
  curl -fsS --max-time 2 "$(router_base_url)/health" 2>/dev/null || true
}

health_field() {
  local json="$1"
  local expr="$2"
  [[ -n "$json" ]] || return 1
  HEALTH_JSON="$json" python3 - "$expr" <<'PY'
import json
import os
import sys

try:
    value = json.loads(os.environ["HEALTH_JSON"])
except Exception:
    raise SystemExit(1)

for part in sys.argv[1].split("."):
    if not part:
        continue
    if not isinstance(value, dict):
        raise SystemExit(1)
    value = value.get(part)
    if value is None:
        raise SystemExit(1)

print(value)
PY
}

start_router() {
  local slug="$1"
  local py
  py="$(router_python)"

  mkdir -p "$STATE_DIR" "$LOG_DIR"
  : >"$ROUTER_LOG"

  echo "starting Marathon backend profile: $slug"
  if command -v setsid >/dev/null 2>&1; then
    PYTHONDONTWRITEBYTECODE=1 setsid "$py" "$ROOT_DIR/scripts/routers/codex_local_router.py" \
      --host "$ROUTER_HOST" \
      --port "$ROUTER_PORT" \
      --default-model "$slug" \
      --state-dir "$STATE_DIR" \
      --log-dir "$LOG_DIR" >>"$ROUTER_LOG" 2>&1 &
  else
    PYTHONDONTWRITEBYTECODE=1 nohup "$py" "$ROOT_DIR/scripts/routers/codex_local_router.py" \
      --host "$ROUTER_HOST" \
      --port "$ROUTER_PORT" \
      --default-model "$slug" \
      --state-dir "$STATE_DIR" \
      --log-dir "$LOG_DIR" >>"$ROUTER_LOG" 2>&1 &
  fi
  echo "$!" >"$ROUTER_PID_FILE"
}

wait_until_ready() {
  local slug="$1"
  local deadline=$((SECONDS + START_TIMEOUT_SECONDS))
  local json current status router_pid

  router_pid="$(cat "$ROUTER_PID_FILE" 2>/dev/null || true)"
  while (( SECONDS < deadline )); do
    if [[ -n "$router_pid" ]] && ! kill -0 "$router_pid" 2>/dev/null; then
      echo "error: router exited before backend became ready." >&2
      echo "router log: $ROUTER_LOG" >&2
      return 1
    fi

    json="$(health_json)"
    current="$(health_field "$json" current_model 2>/dev/null || true)"
    status="$(health_field "$json" backend_health.status 2>/dev/null || true)"
    if [[ "$current" == "$slug" && "$status" == "ok" ]]; then
      echo "backend ready: $(router_base_url) ($slug)"
      return 0
    fi
    sleep 1
  done

  echo "error: backend did not become ready within ${START_TIMEOUT_SECONDS}s." >&2
  echo "router log: $ROUTER_LOG" >&2
  echo "model log: $(profile_log "$slug")" >&2
  return 1
}

start_backend() {
  local profile_arg slug
  profile_arg="${1:-$DEFAULT_PROFILE}"
  if [[ $# -gt 0 ]]; then
    shift
  fi

  if looks_like_model_path "$profile_arg"; then
    export MARATHON_MODEL_PATH
    MARATHON_MODEL_PATH="$(absolute_path "$profile_arg")"
    profile_arg="custom"
  elif [[ "$profile_arg" == "custom" && $# -gt 0 ]]; then
    export MARATHON_MODEL_PATH
    MARATHON_MODEL_PATH="$(absolute_path "$1")"
  fi

  if [[ "$profile_arg" == "custom" ]]; then
    validate_custom_profile
  fi

  slug="$(profile_slug "$profile_arg")"

  local json current status
  json="$(health_json)"
  current="$(health_field "$json" current_model 2>/dev/null || true)"
  status="$(health_field "$json" backend_health.status 2>/dev/null || true)"
  if [[ "$current" == "$slug" && "$status" == "ok" ]]; then
    echo "backend already running: $(router_base_url) ($slug)"
    return 0
  fi

  stop_backend
  start_router "$slug"
  wait_until_ready "$slug"
}

status_backend() {
  local json
  json="$(health_json)"
  if [[ -z "$json" ]]; then
    echo "router: stopped ($(router_base_url))"
    return 1
  fi

  HEALTH_JSON="$json" python3 - "$(router_base_url)" <<'PY'
import json
import os
import sys

payload = json.loads(os.environ["HEALTH_JSON"])
backend = payload.get("backend_health") or {}
status = backend.get("status") or "starting"
current = payload.get("current_model") or "loading"
default = payload.get("default_model") or "unknown"
target = backend.get("target") or ""
message = backend.get("message") or ""

print(f"router: ok ({sys.argv[1]})")
print(f"default model: {default}")
print(f"active model: {current}")
print(f"backend: {status}" + (f" ({target})" if target else ""))
if message and status != "ok":
    print(f"message: {message}")
PY
}

logs_backend() {
  local follow=0
  local target="router"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -f|--follow)
        follow=1
        shift
        ;;
      router|current|128k-single|128k|fast|a3b|qwopus|gemma|custom|qwen3.6-*|qwopus3.6-*|gemma4-*)
        target="$1"
        shift
        ;;
      -*)
        echo "error: unknown logs target or option: $1" >&2
        usage >&2
        exit 1
        ;;
      *)
        target="$1"
        shift
        ;;
    esac
  done

  local log_file json slug
  case "$target" in
    router)
      log_file="$ROUTER_LOG"
      ;;
    current)
      json="$(health_json)"
      slug="$(health_field "$json" current_model 2>/dev/null || true)"
      [[ -n "$slug" ]] || slug="$(profile_slug "$DEFAULT_PROFILE")"
      log_file="$(profile_log "$slug")"
      ;;
    *)
      case "$target" in
        128k-single|128k|fast|a3b|qwopus|gemma|custom|qwen3.6-*|qwopus3.6-*|gemma4-*)
          slug="$(profile_slug "$target")"
          ;;
        *)
          slug="$target"
          ;;
      esac
      log_file="$(profile_log "$slug")"
      ;;
  esac

  if [[ ! -f "$log_file" ]]; then
    echo "log file does not exist yet: $log_file" >&2
    exit 1
  fi

  if [[ "$follow" == "1" ]]; then
    exec tail -n 120 -f "$log_file"
  fi
  exec tail -n 120 "$log_file"
}

cmd="${1:-}"
case "$cmd" in
  start)
    shift
    start_backend "$@"
    ;;
  stop)
    shift || true
    stop_backend
    ;;
  restart)
    shift
    stop_backend
    start_backend "$@"
    ;;
  status)
    shift || true
    status_backend
    ;;
  logs)
    shift
    logs_backend "$@"
    ;;
  help|--help|-h|"")
    usage
    ;;
  *)
    echo "error: unknown backend command: $cmd" >&2
    usage >&2
    exit 1
    ;;
esac
