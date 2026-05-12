#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROUTER_HOST="${MARATHON_PROXY_HOST:-127.0.0.1}"
ROUTER_PORT="${MARATHON_PROXY_PORT:-18111}"
SEARXNG_URL="${MARATHON_SEARXNG_URL:-http://127.0.0.1:18093}"
MODELS_DIR="${MARATHON_MODELS_DIR:-$HOME/models}"

failures=0
warnings=0

pass() {
  printf 'ok    %s\n' "$1"
}

warn() {
  warnings=$((warnings + 1))
  printf 'warn  %s\n' "$1"
}

fail() {
  failures=$((failures + 1))
  printf 'fail  %s\n' "$1"
}

info() {
  printf 'info  %s\n' "$1"
}

have() {
  command -v "$1" >/dev/null 2>&1
}

bytes_human() {
  local path="$1"
  if [[ -e "$path" ]]; then
    du -sh "$path" 2>/dev/null | awk '{print $1}'
  else
    printf '0'
  fi
}

port_owner() {
  local port="$1"
  if have ss; then
    ss -ltnp "( sport = :$port )" 2>/dev/null | sed -n 's/.*pid=\([0-9]\+\).*/pid \1/p' | head -n1
  elif have lsof; then
    local pid
    pid="$(lsof -nP -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null | head -n1 || true)"
    [[ -n "$pid" ]] && printf 'pid %s\n' "$pid"
  fi
}

json_value() {
  local json="$1"
  local expr="$2"
  HEALTH_JSON="$json" python3 - "$expr" <<'PY'
import json
import os
import sys

try:
    value = json.loads(os.environ["HEALTH_JSON"])
except Exception:
    raise SystemExit(1)

for part in sys.argv[1].split("."):
    if not isinstance(value, dict):
        raise SystemExit(1)
    value = value.get(part)
    if value is None:
        raise SystemExit(1)

print(value)
PY
}

model_exists() {
  local env_name="$1"
  local relative="$2"
  local path="${!env_name:-}"
  [[ -n "$path" ]] || path="$MODELS_DIR/$relative"
  [[ -f "$path" ]]
}

printf 'Marathon doctor\n'
printf 'root: %s\n\n' "$ROOT_DIR"

if [[ -x "$ROOT_DIR/bin/marathon" ]]; then
  pass "launcher exists: $ROOT_DIR/bin/marathon"
else
  fail "launcher missing or not executable"
fi

if [[ -x "$ROOT_DIR/.marathon/codex-target/debug/codex" ]]; then
  pass "patched Codex binary exists"
else
  warn "patched Codex binary missing; run: marathon build-codex"
fi

if [[ -x "$ROOT_DIR/.marathon/llama.cpp-build/bin/llama-server" ]]; then
  pass "llama-server exists"
else
  warn "llama-server missing; run: marathon setup-llama or marathon build-llama"
fi

if [[ -x "$ROOT_DIR/.marathon/venv/bin/python3" ]]; then
  pass "router Python venv exists"
else
  warn "router Python venv missing; run: marathon setup-deps"
fi

if have python3; then
  pass "python3 available: $(python3 --version 2>&1)"
else
  fail "python3 not found"
fi

if have docker; then
  pass "docker available"
else
  warn "docker not found; web search setup needs Docker"
fi

if have nvidia-smi; then
  pass "nvidia-smi available"
  gpu_info="$(nvidia-smi --query-gpu=index,name,memory.used,memory.total,power.draw,power.limit --format=csv,noheader,nounits 2>&1 || true)"
  if [[ "$gpu_info" == *"NVIDIA-SMI has failed"* || -z "$gpu_info" ]]; then
    warn "nvidia-smi could not read GPU state: ${gpu_info%%$'\n'*}"
  else
    while IFS= read -r line; do
      [[ -n "$line" ]] && info "gpu: $line"
    done <<<"$gpu_info"
  fi
else
  warn "nvidia-smi not found; GPU health cannot be checked"
fi

if model_exists QWEN36_27B_GGUF "Qwen3.6-27B-GGUF/qwen3.6-27b-q4_k_m.gguf"; then
  pass "Qwen 3.6 27B model found"
else
  warn "Qwen 3.6 27B model not found under $MODELS_DIR"
fi

if model_exists QWEN36_35B_A3B_GGUF "Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf"; then
  pass "Qwen 3.6 35B A3B model found"
else
  info "Qwen 3.6 35B A3B model not found; optional"
fi

if model_exists QWOPUS36_35B_A3B_GGUF "Qwopus3.6-35B-A3B-v1-GGUF/Qwopus3.6-35B-A3B-v1-Q4_K_M.gguf"; then
  pass "Qwopus 35B A3B model found"
else
  info "Qwopus 35B A3B model not found; optional"
fi

if model_exists GEMMA4_26B_A4B_GGUF "gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-Q4_K_M.gguf"; then
  pass "Gemma 4 26B A4B model found"
else
  info "Gemma 4 26B A4B model not found; optional"
fi

router_url="http://$ROUTER_HOST:$ROUTER_PORT"
health_json="$(curl -fsS --max-time 2 "$router_url/health" 2>/dev/null || true)"
if [[ -n "$health_json" ]]; then
  current_model="$(json_value "$health_json" current_model 2>/dev/null || true)"
  backend_status="$(json_value "$health_json" backend_health.status 2>/dev/null || true)"
  pass "router reachable: $router_url"
  if [[ "$backend_status" == "ok" ]]; then
    pass "backend healthy: ${current_model:-unknown}"
  else
    fail "backend status is ${backend_status:-unknown}; run: marathon backend logs"
  fi
else
  owner="$(port_owner "$ROUTER_PORT" || true)"
  if [[ -n "$owner" ]]; then
    fail "router not healthy, but port $ROUTER_PORT is occupied by $owner"
  else
    warn "backend not running; run: marathon backend start 128k-single"
  fi
fi

if curl -fsS --max-time 2 "$SEARXNG_URL/healthz" >/dev/null 2>&1; then
  pass "SearXNG reachable: $SEARXNG_URL"
else
  warn "SearXNG not reachable; run: marathon search up or set MARATHON_WEB_SEARCH_MODE=disabled"
fi

slot_dir="$ROOT_DIR/.marathon/llama-slots"
log_dir="$ROOT_DIR/logs"
state_dir="$ROOT_DIR/.marathon/state"
info "slot snapshots: $(bytes_human "$slot_dir") at $slot_dir"
info "logs: $(bytes_human "$log_dir") at $log_dir"
info "router state: $(bytes_human "$state_dir") at $state_dir"

printf '\n'
if (( failures > 0 )); then
  printf 'Doctor found %d failure(s) and %d warning(s).\n' "$failures" "$warnings"
  exit 1
fi

printf 'Doctor passed with %d warning(s).\n' "$warnings"
