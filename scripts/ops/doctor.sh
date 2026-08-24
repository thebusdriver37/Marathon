#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROUTER_HOST="${MARATHON_PROXY_HOST:-127.0.0.1}"
ROUTER_PORT="${MARATHON_PROXY_PORT:-18111}"
LLAMA_PORT="${MARATHON_LLAMA_PORT:-8082}"
SEARXNG_URL="${MARATHON_SEARXNG_URL:-http://127.0.0.1:18093}"
SEARXNG_CURL=(curl -fsS)
if [[ "$SEARXNG_URL" =~ ^https?://(127\.0\.0\.1|localhost|\[::1\])([:/]|$) ]]; then
  SEARXNG_CURL+=(-H 'X-Forwarded-For: 127.0.0.1')
fi
AI_ROOT="${MARATHON_AI_ROOT:-$HOME/AI}"
MODELS_DIR="${MARATHON_MODELS_DIR:-$AI_ROOT/models/gguf}"
LLAMACPP_BIN="${LLAMACPP_BIN:-$AI_ROOT/backends/llama.cpp-current/build/bin/llama-server}"
SLOT_DIR="${MARATHON_SLOT_SAVE_ROOT:-$AI_ROOT/cache/marathon/slots}"
ROUTER_STATE_DIR="$AI_ROOT/cache/marathon/router"
DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
PATCHED_CODEX_BIN="${MARATHON_CODEX_BIN:-$DATA_HOME/marathon/bin/codex}"
HERMES_BIN="${MARATHON_HERMES_BIN:-hermes}"

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

# Ask the application for its effective paths so catalog edits and environment
# overrides are interpreted exactly as they are by the runtime. The shell
# defaults above keep Doctor useful enough to report a missing Python install.
if have python3; then
  resolved_paths="$(PYTHONPATH="$ROOT_DIR" python3 - <<'PY' 2>/dev/null || true
from marathon_app.catalog import backends, settings
from marathon_app.runtime import AI_ROOT, ROUTER_STATE_DIR, SLOT_ROOT

print(AI_ROOT)
print(settings().model_root)
print(backends()["upstream"].server)
print(SLOT_ROOT)
print(ROUTER_STATE_DIR)
PY
)"
  if [[ -n "$resolved_paths" ]]; then
    mapfile -t resolved <<<"$resolved_paths"
    if (( ${#resolved[@]} == 5 )); then
      AI_ROOT="${resolved[0]}"
      MODELS_DIR="${resolved[1]}"
      LLAMACPP_BIN="${resolved[2]}"
      SLOT_DIR="${resolved[3]}"
      ROUTER_STATE_DIR="${resolved[4]}"
    fi
  fi
fi

printf 'Marathon doctor\n'
printf 'root: %s\n\n' "$ROOT_DIR"

if [[ -x "$ROOT_DIR/bin/marathon" ]]; then
  pass "launcher exists: $ROOT_DIR/bin/marathon"
else
  fail "launcher missing or not executable"
fi

if [[ -x "$PATCHED_CODEX_BIN" ]]; then
  pass "Marathon Codex available: $($PATCHED_CODEX_BIN --version 2>&1)"
  if [[ -f "$PATCHED_CODEX_BIN.source" ]]; then
    info "Marathon Codex source: $(cat "$PATCHED_CODEX_BIN.source")"
  fi
elif have codex; then
  warn "using stock Codex; run 'marathon build-codex' for Marathon's status meters"
else
  fail "Codex is not installed; run: marathon build-codex"
fi

if have "$HERMES_BIN"; then
  pass "Hermes available: $($HERMES_BIN --version 2>&1 | head -n1)"
else
  warn "Hermes is not installed; Codex and Direct Chat remain available"
fi

if [[ -x "$ROOT_DIR/.marathon/venv/bin/python3" ]]; then
  if PYTHONPATH="$ROOT_DIR" "$ROOT_DIR/.marathon/venv/bin/python3" -c 'import aiohttp, huggingface_hub, prompt_toolkit, rich, marathon_app' 2>/dev/null; then
    pass "private router/UI Python environment is ready"
  else
    fail "private Python environment is incomplete; run: marathon setup-deps"
  fi
else
  fail "private Python environment missing; run: marathon setup-deps"
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

model_report="$(PYTHONPATH="$ROOT_DIR" "$ROOT_DIR/.marathon/venv/bin/python3" - <<'PY' 2>/dev/null || printf '0\n'
from marathon_app.catalog import discover_models, settings

print(len(discover_models()))
for root in settings().model_roots:
    print(root)
PY
)"
mapfile -t model_lines <<<"$model_report"
model_count="${model_lines[0]:-0}"
model_root_count=$(( ${#model_lines[@]} - 1 ))
if [[ "$model_count" =~ ^[0-9]+$ ]] && (( model_count > 0 )); then
  pass "$model_count GGUF model(s) discovered across $model_root_count configured folder(s)"
else
  fail "no GGUF models discovered across $model_root_count configured folder(s)"
fi
for ((index = 1; index < ${#model_lines[@]}; index++)); do
  info "model folder: ${model_lines[$index]}"
done

backend_inventory="$(PYTHONPATH="$ROOT_DIR" "$ROOT_DIR/.marathon/venv/bin/python3" - <<'PY' 2>/dev/null || true
from marathon_app.catalog import backend_environment, backends, discover_models

configured = backends()
seen_backends = set()
seen_files = set()
for model in discover_models():
    for profile in model.family.profiles:
        backend_id = profile.backend or model.family.backend
        backend = configured.get(backend_id)
        if backend_id not in seen_backends:
            seen_backends.add(backend_id)
            if backend is None:
                print(f"backend|{backend_id}||")
            else:
                print(f"backend|{backend_id}|{backend.server}|{backend.worker or ''}")
        if backend is None:
            continue
        for key, value in backend_environment(model, backend).items():
            if not key.endswith("_GGUF") or (key, value) in seen_files:
                continue
            seen_files.add((key, value))
            print(f"file|{key}|{value}|")
PY
)"
while IFS='|' read -r kind identity path extra; do
  [[ -z "$kind" ]] && continue
  if [[ "$kind" == "file" ]]; then
    if [[ -f "$path" ]]; then
      pass "$identity sidecar exists: $path"
    else
      fail "$identity sidecar missing: $path"
    fi
    continue
  fi
  backend_id="$identity"
  server="$path"
  worker="$extra"
  if [[ -x "$server" ]]; then
    pass "$backend_id server exists: $server"
  else
    fail "$backend_id server missing or not executable: ${server:-not configured}"
  fi
  if [[ -n "$worker" ]]; then
    if [[ -x "$worker" ]]; then
      pass "$backend_id worker exists: $worker"
    else
      fail "$backend_id worker missing or not executable: $worker"
    fi
  fi
done <<<"$backend_inventory"

router_url="http://$ROUTER_HOST:$ROUTER_PORT"
health_json="$(curl -fsS --max-time 2 "$router_url/health" 2>/dev/null || true)"
if [[ -n "$health_json" ]]; then
  current_model="$(json_value "$health_json" current_model 2>/dev/null || true)"
  backend_status="$(json_value "$health_json" backend_health.status 2>/dev/null || true)"
  pass "router reachable: $router_url"
  if [[ "$backend_status" == "ok" ]]; then
    pass "backend healthy: ${current_model:-unknown}"
  elif curl -fsS --max-time 2 "http://127.0.0.1:$LLAMA_PORT/v1/models" >/dev/null 2>&1; then
    pass "inference backend is ready and awaiting its first routed request"
  else
    fail "backend status is ${backend_status:-unknown}; run: marathon backend logs"
  fi
else
  owner="$(port_owner "$ROUTER_PORT" || true)"
  if [[ -n "$owner" ]]; then
    fail "router not healthy, but port $ROUTER_PORT is occupied by $owner"
  else
    info "foreground backend is stopped (normal when Marathon is not open)"
  fi
fi

if "${SEARXNG_CURL[@]}" --max-time 2 "$SEARXNG_URL/healthz" >/dev/null 2>&1; then
  pass "SearXNG reachable: $SEARXNG_URL"
  search_json="$("${SEARXNG_CURL[@]}" -G --max-time 20 \
    --data-urlencode 'q=SearXNG documentation official' \
    --data-urlencode 'format=json' \
    "$SEARXNG_URL/search" 2>/dev/null || true)"
  search_probe="$(SEARCH_JSON="$search_json" python3 - <<'PY' 2>/dev/null || true
import json
import os

try:
    payload = json.loads(os.environ["SEARCH_JSON"])
except Exception:
    print("0|invalid JSON response")
    raise SystemExit(0)

results = payload.get("results") if isinstance(payload, dict) else None
count = len(results) if isinstance(results, list) else 0
failures = []
raw_failures = payload.get("unresponsive_engines") if isinstance(payload, dict) else None
if isinstance(raw_failures, list):
    for failure in raw_failures:
        if isinstance(failure, list) and failure:
            name = str(failure[0])
            reason = str(failure[1]) if len(failure) > 1 else "failed"
            failures.append(f"{name}: {reason}")
print(f"{count}|{'; '.join(failures)}")
PY
)"
  IFS='|' read -r search_result_count search_failures <<<"$search_probe"
  if [[ "${search_result_count:-0}" =~ ^[0-9]+$ ]] && (( search_result_count > 0 )); then
    pass "SearXNG functional search returned $search_result_count results"
  else
    warn "SearXNG is reachable but search returned no results${search_failures:+ ($search_failures)}"
  fi
else
  warn "SearXNG not reachable; run: marathon search up or set MARATHON_WEB_SEARCH_MODE=disabled"
fi

log_dir="${XDG_STATE_HOME:-$HOME/.local/state}/marathon/logs"
info "AI root: $AI_ROOT"
info "slot snapshots: $(bytes_human "$SLOT_DIR") at $SLOT_DIR"
info "logs: $(bytes_human "$log_dir") at $log_dir"
info "router state: $(bytes_human "$ROUTER_STATE_DIR") at $ROUTER_STATE_DIR"

printf '\n'
if (( failures > 0 )); then
  printf 'Doctor found %d failure(s) and %d warning(s).\n' "$failures" "$warnings"
  exit 1
fi

printf 'Doctor passed with %d warning(s).\n' "$warnings"
