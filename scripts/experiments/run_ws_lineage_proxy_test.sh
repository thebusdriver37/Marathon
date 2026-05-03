#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_PATH="${WS_PROXY_EXPERIMENT_MODEL_PATH:-$HOME/models/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf}"
MODEL_ALIAS="${WS_PROXY_EXPERIMENT_MODEL_ALIAS:-qwen3.5-4b-ws-exp}"
BACKEND_PORT="${WS_PROXY_BACKEND_PORT:-19094}"
PROXY_PORT="${WS_PROXY_PORT:-19114}"
THREADS="${WS_PROXY_THREADS:-12}"
SLOT_SAVE_DIR="${WS_PROXY_SLOT_SAVE_DIR:-$ROOT_DIR/.marathon/ws-lineage-slot-saves}"
LOG_DIR="${WS_PROXY_LOG_DIR:-$ROOT_DIR/logs/ws-lineage-proxy}"
BACKEND_LOG="$LOG_DIR/backend.log"
PROXY_LOG="$LOG_DIR/proxy.log"
TRACE_LOG="$LOG_DIR/responses_ws_lineage_proxy_trace.jsonl"
BACKEND_PID=""
PROXY_PID=""

if [[ -z "${LLAMACPP_BIN:-}" ]]; then
  if [[ -x "$ROOT_DIR/.marathon/llama.cpp-build/bin/llama-server" ]]; then
    export LLAMACPP_BIN="$ROOT_DIR/.marathon/llama.cpp-build/bin/llama-server"
  elif [[ -x "$ROOT_DIR/third_party/llama.cpp/build/bin/llama-server" ]]; then
    export LLAMACPP_BIN="$ROOT_DIR/third_party/llama.cpp/build/bin/llama-server"
  elif [[ -x "$ROOT_DIR/../vLLM_inference/third_party/llama.cpp/build/bin/llama-server" ]]; then
    export LLAMACPP_BIN="$ROOT_DIR/../vLLM_inference/third_party/llama.cpp/build/bin/llama-server"
  fi
fi

mkdir -p "$LOG_DIR"
rm -f "$TRACE_LOG"
mkdir -p "$SLOT_SAVE_DIR"
rm -f "$SLOT_SAVE_DIR"/*.bin 2>/dev/null || true

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "error: experiment model not found: $MODEL_PATH" >&2
  exit 1
fi

port_owner_pid() {
  local port="$1"
  ss -ltnp "( sport = :$port )" 2>/dev/null | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | head -n1
}

wait_http() {
  local url="$1"
  local attempts="${2:-120}"
  while (( attempts > 0 )); do
    if curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
    ((attempts-=1))
  done
  return 1
}

cleanup() {
  local code=$?
  if [[ -n "$PROXY_PID" ]] && kill -0 "$PROXY_PID" 2>/dev/null; then
    kill "$PROXY_PID" 2>/dev/null || true
    wait "$PROXY_PID" 2>/dev/null || true
  fi
  if [[ -n "$BACKEND_PID" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    kill "$BACKEND_PID" 2>/dev/null || true
    wait "$BACKEND_PID" 2>/dev/null || true
  fi
  exit "$code"
}
trap cleanup EXIT

if [[ -n "$(port_owner_pid "$BACKEND_PORT")" ]]; then
  echo "error: backend port $BACKEND_PORT already in use" >&2
  exit 1
fi
if [[ -n "$(port_owner_pid "$PROXY_PORT")" ]]; then
  echo "error: proxy port $PROXY_PORT already in use" >&2
  exit 1
fi

nohup "$ROOT_DIR/scripts/ops/run_llama_server.sh" \
  --model "$MODEL_PATH" \
  --alias "$MODEL_ALIAS" \
  --host 127.0.0.1 \
  --port "$BACKEND_PORT" \
  --ctx 32768 \
  --ngl 999 \
  --split-mode layer \
  --tensor-split 1,1 \
  --threads "$THREADS" \
  --batch 128 \
  --ubatch 64 \
  --flash-attn on \
  --parallel 1 \
  --slot-save-path "$SLOT_SAVE_DIR" \
  >"$BACKEND_LOG" 2>&1 &
BACKEND_PID=$!

echo "started backend pid $BACKEND_PID on port $BACKEND_PORT"
wait_http "http://127.0.0.1:$BACKEND_PORT/health" 180

nohup python3 "$ROOT_DIR/scripts/experiments/responses_ws_lineage_proxy.py" \
  --host 127.0.0.1 \
  --port "$PROXY_PORT" \
  --backend "http://127.0.0.1:$BACKEND_PORT" \
  --model "$MODEL_ALIAS" \
  --display-name "Qwen3.5 4B WS Experiment" \
  --description "Experimental websocket lineage proxy" \
  --slot-id 0 \
  --slot-save-dir "$SLOT_SAVE_DIR" \
  --log-dir "$LOG_DIR" \
  >"$PROXY_LOG" 2>&1 &
PROXY_PID=$!

echo "started proxy pid $PROXY_PID on port $PROXY_PORT"
wait_http "http://127.0.0.1:$PROXY_PORT/health" 60

python3 "$ROOT_DIR/scripts/experiments/test_responses_ws_lineage_proxy.py" \
  --ws-url "ws://127.0.0.1:$PROXY_PORT/v1/responses" \
  --model "$MODEL_ALIAS" \
  --trace-log "$TRACE_LOG"

echo
printf 'trace log: %s\n' "$TRACE_LOG"
printf 'backend log: %s\n' "$BACKEND_LOG"
printf 'proxy log: %s\n' "$PROXY_LOG"
