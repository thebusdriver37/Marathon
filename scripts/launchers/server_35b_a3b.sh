#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODELS_DIR="${MARATHON_MODELS_DIR:-${MODELS_DIR:-$HOME/models}}"
MODEL_PATH="${MODEL_PATH:-${QWEN36_35B_A3B_GGUF:-$MODELS_DIR/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf}}"
MODEL_ALIAS="${MODEL_ALIAS:-qwen3.6-35b-a3b}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18092}"
GPU_DEVICES="${GPU_DEVICES:-1}"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "error: model not found: $MODEL_PATH" >&2
  echo "set QWEN36_35B_A3B_GGUF or MARATHON_MODELS_DIR to your downloaded GGUF location." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"

exec "$ROOT_DIR/scripts/ops/run_llama_server.sh" \
  --model "$MODEL_PATH" \
  --alias "$MODEL_ALIAS" \
  --host "$HOST" \
  --port "$PORT" \
  --ctx "${CTX_SIZE:-32768}" \
  --ngl "${N_GPU_LAYERS:-999}" \
  --split-mode "${SPLIT_MODE:-none}" \
  --threads "${THREADS:-24}" \
  --batch "${BATCH:-640}" \
  --ubatch "${UBATCH:-160}" \
  --flash-attn on \
  --reasoning-format none \
  --reasoning-budget 0 \
  --parallel 1 \
  --cache-type-k "${CACHE_TYPE_K:-q8_0}" \
  --cache-type-v "${CACHE_TYPE_V:-q8_0}" \
  --swa-full \
  "$@"
