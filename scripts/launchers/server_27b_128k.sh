#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODELS_DIR="${MARATHON_MODELS_DIR:-${MODELS_DIR:-$HOME/models}}"

MODEL_PATH="${MODEL_PATH:-${QWEN36_27B_GGUF:-$MODELS_DIR/Qwen3.6-27B-GGUF/qwen3.6-27b-q4_k_m.gguf}}"
MODEL_ALIAS="${MODEL_ALIAS:-qwen3.6-27b-q4-128k}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18091}"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "error: model not found: $MODEL_PATH" >&2
  echo "set QWEN36_27B_GGUF or MARATHON_MODELS_DIR to your downloaded GGUF location." >&2
  exit 1
fi

exec "$ROOT_DIR/scripts/ops/run_llama_server.sh" \
  --model "$MODEL_PATH" \
  --alias "$MODEL_ALIAS" \
  --host "$HOST" \
  --port "$PORT" \
  --ctx "${CTX_SIZE:-131072}" \
  --ngl "${N_GPU_LAYERS:-999}" \
  --split-mode "${SPLIT_MODE:-layer}" \
  --tensor-split "${TENSOR_SPLIT:-1,1}" \
  --threads "${THREADS:-24}" \
  --batch "${BATCH:-256}" \
  --ubatch "${UBATCH:-64}" \
  --flash-attn on \
  --reasoning-format none \
  --reasoning-budget 0 \
  --parallel 1 \
  --cache-type-k "${CACHE_TYPE_K:-q8_0}" \
  --cache-type-v "${CACHE_TYPE_V:-q8_0}" \
  --swa-full \
  "$@"
