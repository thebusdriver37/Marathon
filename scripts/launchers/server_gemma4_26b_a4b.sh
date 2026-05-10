#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODELS_DIR="${MARATHON_MODELS_DIR:-${MODELS_DIR:-$HOME/models}}"

MODEL_PATH="${MODEL_PATH:-${GEMMA4_26B_A4B_GGUF:-$MODELS_DIR/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-Q4_K_M.gguf}}"
MODEL_ALIAS="${MODEL_ALIAS:-gemma4-26b-a4b-it-128k}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18097}"
SLOT_SAVE_ROOT="${MARATHON_SLOT_SAVE_ROOT:-$ROOT_DIR/.marathon/llama-slots}"
SLOT_SAVE_PATH="${SLOT_SAVE_PATH:-$SLOT_SAVE_ROOT/$MODEL_ALIAS}"
CTX_SIZE="${CTX_SIZE:-131072}"
GPU_DEVICES="${GPU_DEVICES:-0}"
THREADS="${THREADS:-24}"
BATCH="${BATCH:-1024}"
UBATCH="${UBATCH:-256}"
CACHE_TYPE_K="${CACHE_TYPE_K:-q8_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q8_0}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
MAIN_GPU="${MAIN_GPU:-0}"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "error: model not found: $MODEL_PATH" >&2
  echo "set GEMMA4_26B_A4B_GGUF or MARATHON_MODELS_DIR to your downloaded GGUF location." >&2
  exit 1
fi

mkdir -p "$SLOT_SAVE_PATH"

export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"

exec "$ROOT_DIR/scripts/ops/run_llama_server.sh" \
  --model "$MODEL_PATH" \
  --alias "$MODEL_ALIAS" \
  --host "$HOST" \
  --port "$PORT" \
  --ctx "$CTX_SIZE" \
  --ngl "$N_GPU_LAYERS" \
  --split-mode none \
  --main-gpu "$MAIN_GPU" \
  --threads "$THREADS" \
  --batch "$BATCH" \
  --ubatch "$UBATCH" \
  --flash-attn on \
  --reasoning-format deepseek \
  --reasoning off \
  --reasoning-budget 0 \
  --parallel 1 \
  --cache-type-k "$CACHE_TYPE_K" \
  --cache-type-v "$CACHE_TYPE_V" \
  --slot-save-path "$SLOT_SAVE_PATH" \
  --jinja \
  "$@"
