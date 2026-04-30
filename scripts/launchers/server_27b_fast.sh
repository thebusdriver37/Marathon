#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODELS_DIR="${MARATHON_MODELS_DIR:-${MODELS_DIR:-$HOME/models}}"
MODEL_PATH="${MODEL_PATH:-${QWEN36_27B_GGUF:-$MODELS_DIR/Qwen3.6-27B-GGUF/qwen3.6-27b-q4_k_m.gguf}}"
MODEL_ALIAS="${MODEL_ALIAS:-qwen3.6-27b-q4}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18090}"
CTX_SIZE="${CTX_SIZE:-32768}"
GPU_DEVICES="${GPU_DEVICES:-0}"
THREADS="${THREADS:-24}"
BATCH="${BATCH:-2048}"
UBATCH="${UBATCH:-512}"
CACHE_TYPE_K="${CACHE_TYPE_K:-q8_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q8_0}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
SPLIT_MODE="${SPLIT_MODE:-none}"
TENSOR_SPLIT="${TENSOR_SPLIT:-}"
MAIN_GPU="${MAIN_GPU:-0}"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "error: model not found: $MODEL_PATH" >&2
  echo "set QWEN36_27B_GGUF or MARATHON_MODELS_DIR to your downloaded GGUF location." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"

args=(
  --model "$MODEL_PATH"
  --alias "$MODEL_ALIAS"
  --host "$HOST"
  --port "$PORT"
  --ctx "$CTX_SIZE"
  --ngl "$N_GPU_LAYERS"
  --split-mode "$SPLIT_MODE"
  --threads "$THREADS"
  --batch "$BATCH"
  --ubatch "$UBATCH"
  --cache-type-k "$CACHE_TYPE_K"
  --cache-type-v "$CACHE_TYPE_V"
  --flash-attn on
  --swa-full
  --parallel 1
  --reasoning-format none
  --reasoning-budget 0
)

if [[ "$SPLIT_MODE" == "none" || "$SPLIT_MODE" == "row" ]]; then
  args+=(--main-gpu "$MAIN_GPU")
fi

if [[ -n "$TENSOR_SPLIT" ]]; then
  args+=(--tensor-split "$TENSOR_SPLIT")
fi

exec "$ROOT_DIR/scripts/ops/run_llama_server.sh" "${args[@]}" "$@"
