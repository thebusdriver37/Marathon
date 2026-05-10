#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-${MARATHON_MODEL_PATH:-}}"
MODEL_ALIAS="${MODEL_ALIAS:-${MARATHON_MODEL_SLUG:-custom}}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-${MARATHON_MODEL_PORT:-18095}}"
SLOT_SAVE_ROOT="${MARATHON_SLOT_SAVE_ROOT:-$ROOT_DIR/.marathon/llama-slots}"
SLOT_SAVE_PATH="${SLOT_SAVE_PATH:-$SLOT_SAVE_ROOT/$MODEL_ALIAS}"
CTX_SIZE="${CTX_SIZE:-${MARATHON_MODEL_CONTEXT:-${MARATHON_CONTEXT:-32768}}}"
N_GPU_LAYERS="${N_GPU_LAYERS:-${MARATHON_N_GPU_LAYERS:-999}}"
SPLIT_MODE="${SPLIT_MODE:-${MARATHON_SPLIT_MODE:-none}}"
TENSOR_SPLIT="${TENSOR_SPLIT:-${MARATHON_TENSOR_SPLIT:-}}"
THREADS="${THREADS:-${MARATHON_THREADS:-$(nproc)}}"
BATCH="${BATCH:-${MARATHON_BATCH:-1024}}"
UBATCH="${UBATCH:-${MARATHON_UBATCH:-256}}"
FLASH_ATTN="${FLASH_ATTN:-${MARATHON_FLASH_ATTN:-auto}}"
CACHE_TYPE_K="${CACHE_TYPE_K:-${MARATHON_CACHE_TYPE_K:-q8_0}}"
CACHE_TYPE_V="${CACHE_TYPE_V:-${MARATHON_CACHE_TYPE_V:-q8_0}}"
GPU_DEVICES="${GPU_DEVICES:-${MARATHON_GPU_DEVICES:-}}"

if [[ -z "$MODEL_PATH" ]]; then
  echo "error: custom model path is required." >&2
  echo "run: marathon backend start custom /path/to/model.gguf" >&2
  echo "or set MARATHON_MODEL_PATH=/path/to/model.gguf" >&2
  exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "error: model not found: $MODEL_PATH" >&2
  exit 1
fi

mkdir -p "$SLOT_SAVE_PATH"

if [[ -n "$GPU_DEVICES" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
fi

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
  --flash-attn "$FLASH_ATTN"
  --cache-type-k "$CACHE_TYPE_K"
  --cache-type-v "$CACHE_TYPE_V"
  --slot-save-path "$SLOT_SAVE_PATH"
  --parallel 1
)

if [[ -n "$TENSOR_SPLIT" ]]; then
  args+=(--tensor-split "$TENSOR_SPLIT")
fi

if [[ -n "${MARATHON_LLAMACPP_ARGS:-}" ]]; then
  read -r -a extra_args <<<"$MARATHON_LLAMACPP_ARGS"
  args+=("${extra_args[@]}")
fi

exec "$ROOT_DIR/scripts/ops/run_llama_server.sh" "${args[@]}" "$@"
