#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

default_llamacpp_dir() {
  if [[ -d "$ROOT_DIR/third_party/llama.cpp/.git" ]]; then
    printf '%s\n' "$ROOT_DIR/third_party/llama.cpp"
  elif [[ -d "$ROOT_DIR/../vLLM_inference/third_party/llama.cpp/.git" ]]; then
    printf '%s\n' "$ROOT_DIR/../vLLM_inference/third_party/llama.cpp"
  else
    printf '%s\n' "$ROOT_DIR/third_party/llama.cpp"
  fi
}

LLAMACPP_DIR="${MARATHON_LLAMACPP_DIR:-${LLAMACPP_DIR:-$(default_llamacpp_dir)}}"
PATCHED_BUILD_DIR="${MARATHON_LLAMACPP_BUILD_DIR:-$ROOT_DIR/.marathon/llama.cpp-build}"
PATCHED_BIN="${MARATHON_LLAMACPP_BIN_PATH:-$PATCHED_BUILD_DIR/bin/llama-server}"
BIN="${LLAMACPP_BIN:-${MARATHON_LLAMACPP_BIN:-$PATCHED_BIN}}"
if [[ ! -x "$BIN" && -x "$LLAMACPP_DIR/build/bin/llama-server" ]]; then
  BIN="$LLAMACPP_DIR/build/bin/llama-server"
fi
if [[ ! -x "$BIN" && -x "$ROOT_DIR/../vLLM_inference/third_party/llama.cpp/build/bin/llama-server" ]]; then
  LLAMACPP_DIR="$ROOT_DIR/../vLLM_inference/third_party/llama.cpp"
  BIN="$LLAMACPP_DIR/build/bin/llama-server"
fi

MODEL_PATH="${MODEL_PATH:-}"
MODEL_ALIAS="${MODEL_ALIAS:-local-qwen}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
CTX_SIZE="${CTX_SIZE:-32768}"
NGL="${NGL:-999}"
SPLIT_MODE="${SPLIT_MODE:-layer}"
TENSOR_SPLIT="${TENSOR_SPLIT:-}"
THREADS="${THREADS:-$(nproc)}"
BATCH="${BATCH:-2048}"
UBATCH="${UBATCH:-512}"
FLASH_ATTN="${FLASH_ATTN:-auto}"
NO_MMAP="${NO_MMAP:-0}"
DRY_RUN=0

declare -a OT_OVERRIDES=()
declare -a PASSTHROUGH_ARGS=()

usage() {
  cat <<EOF
Usage: $(basename "$0") --model /path/to/model.gguf [options]

Options:
  --model PATH                 GGUF model path (required if MODEL_PATH not set)
  --alias NAME                 Model alias exposed by OpenAI API (default: $MODEL_ALIAS)
  --host HOST                  Bind host (default: $HOST)
  --port PORT                  Port (default: $PORT)
  --ctx N                      Context size (default: $CTX_SIZE)
  --ngl N                      Number of layers to offload to GPU (default: $NGL)
  --split-mode MODE            none|layer|row (default: $SPLIT_MODE)
  --tensor-split RATIOS        Comma-separated ratios, ex: 3,2
  --threads N                  CPU threads (default: $THREADS)
  --batch N                    Logical batch size (default: $BATCH)
  --ubatch N                   Physical micro-batch size (default: $UBATCH)
  --ot REGEX=DEVICE            Repeatable tensor override, ex: blk\\.[2][0-3]\\.ffn_.*exps=CPU
  --flash-attn MODE            auto|on|off (default: $FLASH_ATTN)
  --no-flash-attn              Shortcut for --flash-attn off
  --no-mmap                    Disable mmap
  --dry-run                    Print command only
  --help                       Show this help

All unknown flags are passed through to llama-server.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL_PATH="$2"
      shift 2
      ;;
    --alias)
      MODEL_ALIAS="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --ctx)
      CTX_SIZE="$2"
      shift 2
      ;;
    --ngl)
      NGL="$2"
      shift 2
      ;;
    --split-mode)
      SPLIT_MODE="$2"
      shift 2
      ;;
    --tensor-split)
      TENSOR_SPLIT="$2"
      shift 2
      ;;
    --threads)
      THREADS="$2"
      shift 2
      ;;
    --batch)
      BATCH="$2"
      shift 2
      ;;
    --ubatch)
      UBATCH="$2"
      shift 2
      ;;
    --ot)
      OT_OVERRIDES+=("$2")
      shift 2
      ;;
    --no-flash-attn)
      FLASH_ATTN="off"
      shift
      ;;
    --flash-attn)
      FLASH_ATTN="$2"
      shift 2
      ;;
    --no-mmap)
      NO_MMAP=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$MODEL_PATH" ]]; then
  echo "error: model path is required (--model PATH or MODEL_PATH env var)." >&2
  exit 1
fi

cmd=(
  "$BIN"
  -m "$MODEL_PATH"
  --alias "$MODEL_ALIAS"
  --host "$HOST"
  --port "$PORT"
  -c "$CTX_SIZE"
  -ngl "$NGL"
  -sm "$SPLIT_MODE"
  -t "$THREADS"
  -b "$BATCH"
  -ub "$UBATCH"
  --metrics
)

if [[ -n "$TENSOR_SPLIT" ]]; then
  cmd+=(-ts "$TENSOR_SPLIT")
fi

if [[ -n "$FLASH_ATTN" ]]; then
  cmd+=(--flash-attn "$FLASH_ATTN")
fi

if [[ "$NO_MMAP" == "1" ]]; then
  cmd+=(--no-mmap)
fi

for ot in "${OT_OVERRIDES[@]}"; do
  cmd+=(-ot "$ot")
done

if [[ "${#PASSTHROUGH_ARGS[@]}" -gt 0 ]]; then
  cmd+=("${PASSTHROUGH_ARGS[@]}")
fi

printf 'Running: '
printf '%q ' "${cmd[@]}"
printf '\n'

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

if [[ ! -x "$BIN" ]]; then
  echo "error: llama-server binary not found at: $BIN" >&2
  echo "run ./bin/marathon setup-llama (or ./bin/marathon build-llama), or set LLAMACPP_BIN." >&2
  exit 1
fi

exec "${cmd[@]}"
