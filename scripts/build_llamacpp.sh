#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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
PATCH_IN_PLACE="${MARATHON_LLAMACPP_PATCH_IN_PLACE:-0}"
BUILD_LLAMACPP_DIR="$LLAMACPP_DIR"
BUILD_DIR="${MARATHON_LLAMACPP_BUILD_DIR:-$ROOT_DIR/.marathon/llama.cpp-build}"
JOBS="${JOBS:-$(nproc)}"
CUDA_COMPILER="${CUDA_COMPILER:-/usr/local/cuda-12.4/bin/nvcc}"
CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda-12.4}"

if [[ "$PATCH_IN_PLACE" != "1" ]]; then
  BUILD_LLAMACPP_DIR="${MARATHON_PATCHED_LLAMACPP_DIR:-$ROOT_DIR/.marathon/llama.cpp-patched}"
fi

"$ROOT_DIR/scripts/apply_llamacpp_patches.sh"

mkdir -p "$BUILD_DIR"
cmake -S "$BUILD_LLAMACPP_DIR" -B "$BUILD_DIR" \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER="$CUDA_COMPILER" \
  -DCUDAToolkit_ROOT="$CUDA_ROOT" \
  -DCMAKE_BUILD_TYPE=Release

cmake --build "$BUILD_DIR" --config Release -j "$JOBS" --target llama-server llama-cli llama-bench

echo
echo "llama.cpp source: $BUILD_LLAMACPP_DIR"
echo "llama.cpp build: $BUILD_DIR"
for bin in llama-server llama-cli llama-bench; do
  if [[ -x "$BUILD_DIR/bin/$bin" ]]; then
    echo "  - $BUILD_DIR/bin/$bin"
  else
    echo "  - missing: $BUILD_DIR/bin/$bin"
  fi
done
