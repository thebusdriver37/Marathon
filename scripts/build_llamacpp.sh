#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AI_ROOT="${MARATHON_AI_ROOT:-$HOME/AI}"

default_llamacpp_dir() {
  printf '%s\n' "$AI_ROOT/backends/llama.cpp-current"
}

LLAMACPP_DIR="${MARATHON_LLAMACPP_DIR:-${LLAMACPP_DIR:-$(default_llamacpp_dir)}}"
PATCH_IN_PLACE="${MARATHON_LLAMACPP_PATCH_IN_PLACE:-0}"
BUILD_LLAMACPP_DIR="$LLAMACPP_DIR"
BUILD_DIR="${MARATHON_LLAMACPP_BUILD_DIR:-$LLAMACPP_DIR/build}"
JOBS="${JOBS:-$(nproc)}"
GPU_BACKEND="${MARATHON_GPU_BACKEND:-auto}"

if [[ "$PATCH_IN_PLACE" != "1" ]] && compgen -G "$ROOT_DIR/patches/llama.cpp/*.patch" >/dev/null; then
  BUILD_LLAMACPP_DIR="${MARATHON_PATCHED_LLAMACPP_DIR:-$ROOT_DIR/.marathon/llama.cpp-patched}"
fi

"$ROOT_DIR/scripts/apply_llamacpp_patches.sh"

cmake_args=(-DCMAKE_BUILD_TYPE=Release)
if [[ "$GPU_BACKEND" == "cuda" || "$GPU_BACKEND" == "auto" && -n "$(command -v nvcc || true)" ]]; then
  CUDA_COMPILER="${CUDA_COMPILER:-$(command -v nvcc)}"
  CUDA_ROOT="${CUDA_ROOT:-$(cd "$(dirname "$CUDA_COMPILER")/.." && pwd)}"
  cmake_args+=(
    -DGGML_CUDA=ON
    -DCMAKE_CUDA_COMPILER="$CUDA_COMPILER"
    -DCUDAToolkit_ROOT="$CUDA_ROOT"
  )
  echo "building llama.cpp with CUDA from $CUDA_ROOT"
elif [[ "$GPU_BACKEND" == "cpu" || "$GPU_BACKEND" == "auto" ]]; then
  cmake_args+=(-DGGML_CUDA=OFF)
  echo "warning: nvcc was not found; building the CPU backend" >&2
else
  echo "error: unsupported MARATHON_GPU_BACKEND: $GPU_BACKEND (choose auto, cuda, or cpu)" >&2
  exit 1
fi

mkdir -p "$BUILD_DIR"
cmake -S "$BUILD_LLAMACPP_DIR" -B "$BUILD_DIR" "${cmake_args[@]}"

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
