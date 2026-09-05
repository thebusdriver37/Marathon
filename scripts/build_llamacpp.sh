#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/lib/llamacpp_paths.sh"
source "$ROOT_DIR/scripts/lib/build_jobs.sh"
GPU_BACKEND="${MARATHON_GPU_BACKEND:-auto}"

bash "$ROOT_DIR/scripts/apply_llamacpp_patches.sh"

cmake_args=(-DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF)
compiler_candidate="${CUDA_COMPILER:-$(command -v nvcc || true)}"
if [[ "$GPU_BACKEND" == cuda && ! -x "$compiler_candidate" ]]; then
  echo "error: CUDA was requested but nvcc is missing; install the CUDA toolkit" >&2
  exit 1
fi
if [[ "$GPU_BACKEND" == "cuda" || "$GPU_BACKEND" == "auto" && -x "$compiler_candidate" ]]; then
  CUDA_COMPILER="$compiler_candidate"
  CUDA_ROOT="${CUDA_ROOT:-$(cd "$(dirname "$CUDA_COMPILER")/.." && pwd)}"
  cmake_args+=(
    -DGGML_CUDA=ON
    -DCMAKE_CUDA_COMPILER="$CUDA_COMPILER"
    -DCUDAToolkit_ROOT="$CUDA_ROOT"
  )
  echo "building llama.cpp with CUDA from $CUDA_ROOT"
elif [[ "$GPU_BACKEND" == "auto" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  echo "error: NVIDIA tools were found, but nvcc is missing; install the CUDA toolkit and retry" >&2
  echo "CPU inference is available only by explicitly setting MARATHON_GPU_BACKEND=cpu" >&2
  exit 1
elif [[ "$GPU_BACKEND" == "cpu" || "$GPU_BACKEND" == "auto" ]]; then
  cmake_args+=(-DGGML_CUDA=OFF)
  echo "building the CPU backend" >&2
else
  echo "error: unsupported MARATHON_GPU_BACKEND: $GPU_BACKEND (choose auto, cuda, or cpu)" >&2
  exit 1
fi

if [[ "$VARIANT" == qwen38 ]]; then
  if [[ "$GPU_BACKEND" == cpu || ! -x "$compiler_candidate" ]]; then
    echo "error: the optional qwen38 runtime requires CUDA; use upstream for other backends" >&2
    exit 1
  fi
  cmake_args+=(
    -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_NATIVE=OFF
  )
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
