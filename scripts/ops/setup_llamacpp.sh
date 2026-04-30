#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LLAMACPP_DIR="${LLAMACPP_DIR:-$ROOT_DIR/third_party/llama.cpp}"
LLAMACPP_REPO="${LLAMACPP_REPO:-https://github.com/ggml-org/llama.cpp.git}"
LLAMACPP_REF="${LLAMACPP_REF:-master}"
JOBS="${JOBS:-$(nproc)}"
CUDA_COMPILER="${CUDA_COMPILER:-/usr/local/cuda-12.4/bin/nvcc}"
CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda-12.4}"

mkdir -p "$(dirname "$LLAMACPP_DIR")"

if [[ ! -d "$LLAMACPP_DIR/.git" ]]; then
  git clone --depth 1 --branch "$LLAMACPP_REF" "$LLAMACPP_REPO" "$LLAMACPP_DIR"
else
  git -C "$LLAMACPP_DIR" fetch origin "$LLAMACPP_REF" --depth 1
  git -C "$LLAMACPP_DIR" checkout "$LLAMACPP_REF"
  git -C "$LLAMACPP_DIR" pull --ff-only origin "$LLAMACPP_REF"
fi

if [[ -f "$LLAMACPP_DIR/build/CMakeCache.txt" ]]; then
  rm -f "$LLAMACPP_DIR/build/CMakeCache.txt"
fi

cmake -S "$LLAMACPP_DIR" -B "$LLAMACPP_DIR/build" \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER="$CUDA_COMPILER" \
  -DCUDAToolkit_ROOT="$CUDA_ROOT" \
  -DCMAKE_BUILD_TYPE=Release

cmake --build "$LLAMACPP_DIR/build" --config Release -j "$JOBS"

echo
echo "Build complete. Key binaries:"
for bin in llama-server llama-cli llama-bench; do
  if [[ -x "$LLAMACPP_DIR/build/bin/$bin" ]]; then
    echo "  - $LLAMACPP_DIR/build/bin/$bin"
  else
    echo "  - missing: $LLAMACPP_DIR/build/bin/$bin"
  fi
done
