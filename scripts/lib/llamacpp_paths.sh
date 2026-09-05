#!/usr/bin/env bash
# Shared by setup, patching, and building. Never remove an existing source tree.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AI_ROOT="${MARATHON_AI_ROOT:-$HOME/AI}"
VARIANT="${MARATHON_LLAMACPP_VARIANT:-upstream}"
case "$VARIANT" in
  upstream)
    default_source="$AI_ROOT/backends/llama.cpp-current"
    ref_file="$ROOT_DIR/config/llamacpp.ref"
    default_patches="$ROOT_DIR/patches/llama.cpp"
    ;;
  qwen38)
    default_source="$AI_ROOT/backends/llama.cpp-qwen38"
    ref_file="$ROOT_DIR/config/llamacpp-qwen38.ref"
    default_patches="$ROOT_DIR/patches/llama.cpp/qwen38"
    ;;
  *)
    echo "error: unknown runtime '$VARIANT' (choose upstream or qwen38)" >&2
    exit 1
    ;;
esac
LLAMACPP_DIR="${MARATHON_LLAMACPP_DIR:-${LLAMACPP_DIR:-$default_source}}"
PATCH_DIR="${MARATHON_LLAMACPP_PATCH_DIR:-$default_patches}"
PATCH_IN_PLACE="${MARATHON_LLAMACPP_PATCH_IN_PLACE:-0}"
BUILD_DIR="${MARATHON_LLAMACPP_BUILD_DIR:-$LLAMACPP_DIR/build}"
shopt -s nullglob
patches=("$PATCH_DIR"/*.patch)
shopt -u nullglob
BUILD_LLAMACPP_DIR="$LLAMACPP_DIR"
if [[ "$PATCH_IN_PLACE" != 1 && ${#patches[@]} -gt 0 ]] && git -C "$LLAMACPP_DIR" rev-parse HEAD >/dev/null 2>&1; then
  patch_identity="$({
    git -C "$LLAMACPP_DIR" rev-parse HEAD
    for patch in "${patches[@]}"; do git hash-object "$patch"; done
  } | git hash-object --stdin)"
  BUILD_LLAMACPP_DIR="${MARATHON_PATCHED_LLAMACPP_DIR:-$ROOT_DIR/.marathon/llama.cpp-$VARIANT-$patch_identity}"
fi
