#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib/llamacpp_paths.sh"

if [[ ${#patches[@]} -eq 0 ]]; then
  echo "no llama.cpp patches selected ($VARIANT)"
  exit 0
fi
base_ref="$(git -C "$LLAMACPP_DIR" rev-parse HEAD)"
if [[ "$VARIANT" == qwen38 && "$base_ref" != "$(tr -d '[:space:]' < "$ref_file")" ]]; then
  echo "error: qwen38 patches require the exact base in $ref_file" >&2
  exit 1
fi

bash "$ROOT_DIR/scripts/lib/apply_patch_stack.sh" "$LLAMACPP_DIR" "$BUILD_LLAMACPP_DIR" "${patches[@]}"
