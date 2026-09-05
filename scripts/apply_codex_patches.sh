#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODEX_DIR="${MARATHON_CODEX_DIR:-$ROOT_DIR/codex}"
PATCH_DIR="${MARATHON_PATCH_DIR:-$ROOT_DIR/patches/codex}"
PATCH_IN_PLACE="${MARATHON_PATCH_IN_PLACE:-0}"
TARGET_CODEX_DIR="$CODEX_DIR"

if [[ ! -e "$CODEX_DIR/.git" ]]; then
  echo "error: Codex submodule is missing at $CODEX_DIR" >&2
  echo "run: git submodule update --init --recursive" >&2
  exit 1
fi

shopt -s nullglob
patches=("$PATCH_DIR"/*.patch)
shopt -u nullglob
if [[ "${#patches[@]}" -eq 0 ]]; then
  echo "no Codex patches found in $PATCH_DIR"
else
  if [[ "$PATCH_IN_PLACE" != "1" ]]; then
    patch_identity="$({
      git -C "$CODEX_DIR" rev-parse HEAD
      for patch in "${patches[@]}"; do git hash-object "$patch"; done
    } | git hash-object --stdin)"
    TARGET_CODEX_DIR="${MARATHON_PATCHED_CODEX_DIR:-$ROOT_DIR/.marathon/codex-patched-$patch_identity}"
  fi
  bash "$ROOT_DIR/scripts/lib/apply_patch_stack.sh" "$CODEX_DIR" "$TARGET_CODEX_DIR" "${patches[@]}"
fi

echo "patched Codex tree: $TARGET_CODEX_DIR"
