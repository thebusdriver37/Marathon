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
    if [[ "${MARATHON_PATCH_UPDATE:-0}" == "1" && -z "${MARATHON_PATCHED_CODEX_DIR:-}" ]]; then
      # Adopt the current immutable tree on first use, then retain its path
      # across patch edits. A new upstream revision gets a separate workspace.
      build_pointer="$ROOT_DIR/.marathon/codex-build-$(git -C "$CODEX_DIR" rev-parse HEAD).path"
      if [[ -f "$build_pointer" ]]; then
        TARGET_CODEX_DIR="$(cat "$build_pointer")"
      fi
    fi
  fi
  bash "$ROOT_DIR/scripts/lib/apply_patch_stack.sh" "$CODEX_DIR" "$TARGET_CODEX_DIR" "${patches[@]}"
  if [[ -n "${build_pointer:-}" ]]; then
    printf '%s\n' "$TARGET_CODEX_DIR" >"$build_pointer"
  fi
fi

echo "patched Codex tree: $TARGET_CODEX_DIR"
