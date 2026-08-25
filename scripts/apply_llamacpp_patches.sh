#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AI_ROOT="${MARATHON_AI_ROOT:-$HOME/AI}"

default_llamacpp_dir() {
  printf '%s\n' "$AI_ROOT/backends/llama.cpp-current"
}

LLAMACPP_DIR="${MARATHON_LLAMACPP_DIR:-${LLAMACPP_DIR:-$(default_llamacpp_dir)}}"
PATCH_DIR="${MARATHON_LLAMACPP_PATCH_DIR:-$ROOT_DIR/patches/llama.cpp}"
PATCH_IN_PLACE="${MARATHON_LLAMACPP_PATCH_IN_PLACE:-0}"
TARGET_LLAMACPP_DIR="$LLAMACPP_DIR"

shopt -s nullglob
patches=("$PATCH_DIR"/*.patch)
shopt -u nullglob
if [[ "${#patches[@]}" -eq 0 ]]; then
  echo "no llama.cpp patches found"
  exit 0
fi

if ! git -C "$LLAMACPP_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "error: llama.cpp source repo is missing at $LLAMACPP_DIR" >&2
  echo "run: $ROOT_DIR/bin/marathon setup-llama" >&2
  exit 1
fi

if [[ "$PATCH_IN_PLACE" != "1" ]]; then
  TARGET_LLAMACPP_DIR="${MARATHON_PATCHED_LLAMACPP_DIR:-$ROOT_DIR/.marathon/llama.cpp-patched}"
  base_ref="$(git -C "$LLAMACPP_DIR" rev-parse HEAD)"

  if [[ -e "$TARGET_LLAMACPP_DIR" ]]; then
    if ! git -C "$TARGET_LLAMACPP_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
      echo "error: patched llama.cpp target exists but is not a git worktree: $TARGET_LLAMACPP_DIR" >&2
      exit 1
    fi
    git -C "$LLAMACPP_DIR" worktree remove --force "$TARGET_LLAMACPP_DIR" >/dev/null
  fi
  mkdir -p "$(dirname "$TARGET_LLAMACPP_DIR")"
  git -C "$LLAMACPP_DIR" worktree add --detach "$TARGET_LLAMACPP_DIR" "$base_ref" >/dev/null
fi

for patch in "${patches[@]}"; do
  name="$(basename "$patch")"
  if git -C "$TARGET_LLAMACPP_DIR" apply --check "$patch" >/dev/null 2>&1; then
    git -C "$TARGET_LLAMACPP_DIR" apply "$patch"
    echo "applied: $name"
  elif git -C "$TARGET_LLAMACPP_DIR" apply --reverse --check "$patch" >/dev/null 2>&1; then
    echo "already applied: $name"
  else
    echo "error: patch does not apply cleanly: $name" >&2
    echo "inspect with: git -C '$TARGET_LLAMACPP_DIR' apply --check '$patch'" >&2
    exit 1
  fi
done

echo "patched llama.cpp tree: $TARGET_LLAMACPP_DIR"
