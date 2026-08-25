#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODEX_DIR="${MARATHON_CODEX_DIR:-$ROOT_DIR/codex}"
PATCH_DIR="${MARATHON_PATCH_DIR:-$ROOT_DIR/patches/codex}"
PATCH_IN_PLACE="${MARATHON_PATCH_IN_PLACE:-0}"
TARGET_CODEX_DIR="$CODEX_DIR"

if ! git -C "$CODEX_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "error: Codex submodule is missing at $CODEX_DIR" >&2
  echo "run: git submodule update --init --recursive" >&2
  exit 1
fi

if [[ "$PATCH_IN_PLACE" != "1" ]]; then
  TARGET_CODEX_DIR="${MARATHON_PATCHED_CODEX_DIR:-$ROOT_DIR/.marathon/codex-patched}"
  base_ref="$(git -C "$CODEX_DIR" rev-parse HEAD)"

  if [[ -e "$TARGET_CODEX_DIR" ]]; then
    if ! git -C "$TARGET_CODEX_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
      echo "error: patched Codex target exists but is not a git worktree: $TARGET_CODEX_DIR" >&2
      exit 1
    fi
    git -C "$CODEX_DIR" worktree remove --force "$TARGET_CODEX_DIR" >/dev/null
  fi
  mkdir -p "$(dirname "$TARGET_CODEX_DIR")"
  git -C "$CODEX_DIR" worktree add --detach "$TARGET_CODEX_DIR" "$base_ref" >/dev/null
fi

shopt -s nullglob
patches=("$PATCH_DIR"/*.patch)
if [[ "${#patches[@]}" -eq 0 ]]; then
  echo "no Codex patches found in $PATCH_DIR"
  exit 0
fi

for patch in "${patches[@]}"; do
  name="$(basename "$patch")"
  if git -C "$TARGET_CODEX_DIR" apply --check "$patch" >/dev/null 2>&1; then
    git -C "$TARGET_CODEX_DIR" apply "$patch"
    echo "applied: $name"
  elif git -C "$TARGET_CODEX_DIR" apply --reverse --check "$patch" >/dev/null 2>&1; then
    echo "already applied: $name"
  else
    echo "error: patch does not apply cleanly: $name" >&2
    echo "inspect with: git -C '$TARGET_CODEX_DIR' apply --check '$patch'" >&2
    exit 1
  fi
done

echo "patched Codex tree: $TARGET_CODEX_DIR"
