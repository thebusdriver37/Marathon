#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODEX_DIR="${MARATHON_CODEX_DIR:-$ROOT_DIR/codex}"
PATCH_DIR="${MARATHON_PATCH_DIR:-$ROOT_DIR/patches/codex}"

if ! git -C "$CODEX_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "error: Codex submodule is missing at $CODEX_DIR" >&2
  echo "run: git submodule update --init --recursive" >&2
  exit 1
fi

shopt -s nullglob
patches=("$PATCH_DIR"/*.patch)
if [[ "${#patches[@]}" -eq 0 ]]; then
  echo "no Codex patches found in $PATCH_DIR"
  exit 0
fi

for patch in "${patches[@]}"; do
  name="$(basename "$patch")"
  if git -C "$CODEX_DIR" apply --check "$patch" >/dev/null 2>&1; then
    git -C "$CODEX_DIR" apply "$patch"
    echo "applied: $name"
  elif git -C "$CODEX_DIR" apply --reverse --check "$patch" >/dev/null 2>&1; then
    echo "already applied: $name"
  else
    echo "error: patch does not apply cleanly: $name" >&2
    echo "inspect with: git -C '$CODEX_DIR' apply --check '$patch'" >&2
    exit 1
  fi
done
