#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AI_ROOT="${MARATHON_AI_ROOT:-$HOME/AI}"
LLAMACPP_DIR="${MARATHON_LLAMACPP_DIR:-${LLAMACPP_DIR:-$AI_ROOT/backends/llama.cpp-current}}"
LLAMACPP_REPO="${LLAMACPP_REPO:-https://github.com/ggml-org/llama.cpp.git}"
PINNED_REF="$(tr -d '[:space:]' < "$ROOT_DIR/config/llamacpp.ref")"
LLAMACPP_REF="${LLAMACPP_REF:-$PINNED_REF}"

mkdir -p "$(dirname "$LLAMACPP_DIR")"

if [[ ! -d "$LLAMACPP_DIR/.git" ]]; then
  git clone --filter=blob:none --no-checkout "$LLAMACPP_REPO" "$LLAMACPP_DIR"
else
  if ! git -C "$LLAMACPP_DIR" diff --quiet || ! git -C "$LLAMACPP_DIR" diff --cached --quiet; then
    echo "error: llama.cpp source tree has uncommitted changes; commit or discard them first." >&2
    exit 1
  fi
  if [[ -n "$(git -C "$LLAMACPP_DIR" ls-files --others --exclude-standard)" ]]; then
    echo "error: llama.cpp source tree has untracked files; remove or commit them first." >&2
    exit 1
  fi
fi

git -C "$LLAMACPP_DIR" fetch origin "$LLAMACPP_REF" --depth 1
git -C "$LLAMACPP_DIR" checkout --detach FETCH_HEAD

export MARATHON_LLAMACPP_DIR="$LLAMACPP_DIR"
bash "$ROOT_DIR/scripts/build_llamacpp.sh"
