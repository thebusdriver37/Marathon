#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LLAMACPP_DIR="${LLAMACPP_DIR:-$ROOT_DIR/third_party/llama.cpp}"
LLAMACPP_REPO="${LLAMACPP_REPO:-https://github.com/ggml-org/llama.cpp.git}"
LLAMACPP_REF="${LLAMACPP_REF:-master}"

mkdir -p "$(dirname "$LLAMACPP_DIR")"

if [[ ! -d "$LLAMACPP_DIR/.git" ]]; then
  git clone --depth 1 --branch "$LLAMACPP_REF" "$LLAMACPP_REPO" "$LLAMACPP_DIR"
else
  if ! git -C "$LLAMACPP_DIR" diff --quiet || ! git -C "$LLAMACPP_DIR" diff --cached --quiet; then
    echo "error: llama.cpp source tree has uncommitted changes; commit or discard them first." >&2
    exit 1
  fi
  if [[ -n "$(git -C "$LLAMACPP_DIR" ls-files --others --exclude-standard)" ]]; then
    echo "error: llama.cpp source tree has untracked files; remove or commit them first." >&2
    exit 1
  fi
  git -C "$LLAMACPP_DIR" fetch origin "$LLAMACPP_REF" --depth 1
  git -C "$LLAMACPP_DIR" checkout "$LLAMACPP_REF"
  git -C "$LLAMACPP_DIR" pull --ff-only origin "$LLAMACPP_REF"
fi

export MARATHON_LLAMACPP_DIR="$LLAMACPP_DIR"
bash "$ROOT_DIR/scripts/build_llamacpp.sh"
