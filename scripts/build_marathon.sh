#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AI_ROOT="${MARATHON_AI_ROOT:-$HOME/AI}"

default_llamacpp_dir() {
  printf '%s\n' "$AI_ROOT/backends/llama.cpp-current"
}

bash "$ROOT_DIR/scripts/build_codex.sh"

if [[ "${MARATHON_SKIP_LLAMA_BUILD:-0}" == "1" ]]; then
  exit 0
fi

LLAMACPP_DIR="${MARATHON_LLAMACPP_DIR:-${LLAMACPP_DIR:-$(default_llamacpp_dir)}}"
if git -C "$LLAMACPP_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo
  echo "-> Building patched llama.cpp..."
  bash "$ROOT_DIR/scripts/build_llamacpp.sh"
else
  echo
  echo "-> Skipping llama.cpp build: source repo not found at $LLAMACPP_DIR"
  echo "   Run '$ROOT_DIR/bin/marathon setup-llama' to clone/build it later."
fi
