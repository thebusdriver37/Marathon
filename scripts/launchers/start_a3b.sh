#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export MARATHON_DEFAULT_MODEL="${MARATHON_DEFAULT_MODEL:-qwen3.6-35b-a3b}"

exec "$ROOT_DIR/scripts/launchers/start_codex.sh" "$@"
