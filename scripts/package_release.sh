#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-}"
CODEX_BIN="${2:-}"

if [[ -z "$OUTPUT_DIR" || -z "$CODEX_BIN" ]]; then
  echo "usage: scripts/package_release.sh OUTPUT_DIR CODEX_BIN" >&2
  exit 2
fi
if [[ ! -x "$CODEX_BIN" ]]; then
  echo "error: built Codex frontend is missing: $CODEX_BIN" >&2
  exit 1
fi
for marker in features source prompt-hash; do
  if [[ ! -f "$CODEX_BIN.$marker" ]]; then
    echo "error: built Codex marker is missing: $CODEX_BIN.$marker" >&2
    exit 1
  fi
done
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo "error: release packaging currently supports Linux x86_64" >&2
  exit 1
fi

version="$(sed -n 's/^__version__ = "\([^"]*\)"$/\1/p' "$ROOT_DIR/marathon_app/__init__.py")"
if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "error: invalid Marathon version: $version" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
archive="marathon-$version-linux-x86_64.tar.gz"
temporary="$(mktemp -d -t marathon-release.XXXXXX)"
trap 'rm -r -- "$temporary"' EXIT
release="$temporary/marathon-$version"
mkdir -p "$release"

for path in bin marathon_app config docker docs patches scripts README.md; do
  cp -a "$ROOT_DIR/$path" "$release/"
done
find "$release" -type d -name __pycache__ -prune -exec rm -r -- {} +
find "$release" -type f -name '*.pyc' -delete

install -m755 "$CODEX_BIN" "$release/bin/codex"
for marker in features source prompt-hash; do
  install -m644 "$CODEX_BIN.$marker" "$release/bin/codex.$marker"
done
printf '{"schema":1,"version":"%s","target":"linux-x86_64"}\n' "$version" \
  >"$release/.marathon-release.json"

tar -C "$temporary" -czf "$OUTPUT_DIR/$archive" "marathon-$version"
(
  cd "$OUTPUT_DIR"
  sha256sum "$archive" >"$archive.sha256"
)
printf '%s\n' "$OUTPUT_DIR/$archive"
