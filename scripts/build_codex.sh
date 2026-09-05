#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/lib/build_jobs.sh"
CODEX_DIR="${MARATHON_CODEX_DIR:-$ROOT_DIR/codex}"
PATCH_DIR="${MARATHON_PATCH_DIR:-$ROOT_DIR/patches/codex}"
PATCH_IN_PLACE="${MARATHON_PATCH_IN_PLACE:-0}"
BUILD_CODEX_DIR="$CODEX_DIR"
DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
INSTALL_BIN="${MARATHON_CODEX_BIN:-$DATA_HOME/marathon/bin/codex}"
BUILD_PROFILE="${MARATHON_CODEX_BUILD_PROFILE:-release}"
INSTALL_TMP=""
FEATURES_TMP=""
PROMPT_HASH_TMP=""
PROMPT_FILE="$CODEX_DIR/codex-rs/models-manager/prompt.md"

if [[ "$(uname -s)" == Linux ]] && ! command -v bwrap >/dev/null 2>&1; then
  echo "error: install bubblewrap for the Linux sandbox before building; see docs/SETUP.md" >&2
  exit 1
fi

if ! command -v cargo >/dev/null 2>&1; then
  echo "error: Rust cargo is required to build Marathon Codex" >&2
  echo "install Rust from https://rustup.rs, then rerun 'marathon build-codex'" >&2
  exit 1
fi

for tool in git cmake pkg-config; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "error: $tool is required for a source build; see docs/SETUP.md" >&2
    exit 1
  fi
done
if ! pkg-config --exists openssl; then
  echo "error: OpenSSL development libraries are missing (libssl-dev on Ubuntu); see docs/SETUP.md" >&2
  exit 1
fi

if [[ ! -f "$CODEX_DIR/codex-rs/Cargo.toml" ]]; then
  if [[ "$CODEX_DIR" != "$ROOT_DIR/codex" ]]; then
    echo "error: Codex source is missing at $CODEX_DIR" >&2
    exit 1
  fi
  echo "-> Initializing the pinned Codex source..."
  git -C "$ROOT_DIR" submodule update --init --depth 1 codex
fi

source "$ROOT_DIR/scripts/apply_codex_patches.sh"
BUILD_CODEX_DIR="$TARGET_CODEX_DIR"

export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-${MARATHON_CODEX_TARGET_DIR:-$ROOT_DIR/.marathon/codex-target}}"

cleanup() {
  [[ -z "$INSTALL_TMP" ]] || rm -f "$INSTALL_TMP"
  [[ -z "$FEATURES_TMP" ]] || rm -f "$FEATURES_TMP"
  [[ -z "$PROMPT_HASH_TMP" ]] || rm -f "$PROMPT_HASH_TMP"
}
trap cleanup EXIT

if [[ "${MARATHON_CODEX_RUN_TESTS:-0}" == "1" ]]; then
  echo "-> Testing Marathon Codex patches..."
  MARATHON_PATCHED_CODEX_DIR="$BUILD_CODEX_DIR" \
    "$ROOT_DIR/scripts/test_codex_patches.sh"
fi

(
  cd "$BUILD_CODEX_DIR/codex-rs"
  cargo build --locked --profile "$BUILD_PROFILE" -p codex-cli
)

candidate="$CARGO_TARGET_DIR/$BUILD_PROFILE/codex"
"$candidate" --version
"$candidate" --help >/dev/null
"$candidate" exec --help >/dev/null

install_dir="$(dirname "$INSTALL_BIN")"
mkdir -p "$install_dir"
INSTALL_TMP="$(mktemp "$install_dir/.codex.new.XXXXXX")"
install -m755 "$candidate" "$INSTALL_TMP"
if command -v strip >/dev/null 2>&1; then
  strip "$INSTALL_TMP"
fi

if [[ -x "$INSTALL_BIN" ]]; then
  backup_tmp="$(mktemp "$install_dir/.codex.previous.XXXXXX")"
  install -m755 "$INSTALL_BIN" "$backup_tmp"
  mv -f "$backup_tmp" "$INSTALL_BIN.previous"
  if [[ -f "$INSTALL_BIN.source" ]]; then
    cp -f "$INSTALL_BIN.source" "$INSTALL_BIN.previous.source"
  fi
  if [[ -f "$INSTALL_BIN.features" ]]; then
    cp -f "$INSTALL_BIN.features" "$INSTALL_BIN.previous.features"
  fi
  if [[ -f "$INSTALL_BIN.prompt-hash" ]]; then
    cp -f "$INSTALL_BIN.prompt-hash" "$INSTALL_BIN.previous.prompt-hash"
  fi
fi

mv -f "$INSTALL_TMP" "$INSTALL_BIN"
INSTALL_TMP=""

shopt -s nullglob
patches=("$PATCH_DIR"/*.patch)
shopt -u nullglob
patch_manifest="$({
  for patch in "${patches[@]}"; do
    printf '%s %s\n' "$(basename "$patch")" "$(git hash-object "$patch")"
  done
} | git hash-object --stdin)"
source_tmp="$(mktemp "$install_dir/.codex.source.XXXXXX")"
printf '%s:%s\n' "$(git -C "$CODEX_DIR" rev-parse HEAD)" "$patch_manifest" >"$source_tmp"
mv -f "$source_tmp" "$INSTALL_BIN.source"

FEATURES_TMP="$(mktemp "$install_dir/.codex.features.XXXXXX")"
if grep -Rq 'tokens-per-second' "$BUILD_CODEX_DIR/codex-rs/tui/src"; then
  printf 'tokens-per-second\n' >"$FEATURES_TMP"
fi
if grep -q 'MARATHON_LOCAL_ONLY' "$BUILD_CODEX_DIR/codex-rs/http-client/src/local_runtime.rs"; then
  printf 'local-runtime-security\n' >>"$FEATURES_TMP"
fi
mv -f "$FEATURES_TMP" "$INSTALL_BIN.features"
FEATURES_TMP=""

PROMPT_HASH_TMP="$(mktemp "$install_dir/.codex.prompt-hash.XXXXXX")"
if [[ -f "$PROMPT_FILE" ]]; then
  git hash-object "$PROMPT_FILE" >"$PROMPT_HASH_TMP"
else
  printf 'missing\n' >"$PROMPT_HASH_TMP"
fi
mv -f "$PROMPT_HASH_TMP" "$INSTALL_BIN.prompt-hash"
PROMPT_HASH_TMP=""

echo
echo "Codex source: $BUILD_CODEX_DIR"
echo "Codex binary: $INSTALL_BIN"
echo "Codex build: $(cat "$INSTALL_BIN.source")"
