#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CODEX_DIR="${MARATHON_CODEX_DIR:-$ROOT_DIR/codex}"
PATCHES_DIR="${MARATHON_PATCH_DIR:-$ROOT_DIR/patches/codex}"
UPSTREAM_BRANCH="${MARATHON_CODEX_UPSTREAM:-origin/main}"

if ! git -C "$CODEX_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "error: Codex submodule is missing at $CODEX_DIR" >&2
  echo "run: git submodule update --init --recursive" >&2
  exit 1
fi

if ! git -C "$CODEX_DIR" diff --quiet \
  || ! git -C "$CODEX_DIR" diff --cached --quiet; then
  echo "error: Codex submodule has uncommitted changes; commit or discard them first." >&2
  exit 1
fi
if [[ -n "$(git -C "$CODEX_DIR" ls-files --others --exclude-standard)" ]]; then
  echo "error: Codex submodule has untracked files; remove or commit them first." >&2
  exit 1
fi

echo "-> Fetching upstream Codex..."
git -C "$CODEX_DIR" fetch origin

LOCAL_SHA="$(git -C "$CODEX_DIR" rev-parse HEAD)"
UPSTREAM_SHA="$(git -C "$CODEX_DIR" rev-parse "$UPSTREAM_BRANCH")"

if [[ "$LOCAL_SHA" == "$UPSTREAM_SHA" ]]; then
  echo "Codex already at ${UPSTREAM_SHA:0:8}; nothing to do."
  exit 0
fi

echo "-> Codex ${LOCAL_SHA:0:8} -> ${UPSTREAM_SHA:0:8}"

shopt -s nullglob
patches=("$PATCHES_DIR"/*.patch)
shopt -u nullglob

# Preflight: verify every patch applies cleanly on the new upstream
# before we mutate the submodule. Uses a throwaway worktree at the
# target SHA so the live tree is untouched if anything fails.
if (( ${#patches[@]} > 0 )); then
  PREFLIGHT_PARENT="$(mktemp -d -t codex-preflight.XXXXXX)"
  PREFLIGHT_DIR="$PREFLIGHT_PARENT/wt"
  cleanup_preflight() {
    if [[ -d "$PREFLIGHT_DIR" ]]; then
      git -C "$CODEX_DIR" worktree remove --force "$PREFLIGHT_DIR" >/dev/null 2>&1 || true
    fi
    rm -rf "$PREFLIGHT_PARENT"
  }
  trap cleanup_preflight EXIT

  git -C "$CODEX_DIR" worktree add --detach "$PREFLIGHT_DIR" "$UPSTREAM_SHA" >/dev/null
  for patch in "${patches[@]}"; do
    name="$(basename "$patch")"
    if ! git -C "$PREFLIGHT_DIR" apply --check "$patch" >/dev/null 2>&1; then
      echo "error: $name does not apply cleanly on ${UPSTREAM_SHA:0:8}" >&2
      echo "  reproduce: git -C '$PREFLIGHT_DIR' apply --check '$patch'" >&2
      echo "  resolve by rebasing the patch against current upstream." >&2
      exit 1
    fi
  done

  cleanup_preflight
  trap - EXIT
fi

echo "-> Resetting Codex submodule to ${UPSTREAM_SHA:0:8}..."
git -C "$CODEX_DIR" reset --hard "$UPSTREAM_SHA"

echo "-> Building patched Codex..."
bash "$ROOT_DIR/scripts/build_codex.sh"

echo
echo "Codex synced to ${UPSTREAM_SHA:0:8} with ${#patches[@]} patch(es) applied to the build worktree."
echo "Commit the submodule bump in the parent repo:"
echo "  git -C '$ROOT_DIR' add codex && git -C '$ROOT_DIR' commit -m 'codex: sync to ${UPSTREAM_SHA:0:8}'"
