#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CODEX_DIR="${MARATHON_CODEX_DIR:-$ROOT_DIR/codex}"
PATCHES_DIR="${MARATHON_PATCH_DIR:-$ROOT_DIR/patches/codex}"
CODEX_REF_FILE="$ROOT_DIR/config/codex.ref"
if [[ -n "${MARATHON_CODEX_UPSTREAM:-}" ]]; then
  UPSTREAM_REF="$MARATHON_CODEX_UPSTREAM"
elif [[ -f "$CODEX_REF_FILE" ]]; then
  UPSTREAM_REF="$(tr -d '[:space:]' <"$CODEX_REF_FILE")"
else
  echo "error: Codex ref is missing: $CODEX_REF_FILE" >&2
  exit 1
fi
DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
INSTALL_BIN="${MARATHON_CODEX_BIN:-$DATA_HOME/marathon/bin/codex}"

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

echo "-> Fetching upstream Codex tags..."
git -C "$CODEX_DIR" fetch origin --tags

LOCAL_SHA="$(git -C "$CODEX_DIR" rev-parse HEAD)"
UPSTREAM_SHA="$(git -C "$CODEX_DIR" rev-parse "$UPSTREAM_REF^{commit}")"

shopt -s nullglob
patches=("$PATCHES_DIR"/*.patch)
shopt -u nullglob
patch_manifest="$({
  for patch in "${patches[@]}"; do
    printf '%s %s\n' "$(basename "$patch")" "$(git hash-object "$patch")"
  done
} | git hash-object --stdin)"
expected_build="$UPSTREAM_SHA:$patch_manifest"

if [[ "$LOCAL_SHA" == "$UPSTREAM_SHA" ]]; then
  installed_build="$(cat "$INSTALL_BIN.source" 2>/dev/null || true)"
  if [[ -x "$INSTALL_BIN" && "$installed_build" == "$expected_build" \
    && "${MARATHON_FORCE_CODEX_REBUILD:-0}" != "1" ]]; then
    echo "Codex source and Marathon patches are current at ${UPSTREAM_SHA:0:8}."
    exit 0
  fi
  echo "Codex source is at ${UPSTREAM_SHA:0:8}; rebuilding changed Marathon patches."
else
  echo "-> Codex ${LOCAL_SHA:0:8} -> ${UPSTREAM_SHA:0:8}"
fi

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
    rm -r -- "$PREFLIGHT_PARENT"
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
    git -C "$PREFLIGHT_DIR" apply "$patch"
  done

  cleanup_preflight
  trap - EXIT
fi

if [[ "$LOCAL_SHA" != "$UPSTREAM_SHA" ]]; then
  echo "-> Fast-forwarding Codex submodule to ${UPSTREAM_SHA:0:8}..."
  git -C "$CODEX_DIR" merge --ff-only "$UPSTREAM_SHA"
fi

echo "-> Building patched Codex..."
bash "$ROOT_DIR/scripts/build_codex.sh"

echo
echo "Codex $UPSTREAM_REF synced and tested at ${UPSTREAM_SHA:0:8} with ${#patches[@]} Marathon patch(es)."
echo "Commit the submodule bump in the parent repo:"
echo "  git -C '$ROOT_DIR' add codex && git -C '$ROOT_DIR' commit -m 'codex: sync to ${UPSTREAM_SHA:0:8}'"
