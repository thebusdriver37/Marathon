#!/usr/bin/env bash
# Bring an upstream openai/codex release into Marathon's vendored codex/ tree.
#
# codex/ is Marathon's own copy of upstream codex-rs, based on the tag in
# config/codex.ref. This script takes upstream's changes between that tag and
# the requested one and three-way merges them into codex/. Marathon's own edits
# are kept; where both sides changed the same lines, Git leaves conflict markers.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CODEX_DIR="$ROOT_DIR/codex"
CODEX_REF_FILE="$ROOT_DIR/config/codex.ref"
PRUNED_FILE="$ROOT_DIR/config/codex-pruned.txt"
UPSTREAM_URL="${MARATHON_CODEX_UPSTREAM_URL:-https://github.com/openai/codex.git}"
UPSTREAM_DIR="${MARATHON_CODEX_UPSTREAM_DIR:-$ROOT_DIR/.marathon/codex-upstream.git}"
DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
INSTALL_BIN="${MARATHON_CODEX_BIN:-$DATA_HOME/marathon/bin/codex}"

if [[ ! -f "$CODEX_REF_FILE" ]]; then
  echo "error: Codex base ref is missing: $CODEX_REF_FILE" >&2
  exit 1
fi
BASE_REF="$(tr -d '[:space:]' <"$CODEX_REF_FILE")"

if [[ -n "$(git -C "$ROOT_DIR" status --porcelain -- codex config/codex.ref)" ]]; then
  echo "error: codex/ has uncommitted changes; commit or stash them before syncing." >&2
  exit 1
fi

if [[ ! -d "$UPSTREAM_DIR" ]]; then
  echo "-> Caching upstream Codex history in $UPSTREAM_DIR (file contents load on demand)..."
  git clone --quiet --bare --filter=blob:none "$UPSTREAM_URL" "$UPSTREAM_DIR"
fi
if [[ "${MARATHON_CODEX_SKIP_FETCH:-0}" != "1" ]]; then
  echo "-> Fetching upstream Codex tags..."
  git -C "$UPSTREAM_DIR" fetch --quiet --tags origin
fi

TARGET_REF="${1:-${MARATHON_CODEX_UPSTREAM:-}}"
if [[ -z "$TARGET_REF" ]]; then
  TARGET_REF="$(git -C "$UPSTREAM_DIR" tag --list 'rust-v[0-9]*' --sort=-v:refname \
    | grep -Ev 'alpha|beta|rc' | head -n1 || true)"
fi
if [[ -z "$TARGET_REF" ]]; then
  echo "error: no stable upstream Codex tag found; pass one, e.g. marathon update-codex rust-v0.166.0" >&2
  exit 1
fi

BASE_SHA="$(git -C "$UPSTREAM_DIR" rev-parse --verify "$BASE_REF^{commit}")"
TARGET_SHA="$(git -C "$UPSTREAM_DIR" rev-parse --verify "$TARGET_REF^{commit}")"
if [[ "$BASE_SHA" == "$TARGET_SHA" ]]; then
  echo "codex/ is already based on $TARGET_REF (${TARGET_SHA:0:8})."
  exit 0
fi
# Upstream tags each release on its own branch, so compare versions, not ancestry.
if [[ "$(printf '%s\n%s\n' "$BASE_REF" "$TARGET_REF" | sort -V | tail -n1)" != "$TARGET_REF" ]]; then
  echo "error: $TARGET_REF is older than the current base $BASE_REF" >&2
  exit 1
fi

excludes=(--exclude=codex/codex-rs/Cargo.lock)
while IFS= read -r pattern; do
  [[ -z "$pattern" || "$pattern" == \#* ]] && continue
  excludes+=("--exclude=codex/$pattern")
done <"$PRUNED_FILE"

patch_file="$(mktemp "${TMPDIR:-/tmp}/marathon-codex-sync.XXXXXX")"
trap 'rm -f -- "$patch_file"' EXIT
git -C "$UPSTREAM_DIR" diff --binary --full-index "$BASE_SHA" "$TARGET_SHA" \
  -- codex-rs LICENSE NOTICE .cargo .gitignore >"$patch_file"

echo "-> Merging upstream $BASE_REF..$TARGET_REF into codex/..."
upstream_objects="$(git -C "$UPSTREAM_DIR" rev-parse --absolute-git-dir)/objects"
merged=1
# The upstream cache supplies the base file versions the three-way merge needs.
if ! GIT_ALTERNATE_OBJECT_DIRECTORIES="$upstream_objects" \
  git -C "$ROOT_DIR" apply --3way --directory=codex "${excludes[@]}" "$patch_file"; then
  merged=0
fi
printf '%s\n' "$TARGET_REF" >"$CODEX_REF_FILE"

conflicts="$(git -C "$ROOT_DIR" diff --name-only --diff-filter=U -- codex)"
if [[ "$merged" != "1" || -n "$conflicts" ]]; then
  echo
  echo "Upstream changes overlap Marathon's edits. Resolve the conflict markers in:"
  printf '  %s\n' ${conflicts:-"(see the git apply output above)"}
  echo "Then refresh the lockfile and build:"
  echo "  git -C '$UPSTREAM_DIR' show '$TARGET_SHA:codex-rs/Cargo.lock' >'$CODEX_DIR/codex-rs/Cargo.lock'"
  echo "  (cd '$CODEX_DIR/codex-rs' && cargo metadata --format-version 1 >/dev/null)"
  echo "  marathon build-codex"
  exit 1
fi

# Take upstream's pinned versions, then let Cargo drop what pruned crates used.
echo "-> Refreshing Cargo.lock from upstream $TARGET_REF..."
git -C "$UPSTREAM_DIR" show "$TARGET_SHA:codex-rs/Cargo.lock" >"$CODEX_DIR/codex-rs/Cargo.lock"
(cd "$CODEX_DIR/codex-rs" && cargo metadata --format-version 1 >/dev/null)

if [[ "${MARATHON_CODEX_SKIP_BUILD:-0}" != "1" ]]; then
  echo "-> Building Marathon Codex..."
  bash "$ROOT_DIR/scripts/build_codex.sh"
fi

echo
echo "codex/ now follows upstream $TARGET_REF (${TARGET_SHA:0:8}); installed binary: $INSTALL_BIN"
echo "Review and commit the sync:"
echo "  git -C '$ROOT_DIR' add codex config/codex.ref && git -C '$ROOT_DIR' commit -m 'Sync vendored Codex to $TARGET_REF'"
