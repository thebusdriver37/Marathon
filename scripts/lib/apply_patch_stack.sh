#!/usr/bin/env bash
# Apply an overlapping stack once, preserving both upstream and user edits.
set -euo pipefail
source_dir="$1"
target_dir="$2"
shift 2
base_ref="$(git -C "$source_dir" rev-parse HEAD)"
index_dir="$(mktemp -d "${TMPDIR:-/tmp}/marathon-patch-index.XXXXXX")"
trap 'rm -f -- "$index_dir/index" "$index_dir/index.lock"; rmdir -- "$index_dir"' EXIT
export GIT_INDEX_FILE="$index_dir/index"
git -C "$source_dir" read-tree "$base_ref"
git -C "$source_dir" apply --cached "$@"
expected_tree="$(git -C "$source_dir" write-tree)"
unset GIT_INDEX_FILE

if [[ ! -e "$target_dir" ]]; then
  mkdir -p "$(dirname "$target_dir")"
  git -C "$source_dir" worktree add --detach "$target_dir" "$base_ref"
fi
if [[ ! -e "$target_dir/.git" ]]; then
  echo "error: existing patch target is not a separate Git worktree: $target_dir" >&2
  exit 1
fi
if [[ "$(git -C "$target_dir" rev-parse HEAD)" != "$base_ref" ]]; then
  echo "error: existing patch worktree has a different base: $target_dir" >&2
  exit 1
fi
if git -C "$target_dir" diff --quiet && [[ "$(git -C "$target_dir" write-tree)" == "$expected_tree" ]]; then
  echo "patches already applied: $target_dir"
  exit 0
fi
if ! git -C "$target_dir" diff HEAD --quiet || [[ -n "$(git -C "$target_dir" ls-files --others --exclude-standard)" ]]; then
  echo "error: preserving modified worktree: $target_dir" >&2
  echo "choose another patched worktree location; no edits were discarded" >&2
  exit 1
fi
git -C "$target_dir" apply --index "$@"
echo "patched source tree: $target_dir"
